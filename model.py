import os
import gc
import random

import torch
from diffusers import (
    DiffusionPipeline,
    IFImg2ImgPipeline,
    IFImg2ImgSuperResolutionPipeline,
    IFInpaintingPipeline,
    IFInpaintingSuperResolutionPipeline,
    IFPipeline,
    IFSuperResolutionPipeline,
)
from diffusers.pipelines.deepfloyd_if.timesteps import *
from diffusers.utils import pt_to_pil
from huggingface_hub import login
from time import time


def flush():
    gc.collect()
    torch.cuda.empty_cache()


class IFModel:
    timesteps = {
        "None": None,
        "fast27": fast27_timesteps,
        "smart27": smart27_timesteps,
        "smart50": smart50_timesteps,
        "smart100": smart100_timesteps,
        "smart185": smart185_timesteps,
        "super27": super27_timesteps,
        "super40": super40_timesteps,
        "super100": super100_timesteps,
    }

    def __init__(
        self,
        stage_1_offload: str = "cpu",
        stage_2_offload: str = "cpu",
        stage_3_offload: str = "cpu",
        torch_compile=False,
        output_folder: str = "./output",
        save_intermediate=False,
        token=None,
        **kwargs,
    ):
        if token is not None:
            login(token=token)
        self.stage_1 = IFPipeline.from_pretrained(
            "DeepFloyd/IF-I-XL-v1.0",
            variant="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            watermarker=None,
        )
        self.stage_2 = IFSuperResolutionPipeline.from_pretrained(
            "DeepFloyd/IF-II-L-v1.0",
            text_encoder=None,
            variant="fp16",
            torch_dtype=torch.float16,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            watermarker=None,
        )
        self.stage_3 = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
        )
        if torch_compile:
            self.stage_1.text_encoder = torch.compile(self.stage_1.text_encoder)
            self.stage_1.unet = torch.compile(self.stage_1.unet)
            self.stage_2.unet = torch.compile(self.stage_2.unet)
        if stage_1_offload == "cpu":
            self.stage_1.enable_model_cpu_offload()
        elif stage_1_offload == "sequential":
            self.stage_1.enable_sequential_cpu_offload()
        if stage_2_offload == "cpu":
            self.stage_2.enable_model_cpu_offload()
        elif stage_2_offload == "sequential":
            self.stage_2.enable_sequential_cpu_offload()
        if stage_3_offload == "cpu":
            self.stage_3.enable_model_cpu_offload()
        elif stage_3_offload == "sequential":
            self.stage_3.enable_sequential_cpu_offload()

        self.img2img_stage_1 = IFImg2ImgPipeline(
            **self.stage_1.components, requires_safety_checker=False
        )
        self.img2img_stage_2 = IFImg2ImgSuperResolutionPipeline(
            **self.stage_2.components, requires_safety_checker=False
        )
        self.inpainting_stage_1 = IFInpaintingPipeline(
            **self.stage_1.components, requires_safety_checker=False
        )
        self.inpainting_stage_2 = IFInpaintingSuperResolutionPipeline(
            **self.stage_2.components, requires_safety_checker=False
        )
        self.save_intermediate = save_intermediate
        self.output_folder = output_folder
        os.makedirs(output_folder, exist_ok=True)

    def encode(self, prompt, negative_prompt, stage_1):
        if prompt == "":
            prompt = None
        if negative_prompt == "":
            negative_prompt = None
        prompt_embeds, negative_embeds = stage_1.encode_prompt(
            prompt, negative_prompt=negative_prompt
        )
        return prompt_embeds, negative_embeds

    def offload(self, pipe):
        # hack to maybe fix memory leak
        stages = {
            "txt2img": (self.stage_1, self.stage_2),
            "img2img": (self.img2img_stage_1, self.img2img_stage_2),
            "inpainting": (self.inpainting_stage_1, self.inpainting_stage_2),
            "stable": (self.stage_3,),
        }
        for stage in stages[pipe]:
            if hasattr(stage, "text_encoder_offload_hook"):
                stage.text_encoder_offload_hook.offload()
            if hasattr(stage, "unet_offload_hook"):
                stage.unet_offload_hook.offload()
            if hasattr(stage, "final_offload_hook"):
                stage.final_offload_hook.offload()

    def offload_all(self):
        # forcing to offload all models fixes high vram usage when switching between pipes
        self.offload("txt2img")
        self.offload("img2img")
        self.offload("inpainting")
        self.offload("stable")

    def save(self, name, images, stage, stages):
        if (stage == stages) or self.save_intermediate:
            for i, img in enumerate(images):
                img.save(f"{self.output_folder}/{name}_{i}_{stage}.png")

    def txt2img_generator(
        self,
        prompt="",
        negative_prompt="",
        seed: int = 0,
        num_images_per_prompt: int = 1,
        stages: int = 3,
        timesteps_1: str = "None",
        num_inference_steps_1: int = 100,
        guidance_scale_1: float = 7.0,
        timesteps_2: str = "None",
        num_inference_steps_2: int = 50,
        guidance_scale_2: float = 4.0,
        num_inference_steps_3: int = 75,
        guidance_scale_3: float = 9.0,
        noise_level_3: float = 100.0,
    ):
        self.offload_all()
        flush()
        # enforce types
        seed = int(seed)
        num_images_per_prompt = int(num_images_per_prompt)
        stages = int(stages)
        num_inference_steps_1 = int(num_inference_steps_1)
        num_inference_steps_2 = int(num_inference_steps_2)
        num_inference_steps_3 = int(num_inference_steps_3)

        # set seed
        seed = random.randint(0, 2**32) if seed == -1 else seed
        generator = torch.manual_seed(seed)
        t = int(time())

        # encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode(
            prompt, negative_prompt, self.stage_1
        )
        self.offload_all()
        flush()
        timesteps_1 = self.timesteps[timesteps_1]
        if timesteps_1 is not None:
            num_inference_steps_1 = len(timesteps_1)

        # run stage 1
        stage_1_output = self.stage_1(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            num_inference_steps=num_inference_steps_1,
            guidance_scale=guidance_scale_1,
            num_images_per_prompt=num_images_per_prompt,
            output_type="pt",
        ).images
        self.offload_all()
        flush()
        output_images = pt_to_pil(stage_1_output)
        self.save(f"{t}", output_images, 1, stages)
        yield output_images
        if stages == 1:
            return

        prompt_embeds = torch.cat([prompt_embeds] * num_images_per_prompt)
        negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds] * num_images_per_prompt
        )
        timesteps_2 = self.timesteps[timesteps_2]
        if timesteps_2 is not None:
            num_inference_steps_2 = len(timesteps_2)

        # run stage 2
        stage_2_output = self.stage_2(
            image=stage_1_output,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            num_inference_steps=num_inference_steps_2,
            guidance_scale=guidance_scale_2,
            output_type="pt",
        ).images
        self.offload_all()
        flush()
        output_images = pt_to_pil(stage_2_output)
        self.save(f"{t}", output_images, 2, stages)
        yield output_images
        if stages == 2:
            return

        # run stage 3
        stage_3_output = self.stage_3(
            image=stage_2_output,
            prompt=[prompt] * num_images_per_prompt,
            negative_prompt=[negative_prompt] * num_images_per_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps_3,
            guidance_scale=guidance_scale_3,
            noise_level=noise_level_3,
        ).images
        self.offload_all()
        flush()
        self.save(f"{t}", stage_3_output, 3, stages)
        yield stage_3_output
        return

    def img2img_generator(
        self,
        image,
        prompt="",
        negative_prompt="",
        seed: int = 0,
        num_images_per_prompt: int = 1,
        stages: int = 3,
        strength: float = 0.8,
        num_inference_steps_3: int = 75,
        guidance_scale_3: float = 9.0,
        noise_level_3: float = 100.0,
    ):
        self.offload_all()
        flush()
        # enforce types
        seed = int(seed)
        num_images_per_prompt = int(num_images_per_prompt)
        stages = int(stages)
        t = int(time())

        # set seed
        seed = random.randint(0, 2**32) if seed == -1 else seed
        generator = torch.manual_seed(seed)

        # encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode(
            prompt, negative_prompt, self.img2img_stage_1
        )
        self.offload_all()
        flush()
        # run stage 1
        stage_1_output = self.img2img_stage_1(
            image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            output_type="pt",
        ).images
        self.offload_all()
        flush()
        output_images = pt_to_pil(stage_1_output)
        self.save(f"{t}", output_images, 1, stages)
        yield output_images
        if stages == 1:
            return

        prompt_embeds = torch.cat([prompt_embeds] * num_images_per_prompt)
        negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds] * num_images_per_prompt
        )
        image = [image] * num_images_per_prompt

        # run stage 2
        stage_2_output = self.img2img_stage_2(
            image=stage_1_output,
            original_image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            strength=strength,
            output_type="pt",
        ).images
        self.offload_all()
        flush()
        output_images = pt_to_pil(stage_2_output)
        self.save(f"{t}", output_images, 2, stages)
        yield output_images
        if stages == 2:
            return

        # run stage 3
        stage_3_output = self.stage_3(
            image=stage_2_output,
            prompt=[prompt] * num_images_per_prompt,
            negative_prompt=[negative_prompt] * num_images_per_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps_3,
            guidance_scale=guidance_scale_3,
            noise_level=noise_level_3,
        ).images
        self.offload_all()
        flush()
        self.save(f"{t}", stage_3_output, 3, stages)
        yield stage_3_output
        return

    def inpainting_generator(
        self,
        image_and_mask,
        prompt="",
        negative_prompt="",
        seed: int = 0,
        num_images_per_prompt: int = 1,
        stages: int = 3,
        strength: float = 0.8,
        num_inference_steps_3: int = 75,
        guidance_scale_3: float = 9.0,
        noise_level_3: float = 100.0,
    ):
        self.offload_all()
        flush()
        image = image_and_mask["image"]
        mask = image_and_mask["mask"]
        # enforce types
        seed = int(seed)
        num_images_per_prompt = int(num_images_per_prompt)
        stages = int(stages)
        t = int(time())

        # set seed
        seed = random.randint(0, 2**32) if seed == -1 else seed
        generator = torch.manual_seed(seed)

        # encode prompt
        prompt_embeds, negative_prompt_embeds = self.encode(
            prompt, negative_prompt, self.inpainting_stage_1
        )
        self.offload_all()
        flush()
        # run stage 1
        stage_1_output = self.inpainting_stage_1(
            image=image,
            mask_image=mask,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            output_type="pt",
        ).images
        self.offload_all()
        flush()
        output_images = pt_to_pil(stage_1_output)
        self.save(f"{t}", output_images, 1, stages)
        yield output_images
        if stages == 1:
            return

        prompt_embeds = torch.cat([prompt_embeds] * num_images_per_prompt)
        negative_prompt_embeds = torch.cat(
            [negative_prompt_embeds] * num_images_per_prompt
        )
        image = [image] * num_images_per_prompt
        # run stage 2
        stage_2_output = self.inpainting_stage_2(
            image=stage_1_output,
            mask_image=mask,
            original_image=image,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            generator=generator,
            strength=strength,
            output_type="pt",
        ).images
        self.offload_all()
        flush()
        output_images = pt_to_pil(stage_2_output)
        self.save(f"{t}", output_images, 2, stages)
        yield output_images
        if stages == 2:
            return

        # run stage 3
        stage_3_output = self.stage_3(
            image=stage_2_output,
            prompt=[prompt] * num_images_per_prompt,
            negative_prompt=[negative_prompt] * num_images_per_prompt,
            generator=generator,
            num_inference_steps=num_inference_steps_3,
            guidance_scale=guidance_scale_3,
            noise_level=noise_level_3,
        ).images
        self.offload_all()
        flush()
        self.save(f"{t}", stage_3_output, 3, stages)
        yield stage_3_output
        return
