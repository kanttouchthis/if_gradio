import argparse
import os
import random
import json
from time import time

import gradio as gr
import torch
from diffusers import IFPipeline, IFSuperResolutionPipeline
from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline
from diffusers import IFInpaintingPipeline, IFInpaintingSuperResolutionPipeline
from diffusers import DiffusionPipeline
from diffusers.pipelines.deepfloyd_if.timesteps import *
from diffusers.utils import pt_to_pil
from huggingface_hub import login
from PIL import Image

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


def setup(cpu_offload=True, xformers_mea=False):
    stage_1 = IFPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0",
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        watermarker=None,
    )
    stage_2 = IFSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
        watermarker=None,
    )
    stage_3 = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16
    )
    if cpu_offload:
        stage_1.enable_model_cpu_offload()
        stage_2.enable_model_cpu_offload()
        stage_3.enable_model_cpu_offload()
    if xformers_mea:
        stage_1.enable_xformers_memory_efficient_attention()
        stage_2.enable_xformers_memory_efficient_attention()
        stage_3.enable_xformers_memory_efficient_attention()
    os.makedirs("outputs", exist_ok=True)
    return stage_1, stage_2, stage_3


def convert_pipe(pipe_type="txt2img"):
    global stage_1, stage_2
    if pipe_type == "txt2img" and not isinstance(stage_1, IFPipeline):
        print("Converting to txt2img")
        stage_1 = IFPipeline(**stage_1.components)
        stage_2 = IFSuperResolutionPipeline(**stage_2.components)
    elif pipe_type == "img2img" and not isinstance(stage_1, IFImg2ImgPipeline):
        print("Converting to img2img")
        stage_1 = IFImg2ImgPipeline(**stage_1.components)
        stage_2 = IFImg2ImgSuperResolutionPipeline(**stage_2.components)
    elif pipe_type == "inpainting" and not isinstance(stage_1, IFInpaintingPipeline):
        print("Converting to inpainting")
        stage_1 = IFInpaintingPipeline(**stage_1.components)
        stage_2 = IFInpaintingSuperResolutionPipeline(**stage_2.components)
    else:
        return


def resize_with_aspect_ratio(image, size, resample=Image.NEAREST):
    w, h = image.size
    if w > h:
        new_w = size
        new_h = int(h * (size / w))
    else:
        new_w = int(w * (size / h))
        new_h = size
    return image.resize((new_w, new_h), resample=resample)


def embed(prompt, negative_prompt, stage_1):
    if prompt == "":
        prompt = None
    if negative_prompt == "":
        negative_prompt = None
    prompt_embeds, negative_embeds = stage_1.encode_prompt(
        prompt, negative_prompt=negative_prompt
    )
    return prompt_embeds, negative_embeds


def txt2img_interface(
    prompt,
    negative_prompt,
    seed,
    batch_size,
    stages,
    stage_1_timesteps,
    stage_1_steps,
    stage_1_guidance_scale,
    stage_2_timesteps,
    stage_2_steps,
    stage_2_guidance_scale,
    stage_3_steps,
    stage_3_guidance_scale,
    stage_3_noise_level,
):
    convert_pipe("txt2img")
    global stage_1, stage_2, stage_3
    t = int(time())
    stages = int(stages)
    seed = random.randint(0, 2**32) if seed == -1 else int(seed)
    generator = torch.manual_seed(seed)
    stage_1_timesteps = timesteps[stage_1_timesteps]
    stage_2_timesteps = timesteps[stage_2_timesteps]
    if stage_1_timesteps is not None:
        stage_1_steps = len(stage_1_timesteps)
    if stage_2_timesteps is not None:
        stage_2_steps = len(stage_2_timesteps)
    # embed
    prompt_embeds, negative_embeds = embed(prompt, negative_prompt, stage_1)

    # stage 1
    images = []
    image1 = stage_1(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        timesteps=stage_1_timesteps,
        num_inference_steps=int(stage_1_steps),
        guidance_scale=stage_1_guidance_scale,
        num_images_per_prompt=int(batch_size),
        output_type="pt",
    ).images
    for i, img in enumerate(pt_to_pil(image1)):
        img.save(f"outputs/{t}_1_{i}.png")
        images.append((img, f"prompt:'{prompt}' seed:{seed} n:{i}"))
    yield images
    if stages == 1:
        return

    prompt_embeds = torch.cat([prompt_embeds] * batch_size)
    negative_embeds = torch.cat([negative_embeds] * batch_size)
    # stage 2
    images = []
    image2 = stage_2(
        image=image1,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        timesteps=stage_2_timesteps,
        num_inference_steps=int(stage_2_steps),
        guidance_scale=stage_2_guidance_scale,
        output_type="pt",
    ).images
    for i, img in enumerate(pt_to_pil(image2)):
        img.save(f"outputs/{t}_2_{i}.png")
        images.append((img, f"prompt:'{prompt}' seed:{seed} n:{i}"))
    yield images
    if stages == 2:
        return

    # stage 3
    images = []
    image3 = stage_3(
        prompt=[prompt] * batch_size,
        image=image2,
        generator=generator,
        num_inference_steps=int(stage_3_steps),
        guidance_scale=stage_3_guidance_scale,
        noise_level=stage_3_noise_level,
    ).images
    for i, img in enumerate(image3):
        img.save(f"outputs/{t}_3_{i}.png")
        images.append((img, f"prompt:'{prompt}' seed:{seed} n:{i}"))
    yield images


def img2img_interface(image, prompt, negative_prompt, seed, stages, strength):
    convert_pipe("img2img")
    global stage_1, stage_2, stage_3
    t = int(time())
    stages = int(stages)
    seed = random.randint(0, 2**32) if seed == -1 else seed
    generator = torch.manual_seed(seed)
    image = Image.fromarray(image.astype("uint8"), "RGB")
    # embed
    prompt_embeds, negative_embeds = embed(prompt, negative_prompt, stage_1)

    # stage 1
    image1 = stage_1(
        image=image,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        strength=strength,
        generator=generator,
        output_type="pt",
    ).images
    image1_out = pt_to_pil(image1)[0]
    image1_out.save(f"outputs/{t}_1.png")
    yield resize_with_aspect_ratio(image1_out, 1024)
    if stages == 1:
        return

    # stage 2
    image2 = stage_2(
        image=image1,
        original_image=image,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        strength=strength,
        generator=generator,
        output_type="pt",
    ).images
    image2_out = pt_to_pil(image2)[0]
    image2_out.save(f"outputs/{t}_2.png")
    yield resize_with_aspect_ratio(image2_out, 1024)
    if stages == 2:
        return

    # stage 3
    image3 = stage_3(
        prompt=prompt, image=image2, generator=generator, noise_level=100
    ).images
    image3[0].save(f"outputs/{t}_3.png")
    yield image3[0]


def inpainting_interface(
    image_and_mask, prompt, negative_prompt, seed, stages, strength
):
    convert_pipe("inpainting")
    global stage_1, stage_2, stage_3
    t = int(time())
    stages = int(stages)
    seed = random.randint(0, 2**32) if seed == -1 else seed
    generator = torch.manual_seed(seed)
    image = Image.fromarray(image_and_mask["image"].astype("uint8"), "RGB")
    mask = Image.fromarray(image_and_mask["mask"].astype("uint8"), "RGBA")
    # embed
    prompt_embeds, negative_embeds = embed(prompt, negative_prompt, stage_1)

    # stage 1
    image1 = stage_1(
        image=image,
        mask_image=mask,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        strength=strength,
        generator=generator,
        output_type="pt",
    ).images
    image1_out = pt_to_pil(image1)[0]
    image1_out.save(f"outputs/{t}_1.png")
    yield resize_with_aspect_ratio(image1_out, 1024)
    if stages == 1:
        return

    # stage 2
    image2 = stage_2(
        image=image1,
        mask_image=mask,
        original_image=image,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        strength=strength,
        generator=generator,
        output_type="pt",
    ).images
    image2_out = pt_to_pil(image2)[0]
    image2_out.save(f"outputs/{t}_2.png")
    yield resize_with_aspect_ratio(image2_out, 1024)
    if stages == 2:
        return

    # stage 3
    image3 = stage_3(
        prompt=prompt, image=image2, generator=generator, noise_level=100
    ).images
    image3[0].save(f"outputs/{t}_3.png")
    yield image3[0]


if __name__ == "__main__":
    config = json.loads(open("config.json", "r").read())
    if not config["ui_test"]:
        login(open("key.txt", "r").read())
        stage_1, stage_2, stage_3 = setup(
            cpu_offload=config["cpu_offload"], xformers_mea=config["xformers_mea"]
        )
    with gr.Blocks() as demo:
        gr.Markdown("deep-floyd IF")
        with gr.Tab("txt2img"):
            with gr.Row():
                with gr.Column():
                    txt2img_prompt = gr.Textbox(label="Prompt")
                    txt2img_negative_prompt = gr.Textbox(label="Negative Prompt")
                    txt2img_seed = gr.Number(label="Seed", default=0)
                    txt2img_batch_size = gr.Slider(
                        1, 9, 1, step=1, label="Images per batch"
                    )
                    txt2img_stages = gr.Slider(1, 3, 3, step=1, label="Stages")
                    txt2img_button = gr.Button("Generate")
                    with gr.Accordion("Stage 1", open=False):
                        txt2img_stage_1_timesteps = gr.Radio(
                            choices=[
                                "None",
                                "fast27",
                                "smart27",
                                "smart50",
                                "smart100",
                                "smart185",
                                "super27",
                                "super40",
                                "super100",
                            ],
                            value="None",
                            label="Timesteps",
                        )
                        txt2img_stage_1_steps = gr.Slider(
                            1, 200, 100, step=1, label="Steps"
                        )
                        txt2img_stage_1_guidance_scale = gr.Slider(
                            0, 30, 7, step=0.1, label="Guidance Scale"
                        )
                    with gr.Accordion("Stage 2", open=False):
                        txt2img_stage_2_timesteps = gr.Radio(
                            choices=[
                                "None",
                                "fast27",
                                "smart27",
                                "smart50",
                                "smart100",
                                "smart185",
                                "super27",
                                "super40",
                                "super100",
                            ],
                            value="None",
                            label="Timesteps",
                        )
                        txt2img_stage_2_steps = gr.Slider(
                            1, 100, 50, step=1, label="Steps"
                        )
                        txt2img_stage_2_guidance_scale = gr.Slider(
                            0, 30, 4, step=0.1, label="Guidance Scale"
                        )
                    with gr.Accordion("Stage 3", open=False):
                        txt2img_stage_3_steps = gr.Slider(
                            1, 100, 75, step=1, label="Steps"
                        )
                        txt2img_stage_3_guidance_scale = gr.Slider(
                            0, 30, 9, step=0.1, label="Guidance Scale"
                        )
                        txt2img_stage_3_noise_level = gr.Slider(
                            0, 100, 100, step=1, label="Noise Level"
                        )
                with gr.Column():
                    txt2img_output = gr.Gallery()
                    txt2img_output.style(columns=[3], rows=[3], object_fit="contain", height="auto")

        with gr.Tab("img2img"):
            with gr.Row():
                with gr.Column():
                    img2img_image = gr.Image()
                    img2img_button = gr.Button("Generate")
                    img2img_prompt = gr.Textbox(label="Prompt")
                    img2img_negative_prompt = gr.Textbox(label="Negative Prompt")
                    img2img_seed = gr.Number(label="Seed", default=0)
                    img2img_stages = gr.Slider(1, 3, 3, step=1, label="Stages")
                    img2img_strength = gr.Slider(
                        0,
                        1,
                        0.8,
                        step=0.01,
                        label="Strength",
                    )
                with gr.Column():
                    img2img_output = gr.Image()

        with gr.Tab("inpainting"):
            with gr.Row():
                with gr.Column():
                    inpainting_image = gr.ImageMask()
                    inpainting_button = gr.Button("Generate")
                    inpainting_prompt = gr.Textbox(label="Prompt")
                    inpainting_negative_prompt = gr.Textbox(label="Negative Prompt")
                    inpainting_seed = gr.Number(label="Seed", default=0)
                    inpainting_stages = gr.Slider(1, 3, 3, step=1, label="Stages")
                    inpainting_strength = gr.Slider(
                        0,
                        1,
                        0.8,
                        step=0.01,
                        label="Strength",
                    )
                with gr.Column():
                    inpainting_output = gr.Image()

        txt2img_button.click(
            txt2img_interface,
            inputs=[
                txt2img_prompt,
                txt2img_negative_prompt,
                txt2img_seed,
                txt2img_batch_size,
                txt2img_stages,
                txt2img_stage_1_timesteps,
                txt2img_stage_1_steps,
                txt2img_stage_1_guidance_scale,
                txt2img_stage_2_timesteps,
                txt2img_stage_2_steps,
                txt2img_stage_2_guidance_scale,
                txt2img_stage_3_steps,
                txt2img_stage_3_guidance_scale,
                txt2img_stage_3_noise_level,
            ],
            outputs=txt2img_output,
        )
        img2img_button.click(
            img2img_interface,
            inputs=[
                img2img_image,
                img2img_prompt,
                img2img_negative_prompt,
                img2img_seed,
                img2img_stages,
                img2img_strength,
            ],
            outputs=img2img_output,
        )

        inpainting_button.click(
            inpainting_interface,
            inputs=[
                inpainting_image,
                inpainting_prompt,
                inpainting_negative_prompt,
                inpainting_seed,
                inpainting_stages,
                inpainting_strength,
            ],
            outputs=inpainting_output,
        )
demo.queue()
demo.launch()
