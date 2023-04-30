import argparse
import os
import random
from time import time

import gradio as gr
import torch
from diffusers import IFPipeline, IFSuperResolutionPipeline
from diffusers import IFImg2ImgPipeline, IFImg2ImgSuperResolutionPipeline
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from huggingface_hub import login
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cpu_offload", action="store_true", help="Disable CPU offload"
    )
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="Use XFormers memory efficient attention",
    )
    return parser.parse_args()


def setup(cpu_offload=True, xformers_mea=False):
    stage_1 = IFPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
    )
    stage_2 = IFSuperResolutionPipeline.from_pretrained(
        "DeepFloyd/IF-II-L-v1.0",
        text_encoder=None,
        variant="fp16",
        torch_dtype=torch.float16,
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


def txt2img_interface(prompt, negative_prompt, seed, stages):
    global stage_1, stage_2, stage_3
    convert_pipe("txt2img")
    t = int(time())
    stages = int(stages)
    prompt_embeds, negative_embeds = embed(prompt, negative_prompt, stage_1)
    seed = random.randint(0, 2**32) if seed == -1 else seed
    generator = torch.manual_seed(seed)
    image1 = stage_1(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        output_type="pt",
    ).images
    image1_out = pt_to_pil(image1)[0]
    image1_out.save(f"outputs/{t}_1.png")
    yield resize_with_aspect_ratio(image1_out, 1024)
    if stages == 1:
        return
    image2 = stage_2(
        image=image1,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        output_type="pt",
    ).images
    image2_out = pt_to_pil(image2)[0]
    image2_out.save(f"outputs/{t}_2.png")
    yield resize_with_aspect_ratio(image2_out, 1024)
    if stages == 2:
        return
    image3 = stage_3(
        prompt=prompt, image=image2, generator=generator, noise_level=100
    ).images
    image3[0].save(f"outputs/{t}_3.png")
    yield image3[0]


def img2img_interface(image, prompt, negative_prompt, seed, stages, strength):
    convert_pipe("img2img")
    global stage_1, stage_2, stage_3
    image = Image.fromarray(image.astype("uint8"), "RGB")
    t = int(time())
    stages = int(stages)
    prompt_embeds, negative_embeds = embed(prompt, negative_prompt, stage_1)
    seed = random.randint(0, 2**32) if seed == -1 else seed
    generator = torch.manual_seed(seed)
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
    image3 = stage_3(
        prompt=prompt, image=image2, generator=generator, noise_level=100
    ).images
    image3[0].save(f"outputs/{t}_3.png")
    yield image3[0]


if __name__ == "__main__":
    args = get_args()
    login(open("key.txt", "r").read())
    stage_1, stage_2, stage_3 = setup(
        cpu_offload=args.cpu_offload, xformers_mea=args.xformers
    )
    with gr.Blocks() as demo:
        gr.Markdown("deep-floyd IF")
        with gr.Tab("txt2img"):
            with gr.Row():
                with gr.Column():
                    txt2img_prompt = gr.Textbox(label="Prompt")
                    txt2img_negative_prompt = gr.Textbox(label="Negative Prompt")
                    txt2img_seed = gr.Number(label="Seed", default=0)
                    txt2img_stages = gr.Slider(1, 3, 3, step=1, label="Stages")
                    txt2img_button = gr.Button("Generate")
                with gr.Column():
                    txt2img_output = gr.Image()

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

        txt2img_button.click(
            txt2img_interface,
            inputs=[
                txt2img_prompt,
                txt2img_negative_prompt,
                txt2img_seed,
                txt2img_stages,
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
demo.queue()
demo.launch()
