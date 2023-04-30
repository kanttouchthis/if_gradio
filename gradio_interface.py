import argparse
import os
import random
from time import time

import gradio as gr
import torch
from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
from huggingface_hub import login
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu_offload", action="store_true", help="Use CPU offload")
    parser.add_argument(
        "--xformers",
        action="store_true",
        help="Use XFormers memory efficient attention",
    )
    return parser.parse_args()


def setup(cpu_offload=False, xformers_mea=False):
    stage_1 = DiffusionPipeline.from_pretrained(
        "DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16
    )
    stage_2 = DiffusionPipeline.from_pretrained(
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


def embed(prompt, negative_prompt, stage_1):
    if prompt == "":
        prompt = None
    if negative_prompt == "":
        negative_prompt = None
    prompt_embeds, negative_embeds = stage_1.encode_prompt(
        prompt, negative_prompt=negative_prompt
    )
    return prompt_embeds, negative_embeds


def interface_iterative(prompt, negative_prompt, seed, stages):
    global stage_1, stage_2, stage_3
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
    image1 = pt_to_pil(image1)[0]
    image1.save(f"outputs/{t}_1.png")
    yield image1.resize((1024, 1024), resample=Image.NEAREST)
    if stages == 1:
        return
    image2 = stage_2(
        image=image1,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_embeds,
        generator=generator,
        output_type="pt",
    ).images
    image2 = pt_to_pil(image2)[0]
    image2.save(f"outputs/{t}_2.png")
    yield image2.resize((1024, 1024), resample=Image.NEAREST)
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
    demo = gr.Interface(
        fn=interface_iterative,
        inputs=["text", "text", "number", gr.Slider(1, 3, 3, step=1)],
        outputs=["image"],
        title="IF",
        description="Generate an image from text using IF.",
    )
    demo.queue(concurrency_count=1)
    demo.launch()
