from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil
import torch
import time
import os
import random
from huggingface_hub import login

os.environ['FORCE_MEM_EFFICIENT_ATTN'] = "1"
login(open("key.txt", "r").read())

#torch.cuda.set_per_process_memory_fraction(11/24, device="cuda:0")
memory_efficient = False 
cpu_offload = True
save_intermediate = True

stage_1 = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0", variant="fp16", torch_dtype=torch.float16)
if memory_efficient:
    stage_1.enable_xformers_memory_efficient_attention()
if cpu_offload:
    stage_1.enable_model_cpu_offload()

stage_2 = DiffusionPipeline.from_pretrained(
    "DeepFloyd/IF-II-L-v1.0", text_encoder=None, variant="fp16", torch_dtype=torch.float16
)
if memory_efficient:
    stage_2.enable_xformers_memory_efficient_attention() 
if cpu_offload:
    stage_2.enable_model_cpu_offload()

safety_modules = {"feature_extractor": stage_1.feature_extractor, "safety_checker": stage_1.safety_checker, "watermarker": stage_1.watermarker}
stage_3 = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-x4-upscaler", torch_dtype=torch.float16)# , **safety_modules
if memory_efficient:
    stage_3.enable_xformers_memory_efficient_attention()
if cpu_offload:
    stage_3.enable_model_cpu_offload()

def generate(prompt = 'a photo of a kangaroo wearing an orange hoodie and blue sunglasses standing in front of the eiffel tower holding a sign that says "very deep learning"',
            negative_prompt = None,
            seed=0,
            output_path="."
            ):
    os.makedirs(output_path, exist_ok=True)
    generator = torch.manual_seed(seed)
    prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt, negative_prompt=negative_prompt)
    t = int(time.time())
    # stage 1
    image = stage_1(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt").images
    if save_intermediate:
        pt_to_pil(image)[0].save(f"{output_path}/{t}_if_stage_I.png")

    # stage 2
    image = stage_2(
        image=image, prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, generator=generator, output_type="pt"
    ).images
    if save_intermediate:
        pt_to_pil(image)[0].save(f"{output_path}/{t}_if_stage_II.png")

    # stage 3
    image = stage_3(prompt=prompt, image=image, generator=generator, noise_level=100).images
    image[0].save(f"{output_path}/{t}_if_stage_III.png")

if __name__ == "__main__":
    print("\n")
    while True:
        try:
            prompt = input("prompt: ")
            negative_prompt = input("negative prompt: ")
            seed = input("seed: ")
            if negative_prompt == "":
                negative_prompt = None
            if seed == "":
                seed = random.randint(0, 10000000)
            else:
                seed = int(seed)
            generate(prompt, negative_prompt=negative_prompt, seed=seed, output_path="./results")
        except KeyboardInterrupt:
            exit()
        except Exception as e:
            print(e)
            continue