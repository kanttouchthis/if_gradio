import gradio as gr


def setup(model):
    with gr.Blocks() as demo:
        gr.Markdown("deep-floyd IF")
        with gr.Tab("txt2img"):
            with gr.Row():
                with gr.Column():
                    prompt = gr.Textbox(label="prompt")
                    negative_prompt = gr.Textbox(label="negative prompt")
                    seed = gr.Number(value=0, label="seed")
                    num_images_per_prompt = gr.Number(
                        value=1, label="num images per prompt"
                    )
                    stages = gr.Slider(
                        minimum=1, maximum=3, value=3, step=1, label="stages"
                    )
                    with gr.Accordion("Stage 1", open=False):
                        timesteps_1 = gr.Dropdown(
                            [
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
                            label="timesteps",
                            value="None",
                        )
                        num_inference_steps_1 = gr.Slider(
                            minimum=1,
                            maximum=200,
                            value=100,
                            step=1,
                            label="num inference steps",
                        )
                        guidance_scale_1 = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=7.0,
                            step=0.1,
                            label="guidance scale",
                        )
                    with gr.Accordion("Stage 2", open=False):
                        timesteps_2 = gr.Dropdown(
                            [
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
                            label="timesteps",
                            value="None",
                        )
                        num_inference_steps_2 = gr.Slider(
                            minimum=1,
                            maximum=100,
                            value=50,
                            step=1,
                            label="num inference steps",
                        )
                        guidance_scale_2 = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=4.0,
                            step=0.1,
                            label="guidance scale",
                        )
                    with gr.Accordion("Stage 3", open=False):
                        num_inference_steps_3 = gr.Slider(
                            minimum=1,
                            maximum=150,
                            value=75,
                            step=1,
                            label="num inference steps",
                        )
                        guidance_scale_3 = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=9.0,
                            step=0.1,
                            label="guidance scale",
                        )
                        noise_level_3 = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=100,
                            step=1,
                            label="noise level",
                        )
                with gr.Column():
                    output = gr.Gallery().style(
                        columns=[3], rows=[3], object_fit="contain", height="auto"
                    )
                    run_button = gr.Button(value="Generate")
                    run_button.click(
                        model.txt2img_generator,
                        inputs=[
                            prompt,
                            negative_prompt,
                            seed,
                            num_images_per_prompt,
                            stages,
                            timesteps_1,
                            num_inference_steps_1,
                            guidance_scale_1,
                            timesteps_2,
                            num_inference_steps_2,
                            guidance_scale_2,
                            num_inference_steps_3,
                            guidance_scale_3,
                            noise_level_3,
                        ],
                        outputs=[output],
                    )
        with gr.Tab("img2img"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label="image", type="pil")
                    prompt = gr.Textbox(label="prompt")
                    negative_prompt = gr.Textbox(label="negative prompt")
                    seed = gr.Number(value=0, label="seed")
                    num_images_per_prompt = gr.Number(
                        value=1, label="num images per prompt"
                    )
                    stages = gr.Slider(
                        minimum=1, maximum=3, value=3, step=1, label="stages"
                    )
                    strength = gr.Slider(
                        minimum=0, maximum=1, value=0.8, step=0.01, label="strength"
                    )
                    with gr.Accordion("Stage 3", open=False):
                        num_inference_steps_3 = gr.Slider(
                            minimum=1,
                            maximum=150,
                            value=75,
                            step=1,
                            label="num inference steps",
                        )
                        guidance_scale_3 = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=9.0,
                            step=0.1,
                            label="guidance scale",
                        )
                        noise_level_3 = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=100,
                            step=1,
                            label="noise level",
                        )
                with gr.Column():
                    output = gr.Gallery().style(
                        columns=[3], rows=[3], object_fit="contain", height="auto"
                    )
                    run_button = gr.Button(value="Generate")
                    run_button.click(
                        model.img2img_generator,
                        inputs=[
                            image,
                            prompt,
                            negative_prompt,
                            seed,
                            num_images_per_prompt,
                            stages,
                            strength,
                            num_inference_steps_3,
                            guidance_scale_3,
                            noise_level_3,
                        ],
                        outputs=[output],
                    )
        with gr.Tab("inpainting"):
            with gr.Row():
                with gr.Column():
                    image = gr.Image(label="image", tool="sketch", type="pil")
                    prompt = gr.Textbox(label="prompt")
                    negative_prompt = gr.Textbox(label="negative prompt")
                    seed = gr.Number(value=0, label="seed")
                    num_images_per_prompt = gr.Number(
                        value=1, label="num images per prompt"
                    )
                    stages = gr.Slider(
                        minimum=1, maximum=3, value=3, step=1, label="stages"
                    )
                    strength = gr.Slider(
                        minimum=0, maximum=1, value=0.8, step=0.01, label="strength"
                    )
                    with gr.Accordion("Stage 3", open=False):
                        num_inference_steps_3 = gr.Slider(
                            minimum=1,
                            maximum=150,
                            value=75,
                            step=1,
                            label="num inference steps",
                        )
                        guidance_scale_3 = gr.Slider(
                            minimum=0,
                            maximum=30,
                            value=9.0,
                            step=0.1,
                            label="guidance scale",
                        )
                        noise_level_3 = gr.Slider(
                            minimum=0,
                            maximum=100,
                            value=100,
                            step=1,
                            label="noise level",
                        )
                with gr.Column():
                    output = gr.Gallery().style(
                        columns=[3], rows=[3], object_fit="contain", height="auto"
                    )
                    run_button = gr.Button(value="Generate")
                    run_button.click(
                        model.inpainting_generator,
                        inputs=[
                            image,
                            prompt,
                            negative_prompt,
                            seed,
                            num_images_per_prompt,
                            stages,
                            strength,
                            num_inference_steps_3,
                            guidance_scale_3,
                            noise_level_3,
                        ],
                        outputs=[output],
                    )
    return demo


if __name__ == "__main__":
    demo = setup()
    demo.queue()
    demo.launch()
