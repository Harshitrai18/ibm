import matplotlib.pyplot as plt
from diffusers import StableDiffusionPipeline
import torch
import gradio as gr

from huggingface_hub import login
login("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")

# Check for GPU and set device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32, # Use float16 for GPU, float32 for CPU
    use_safetensors=True
)
pipe = pipe.to(device)

def generate_image(prompt):
    image = pipe(prompt).images[0]
    return image

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Image(label="Generated Image"),
    title="Stable Diffusion Image Generator"
)

# Launch the Gradio interface
iface.launch(debug=True)