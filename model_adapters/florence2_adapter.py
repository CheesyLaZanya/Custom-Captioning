# Standard library imports
from typing import List

# Third-party library imports
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

# https://huggingface.co/microsoft/Florence-2-large-ft

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32


def load_model(model_name: str) -> tuple[AutoModelForCausalLM, AutoProcessor]:
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True
    ).to(device).eval()
    return model, processor


def generate_caption(model: AutoModelForCausalLM, processor: AutoProcessor, image: Image.Image, prompt: str) -> str:
    if image.mode != "RGB":
        image = image.convert("RGB")

    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        num_beams=3
    ).to(device)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    processed_output = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))
    caption = processed_output.get(prompt, '')

    return caption


def get_suggested_models() -> List[str]:
    suggested_models = [
        'microsoft/Florence-2-large-ft'
    ]
    return suggested_models


def get_suggested_prompts() -> List[str]:
    suggested_prompts = [
        '<CAPTION>',
        '<DETAILED_CAPTION>',
        '<MORE_DETAILED_CAPTION>',
        '<OCR>'
    ]
    return suggested_prompts
