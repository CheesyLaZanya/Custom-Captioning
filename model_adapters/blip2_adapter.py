# Standard library imports
from typing import List

# Third-party library imports
from PIL import Image
import torch
from transformers import AutoProcessor, Blip2ForConditionalGeneration


def load_model(model_name: str) -> tuple[Blip2ForConditionalGeneration, AutoProcessor]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device).eval()
    return model, processor


def generate_annotation(model: Blip2ForConditionalGeneration, processor: AutoProcessor, image: Image.Image, prompt: str) -> str:
    device = next(model.parameters()).device

    # Prepare the prompt using the appropriate template
    full_prompt = f"Question: {prompt} Answer:"

    print(f"Running prompt: {full_prompt}")

    image = image.convert('RGB')

    inputs = processor(image, text=full_prompt, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=100)
    annotation = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # Remove the initial prompt from the annotation
    annotation = annotation.replace(full_prompt, "").strip()

    return annotation.strip()


def get_suggested_models() -> List[str]:
    suggested_models = [
        'Salesforce/blip2-opt-2.7b'
    ]
    return suggested_models


def get_suggested_prompts() -> List[str]:
    suggested_prompts = [
        'What is the content of this image?'
    ]
    return suggested_prompts
