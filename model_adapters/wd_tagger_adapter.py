# Standard library imports
import csv
from pathlib import Path
from typing import List

# Third-party library imports
import huggingface_hub
import numpy as np
from onnxruntime import InferenceSession
from PIL import Image


class WdTaggerProcessor:
    def __init__(self, model_id: str):
        tags_path = Path(model_id) / 'selected_tags.csv'
        if not tags_path.is_file():
            tags_path = huggingface_hub.hf_hub_download(model_id, filename='selected_tags.csv')
        self.tags = []
        with open(tags_path, 'r') as tags_file:
            reader = csv.DictReader(tags_file)
            for line in reader:
                tag = line['name']
                if tag not in ['0_0', '(o)_(o)', '+_+', '+_-', '._.', '<o>_<o>', '<|>_<|>', '=_=', '>_<', '3_3', '6_9', '>_o', '@_@', '^_^', 'o_o', 'u_u', 'x_x', '|_|', '||_||']:
                    tag = tag.replace('_', ' ')
                self.tags.append(tag)

    def preprocess_image(self, image: Image.Image):
        # Ensure the image is in RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Pad the image to make it square
        max_dimension = max(image.size)
        canvas = Image.new('RGB', (max_dimension, max_dimension), (255, 255, 255))
        horizontal_padding = (max_dimension - image.width) // 2
        vertical_padding = (max_dimension - image.height) // 2
        canvas.paste(image, (horizontal_padding, vertical_padding))

        # Resize the image to 448x448 (typical input size for WD Tagger)
        image = canvas.resize((448, 448), resample=Image.Resampling.BICUBIC)

        # Convert the image to a numpy array
        image_array = np.array(image, dtype=np.float32)
        image_array = image_array[:, :, ::-1]  # Reverse the order of the color channels
        image_array = np.expand_dims(image_array, axis=0)  # Add a batch dimension

        return image_array


def load_model(model_name: str):
    model_path = Path(model_name) / 'model.onnx'
    if not model_path.is_file():
        model_path = huggingface_hub.hf_hub_download(model_name, filename='model.onnx')
    model = InferenceSession(model_path)
    processor = WdTaggerProcessor(model_name)
    return model, processor


def generate_caption(model: InferenceSession, processor: WdTaggerProcessor, image: Image.Image, prompt: str) -> str:
    # Preprocess the image
    image_array = processor.preprocess_image(image)

    # Run inference
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    probabilities = model.run([output_name], {input_name: image_array})[0][0]

    # Get top tags
    top_indices = probabilities.argsort()[-10:][::-1]  # Get indices of top 10 probabilities
    top_tags = [processor.tags[i] for i in top_indices]
    top_probs = probabilities[top_indices]

    # Format the output
    caption = ", ".join([f"{tag} ({prob:.2f})" for tag, prob in zip(top_tags, top_probs)])
    return caption


def get_suggested_models() -> List[str]:
    suggested_models = [
        'SmilingWolf/wd-vit-tagger-v3'
    ]
    return suggested_models


def get_suggested_prompts() -> List[str]:
    suggested_prompts = [
        ''
    ]
    return suggested_prompts
