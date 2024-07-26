# Standard library imports
import csv
import importlib
import io
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union
from urllib.parse import urlparse

# Third-party library imports
import requests
import torch
from datasets import load_dataset, load_dataset_builder
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer


def get_available_model_types():
    """
    Discover and return a list of available model types based on adapter files.

    This function scans the 'model_adapters' directory (located in the same
    directory as the script) for Python files matching the pattern '*_adapter.py'.
    It extracts the model type from each file name by removing the '_adapter' suffix.

    Returns:
        List[str]: A list of available model types, derived from the names of
                   the adapter files in the 'model_adapters' directory.

    Note:
        - The function assumes that each '*_adapter.py' file in the 'model_adapters'
          directory corresponds to a distinct model type.
        - The model type is inferred from the file name by removing the '_adapter' suffix.
          For example, 'florence2_adapter.py' would yield 'florence2' as a model type.
        - If no adapter files are found, an empty list is returned.

    Raises:
        OSError: If there's an error accessing the 'model_adapters' directory.
    """

    adapter_dir = Path(__file__).parent / "model_adapters"
    model_types = []
    for file in adapter_dir.glob("*_adapter.py"):
        model_type = file.stem.replace("_adapter", "")
        model_types.append(model_type)
    return model_types


def load_model(model_name: str, model_type: str) -> Tuple[torch.nn.Module, Union[AutoProcessor, AutoTokenizer]]:
    """
    Load a model and its processor based on the given model name and type.

    Args:
        model_name (str): The name or path of the model to load.
        model_type (str): The type of the model (e.g., 'florence2', 'blip2').

    Returns:
        Tuple[torch.nn.Module, Union[AutoProcessor, AutoTokenizer]]: The loaded model and its processor.

    Raises:
        ValueError: If the model type is unsupported.
    """

    try:
        print(f"Attempting to load {model_type} type model {model_name}...")
        adapter = importlib.import_module(f"model_adapters.{model_type}_adapter")
        return adapter.load_model(model_name)
    except ImportError:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_suggested_models(model_type: str) -> List[str]:
    """
    Get a list of suggested models for the given model type.

    Args:
        model_type (str): The type of the model.

    Returns:
        List[str]: A list of suggested model names.

    Raises:
        ValueError: If the model type is unsupported.
    """

    try:
        print(f"Attempting to load {model_type} type suggested models...")
        adapter = importlib.import_module(f"model_adapters.{model_type}_adapter")
        return adapter.get_suggested_models()
    except ImportError:
        raise ValueError(f"Unsupported model type: {model_type}")


def get_suggested_prompts(model_type: str) -> List[str]:
    """
    Get a list of suggested prompts for the given model type.

    Args:
        model_type (str): The type of the model.

    Returns:
        List[str]: A list of suggested prompts.

    Raises:
        ValueError: If the model type is unsupported.
    """

    try:
        print(f"Attempting to load {model_type} type suggested prompts...")
        adapter = importlib.import_module(f"model_adapters.{model_type}_adapter")
        return adapter.get_suggested_prompts()
    except ImportError:
        raise ValueError(f"Unsupported model type: {model_type}")


def load_hugging_face_images(input: str, maximum_images: int = 0) -> List[Tuple[Image.Image, str]]:
    """
    Load images from a Hugging Face dataset.

    This function takes a Hugging Face dataset name as input and returns a list of tuples,
    where each tuple contains an image and its identifier.

    Args:
        input (str): The name of the Hugging Face dataset, prefixed with 'hf://'.
        maximum_images (int, optional): The maximum number of images to load. If 0 or negative,
                                        all images in the dataset will be loaded. Defaults to 0.

    Returns:
        List[Tuple[Image.Image, str]]: A list of tuples, where each tuple contains:
            - Image.Image: The loaded image.
            - str: The image identifier (either 'image_{id}' if 'id' exists, or 'unknown').

    Raises:
        ValueError: If the specified dataset does not contain an 'image' column.

    Note:
        This function assumes that the dataset has a 'train' split and an 'image' column.
    """
    random_selection = True
    # stream = True
    stream = False

    images = []
    print(f"Loading Hugging Face Dataset from {input}")
    dataset_name = input[5:]
    dataset = load_dataset(dataset_name, split='train', streaming=stream)

    if 'image' not in dataset.features:
        raise ValueError(f"Dataset {dataset_name} does not contain an 'image' column")

    if stream is False:
        dataset = dataset.to_iterable_dataset()

    if random_selection:
        dataset = dataset.shuffle(seed=42)

    if maximum_images > 0:
        chosen_images = dataset.take(maximum_images)
    else:
        builder = load_dataset_builder(dataset_name)
        total_examples = builder.info.splits['train'].num_examples
        print(f"Loading {total_examples} images . . .")
        chosen_images = dataset.take(total_examples)

    # Determine the image type from the first item
    first_image = next(iter(chosen_images))
    image_type = "unknown"

    if isinstance(first_image['image'], dict) and 'bytes' in first_image['image']:
        image_type = "dictionary"
    elif isinstance(first_image['image'], Image.Image):
        image_type = "image"
    elif isinstance(first_image['image'], str):
        image_type = "string"
    else:
        raise ValueError(f"Unknown image format: {type(first_image['image'])}")

    for item in chosen_images:
        if image_type == 'dictionary':
            # If it's a dict with 'bytes', convert to PIL Image
            image = Image.open(io.BytesIO(item['image']['bytes']))
        elif image_type == 'image':
            # If it's already a PIL Image, use it directly
            image = item['image']
        elif image_type == 'string':
            # If it's a string (possibly a file path), try to open it
            image = Image.open(item['image'])
        else:
            # If we can't handle the image format throw an error
            raise ValueError(f"Unknown image format: {type(item['image'])}")

        identifier = f"image_{item['id']}" if 'id' in item else "unknown"
        images.append((image, identifier))

    return images


def load_image_from_url(input: str) -> Tuple[Image.Image, str]:
    """
    Load an image from a given URL.

    This function downloads an image from the specified URL and returns it along with its filename.

    Args:
        input (str): The URL of the image to be loaded.

    Returns:
        Tuple[Image.Image, str]: A tuple containing:
            - Image.Image: The loaded image.
            - str: The filename extracted from the URL path.

    Raises:
        requests.exceptions.RequestException: If there's an error in downloading the image.
        PIL.UnidentifiedImageError: If the downloaded content cannot be opened as an image.
    """
    print(f"Loading image from URL {input}")

    response = requests.get(input, stream=True)
    response.raise_for_status()
    image = Image.open(response.raw)
    filename = urlparse(input).path
    return (image, filename)


def load_image_from_local_file(input: str) -> Tuple[Image.Image, str]:
    """
    Load an image from a local file.

    This function opens an image from the specified local file path and returns it along with its filename.

    Args:
        input (str): The local file path of the image to be loaded.

    Returns:
        Tuple[Image.Image, str]: A tuple containing:
            - Image.Image: The loaded image.
            - str: The filename of the image.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        PIL.UnidentifiedImageError: If the file cannot be opened as an image.
    """
    print(f"Loading local image from file {input}")

    image = Image.open(input)
    filename = os.path.basename(input)
    return (image, filename)


def load_images_from_local_folder(input: str) -> List[Tuple[Image.Image, str]]:
    """
    Load images recursively from a local folder.

    This function walks through the specified folder and its subfolders, loading all supported image files.
    Supported formats are: PNG, JPG, JPEG, TIFF, BMP, GIF, and WebP.

    Args:
        input (str): The path to the local folder containing images.

    Returns:
        List[Tuple[Image.Image, str]]: A list of tuples, where each tuple contains:
            - Image.Image: The loaded image.
            - str: The relative path of the image file from the input directory.

    Note:
        If an error occurs while loading an image, it will be skipped and the error will be printed.
    """
    images = []

    print(f"Loading local images recursively from folder {input}")

    for root, _, files in os.walk(input):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp')):
                try:
                    print(f"Loading local image from file {file_path}")
                    image = Image.open(file_path)
                    # Use the relative path from the input directory as the image identifier
                    relative_path = os.path.relpath(file_path, input)
                    images.append((image, relative_path))
                except Exception as e:
                    print(f"Error loading image {file_path}: {e}")

    return images


def load_images(image_input: str, maximum_images: int) -> List[Tuple[Image.Image, str]]:
    """
    Load one or more images from the given input.

    Args:
        image_input (str): A comma-separated list of image sources (URLs, file paths, or dataset references).

    Returns:
        List[Tuple[Image.Image, str]]: A list of tuples containing the loaded images and their filenames.

    Raises:
        ValueError: If no valid images are found in the input.
    """

    print("Loading image(s)...")
    images = []

    inputs = [input.strip() for input in image_input.split(',')]

    for input in inputs:
        if input.startswith(('http://', 'https://')):
            images.append(load_image_from_url(input))
        elif input.startswith('hf://'):
            images.extend(load_hugging_face_images(input, maximum_images))
        elif os.path.isfile(input):
            images.append(load_image_from_local_file(input))
        elif os.path.isdir(input):
            images.extend(load_images_from_local_folder(input))
        else:
            print(f"Warning: Invalid image input: {input}")

    if not images:
        raise ValueError(f"No valid images found in input: {image_input}")

    return images


def load_prompts(prompt: str, prompt_file: str) -> List[str]:
    """
    Load prompts from a given prompt string and/or a prompt file.

    Args:
        prompt (str): A single prompt string.
        prompt_file (str): Path to a file containing multiple prompts.

    Returns:
        List[str]: A list of loaded prompts.

    Raises:
        ValueError: If no prompts are provided.
    """

    print("Loading prompt(s)...")
    prompts = []

    if prompt:
        prompts.append(prompt)
        print(f"Loaded prompt '{prompt}'")

    if prompt_file:
        with open(prompt_file, 'r') as f:
            prompts.extend(line.strip() for line in f if line.strip())
        print(f"Loaded prompts from {prompt_file}")

    if not prompts:
        raise ValueError("No prompts provided. Please specify either a prompt or a prompt file.")

    return prompts


def generate_caption(model: torch.nn.Module, processor: Union[AutoProcessor, AutoTokenizer], image: Image.Image, prompt: str, model_type: str) -> str:
    """
    Generate a caption for the given image using the specified model and prompt.

    Args:
        model (torch.nn.Module): The loaded model.
        processor (Union[AutoProcessor, AutoTokenizer]): The model's processor.
        image (Image.Image): The input image.
        prompt (str): The prompt to use for captioning.
        model_type (str): The type of the model.

    Returns:
        str: The generated caption.
    """

    print("Generating caption...")
    adapter = importlib.import_module(f"model_adapters.{model_type}_adapter")
    return adapter.generate_caption(model, processor, image, prompt)


def save_results(results: List[Dict], output_format: str, output_file: str, append: bool) -> None:
    """
    Save the captioning results to a file.

    Args:
        results (List[Dict]): A list of dictionaries containing the captioning results.
        output_format (str): The format to save the results in ('csv' or 'jsonl').
        output_file (str): The name of the output file.
        append (bool): Whether to append to an existing file or create a new one.

    Raises:
        ValueError: If an unsupported output format is specified.
    """

    print("Saving results...")
    output_folder = os.path.join(os.getcwd(), "outputs")
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, output_file)

    # Ensure the output file has the correct extension
    if not output_path.endswith(f'.{output_format}'):
        output_path += f'.{output_format}'

    mode = 'a' if append else 'w'
    header = not os.path.exists(output_path) or not append

    fieldnames = ['file_name', 'url', 'prompt', 'caption', 'model_type', 'model_name']

    if output_format == 'csv':
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if header:
                writer.writeheader()
            writer.writerows(results)
    elif output_format == 'tsv':
        with open(output_path, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore', delimiter='\t')
            if header:
                writer.writeheader()
            writer.writerows(results)
    elif output_format == 'jsonl':
        with open(output_path, mode, encoding='utf-8') as f:
            for result in results:
                result_without_image = {key: value for key, value in result.items() if key in fieldnames}
                json.dump(result_without_image, f, ensure_ascii=False)
                f.write('\n')
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    print(f"Results saved to {output_path}")
