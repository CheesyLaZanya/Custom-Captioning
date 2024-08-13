# Standard library imports
import argparse
import warnings

# Third-party library imports
import transformers

# Local imports
from core_functions import (
    generate_annotation,
    get_available_model_types,
    load_images,
    load_model,
    load_prompts,
    save_results,
)
from ui.debug_ui import create_debug_interface
from ui.annotation_ui import create_ui_interface
from utils import load_config

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

# Disable some warnings for Dolphin
transformers.logging.set_verbosity_error()
transformers.logging.disable_progress_bar()


def main(args: argparse.Namespace) -> None:
    """
    The main function that orchestrates the image annotation process.

    Args:
        args (argparse.Namespace): The command-line arguments.
    """

    config = {}
    if args.config:
        config = load_config(args.config)

    ui = args.ui or config.get('ui', False)
    if ui:
        print("Launching UI mode...")
        iface = create_ui_interface()
        iface.launch(inbrowser=True, show_api=False)
        return

    models = config.get('models', []) if config else []
    image_input = args.image_input or config.get('image_input')
    maximum_images = args.maximum_images or config.get('maximum_images')
    output_format = args.output_format or config.get('output_format', 'jsonl')
    output_file = args.output_file or config.get('output_file', 'train.jsonl')
    append = args.append or config.get('append', False)
    debug = args.debug or config.get('debug', False)

    if not models:
        models.append({
            "model_name": args.model_name,
            "model_type": args.model_type,
            "prompts": load_prompts(args.prompt, args.prompt_file)
        })

    print("Processing Command...")
    images = load_images(image_input, maximum_images)
    results = []

    for model_config in models:
        model, processor = load_model(model_config['model_name'], model_config['model_type'])
        prompts = model_config.get('prompts', [])

        for image, filename in images:
            for prompt in prompts:
                vlm_annotation = generate_annotation(model, processor, image, prompt, model_config['model_type'])
                results.append({
                    "image": image,
                    "file_name": filename,
                    "url": image_input if image_input.startswith(('http://', 'https://', 'hf://')) else "local",
                    "prompt": prompt,
                    "vlm_annotation": vlm_annotation,
                    "model_type": model_config['model_type'],
                    "model_name": model_config['model_name']
                })

    save_results(results, output_format, output_file, append)

    if debug:
        print("Launching debug UI...")
        debug_iface = create_debug_interface(results)
        debug_iface.launch(inbrowser=True, show_api=False)


if __name__ == "__main__":
    model_types = get_available_model_types()

    parser = argparse.ArgumentParser(description="VLM Annotation Script")
    parser.add_argument("--model_name", type=str, help="Name or path of the model")
    parser.add_argument("--model_type", type=str, choices=model_types, help="Type of the model")
    parser.add_argument("--image_input", type=str, help="Image URL, dataset URL, or local path")
    parser.add_argument("--maximum_images", type=int, default=10, help="The maximum number of images to process")
    parser.add_argument("--prompt", type=str, help="Single prompt for annotation")
    parser.add_argument("--prompt_file", type=str, help="File containing multiple prompts")
    parser.add_argument("--output_format", type=str, choices=["csv", "tsv", "jsonl"], help="Output format")
    parser.add_argument("--output_file", type=str, help="Output file name")
    parser.add_argument("--append", action="store_true", help="Append results to the output file instead of overwriting")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with visual output")
    parser.add_argument("--ui", action="store_true", help="Launch interactive UI mode")
    parser.add_argument("--config", type=str, help="Path to the configuration YAML file")

    args = parser.parse_args()
    main(args)
