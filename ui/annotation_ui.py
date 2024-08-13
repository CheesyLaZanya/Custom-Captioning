# Standard library imports
import base64
from io import BytesIO
import tkinter as tk
from tkinter import filedialog

# Third-party library imports
import gradio as gr

# Local imports
from core_functions import (
    generate_annotation,
    get_available_model_types,
    get_suggested_models,
    get_suggested_prompts,
    load_images,
    load_model,
    save_results,
)


def get_folder_path() -> str:
    """
    Open a folder selection dialog and return the selected folder path.

    This function creates a Tkinter root window (hidden from view),
    opens a directory selection dialog, and returns the path of the
    selected directory.

    Returns:
        str: The path of the selected folder. If no folder is selected,
             an empty string is returned.

    Note:
        This function will pause the execution of the program until
        the user selects a folder or closes the dialog.
    """

    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", 1)
    folder_selected = filedialog.askdirectory()
    return folder_selected


def create_ui_interface() -> gr.Blocks:
    def load_preview(image_input, maximum_images):
        try:
            preview_limit = 10
            images = load_images(image_input, maximum_images)
            if not images:
                return None
            return [(img, filename) for img, filename in images[:preview_limit]]
        except Exception as e:
            print(e)
            return None

    def update_textbox(file_objects):
        paths = []

        if file_objects:
            paths.extend(file_object.name for file_object in file_objects)

        return ", ".join(paths)

    def select_folder():
        folder_path = get_folder_path()
        return folder_path

    def change_model_type(model_type):
        try:
            model_choices = get_suggested_models(model_type)
            model_name_input = gr.Dropdown(
                choices=model_choices,
                allow_custom_value=True,
                label="Enter Model Name",
                value=model_choices[0]
            )

            prompt_choices = get_suggested_prompts(model_type)
            prompt_input = gr.Dropdown(
                choices=prompt_choices,
                allow_custom_value=True,
                label="Enter Prompt",
                value=prompt_choices[0]
            )
            return model_name_input, prompt_input
        except Exception as e:
            print(e)
            return None

    def generate_annotation_ui(model_type, model_name, image_input, prompt, maximum_images):
        try:
            model, processor = load_model(model_name, model_type)
            images = load_images(image_input, maximum_images)
            if not images:
                return "Error: No valid images found.", None, None

            results = []
            for image, filename in images:
                vlm_annotation = generate_annotation(model, processor, image, prompt, model_type)
                results.append({
                    "image": image,
                    "file_name": filename,
                    "url": image_input if image_input.startswith(('http://', 'https://', 'hf://')) else "local",
                    "prompt": prompt,
                    "vlm_annotation": vlm_annotation,
                    "model_type": model_type,
                    "model_name": model_name
                })

            # Prepare data for gallery and dataframe
            gallery_images = [(res["image"], f"{res['file_name']}: {res['vlm_annotation'][:50]}...") for res in results]

            df_data = []
            for res in results:
                # Convert PIL Image to base64
                buffered = BytesIO()
                res["image"].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_html = f'<img src="data:image/png;base64,{img_str}" width="100" height="100">'

                truncated_file_name = (res["file_name"][:14] + '...') if len(res["file_name"]) > 14 else res["file_name"]
                file_name_html = f'<span title="{res["file_name"]}">{truncated_file_name}</span>'

                annotation_html = f'<div style="white-space: normal;">{res["vlm_annotation"]}</div>'

                df_data.append([img_html, file_name_html, annotation_html])

            return "Annotations generated successfully.", gallery_images, df_data, results
        except Exception as e:
            return f"Error: {str(e)}", None, None, None

    def ui_save_results(results, output_format, output_file):
        if not results:
            return "Error: No results to save."
        if not output_file:
            return "Error: Please provide a filename."

        try:
            save_results(results, output_format, output_file, append=False)
            return f"Results saved successfully to {output_file}.{output_format}"
        except Exception as e:
            return f"Error saving results: {str(e)}"

    model_types = get_available_model_types()

    with gr.Blocks() as iface:
        gr.Markdown("# Image Annotation UI")
        with gr.Row():
            with gr.Column():
                model_type_input = gr.Dropdown(choices=model_types, label="Select Model Type")
                model_name_input = gr.Dropdown(
                    choices=[],
                    allow_custom_value=True,
                    label="Enter Model Name",
                )
                with gr.Group():
                    with gr.Row():
                        file_input = gr.File(
                            label="Select image file(s) or drop them here",
                            file_types=["image"],
                            file_count="multiple",
                            type="filepath"
                        )
                        folder_button = gr.Button("Or Select a Folder 📂")
                    image_input = gr.Textbox(label="Enter Image Path/URL/Dataset")
                    maximum_images = gr.Number(
                        label="Maximum Images",
                        value=10
                    )
                prompt_input = gr.Dropdown(
                    choices=["Describe the image."],
                    allow_custom_value=True,
                    label="Enter Prompt",
                    value="Describe the image."
                )
                run_button = gr.Button("Generate Annotations")
            with gr.Column():
                output = gr.Textbox(label="Status")
                gallery = gr.Gallery(
                    label="Image Preview",
                    show_label=False,
                    columns=[2],
                    height="auto"
                )

        results_df = gr.Dataframe(
            headers=["Thumbnail", "Filename", "Annotation"],
            datatype=["html", "html", "html"],
            label="Results"
        )

        with gr.Row():
            output_format = gr.Dropdown(
                choices=["csv", "tsv", "jsonl"],
                label="Output Format",
                value="jsonl"
            )
            output_file = gr.Textbox(
                label="Output Filename",
                value="train"
            )
            save_button = gr.Button("Save Results")

        results_state = gr.State([])

        model_type_input.change(
            change_model_type,
            inputs=[model_type_input],
            outputs=[model_name_input, prompt_input]
        )

        folder_button.click(
            select_folder,
            inputs=[],
            outputs=[image_input]
        )

        file_input.change(
            update_textbox,
            inputs=[file_input],
            outputs=[image_input]
        )

        image_input.change(
            load_preview,
            inputs=[image_input, maximum_images],
            outputs=[gallery]
        )

        run_button.click(
            generate_annotation_ui,
            inputs=[model_type_input, model_name_input, image_input, prompt_input, maximum_images],
            outputs=[output, gallery, results_df, results_state]
        )

        save_button.click(
            ui_save_results,
            inputs=[results_state, output_format, output_file],
            outputs=[output]
        )

    return iface
