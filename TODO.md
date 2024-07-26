# Issues
- [x] In the webUI, if I try to export jsonl with .webp images, it says they aren't serializable so something is wrong there.
- [ ] When loading a large dataset, we should pull images 1 by 1 or in small batches, rather than trying to load them all at once?

# Features
## New Models
- [x] Add support for "SmilingWolf/wd-vit-tagger-v3"
    - https://huggingface.co/SmilingWolf/wd-vit-tagger-v3
- [x] Add support for "THUDM/cogvlm-chat-hf"
    - https://huggingface.co/THUDM/cogvlm-chat-hf

## Other Features
- [ ] Add support for revisions/versions
- [x] Make UI support suggested models
- [x] Make UI support suggested prompts
- [ ] Load a set of URLs from a file to run on each.
- [ ] Support multiple formats for URL lists?
- [x] Investigate negative prompts
    - Tried this out with the florence2 model, tokenization differences between the bad words and the generated ids caused issues here.
- [ ] Investigate any settings like temperature?
- [x] Max length on file names in the UI's results table and truncation.
- [x] Tooltip for the filename?
- [x] Default value for output format
- [ ] Build app into an exe?
    - https://github.com/jhc13/taggui/blob/main/taggui-windows.spec ?
- [ ] Unit tests?
- [ ] Update image list when maximum images is updated?

# Testing

## Setup and Installation
- [x] Clone the repository successfully
- [x] Create and activate virtual environment
- [x] Install all dependencies from requirements.txt without errors
- [x] Verify CUDA installation (if applicable)

## Command-line Interface
- [x] Run script with --help to verify all options are displayed correctly
- [x] Test each supported model type (llava, blip2, florence2, dolphin)
- [x] Test with a single image URL
- [x] Test with a local image file
- [x] Test with a directory of images
- [x] Test with a Hugging Face dataset
- [x] Verify single prompt functionality
- [x] Verify multiple prompts from a file
- [x] Test CSV output format
- [x] Test JSONL output format
- [x] Verify debug mode functionality

## UI Mode
- [x] Launch UI mode successfully
- [x] Upload an image and generate a caption
- [x] Test with different prompts
- [x] Verify that the generated caption is displayed correctly

## Error Handling
- [ ] Test with an invalid model name
- [ ] Test with an unsupported model type
- [ ] Test with an invalid image URL
- [ ] Test with a non-existent local file
- [ ] Test with an empty prompt
- [ ] Test with a non-existent prompt file

## Performance
- [ ] Verify reasonable processing time for single images
- [ ] Test with a large batch of images to check performance

## Output Verification
- [x] Check that output files are created correctly

## Cross-platform Testing
- [x] Test on Windows
- [x] Test on Linux

## GPU Utilization (if applicable)
- [x] Verify that the GPU is being utilized when available
- [x] Check for any CUDA-related errors
