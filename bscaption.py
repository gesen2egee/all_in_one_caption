from torch import nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast, AutoModelForCausalLM
from pathlib import Path
import torch
import torch.amp.autocast_mode
from PIL import Image
import os
import requests
import traceback
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from imgutils.tagging import get_wd14_tags, tags_to_text
from imgutils.validate import anime_completeness
from tqdm import tqdm
import fnmatch
import argparse
import warnings
import logging
from typing import List, Union

CLIP_PATH = "google/siglip-so400m-patch14-384"
VLM_PROMPT = "A descriptive caption for this image:\n"
MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
CHECKPOINT_PATH = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/resolve/main/wpkklhc6/image_adapter.pt"
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("transformers").setLevel(logging.ERROR)

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        return self.linear2(self.activation(self.linear1(vision_outputs)))

def load_models():
    print("Loading CLIP 📎")
    clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
    clip_model = AutoModel.from_pretrained(CLIP_PATH).vision_model.eval().requires_grad_(False).to("cuda")

    print("Loading tokenizer 🪙")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast)), f"Tokenizer is of type {type(tokenizer)}"

    print("Loading LLM 🤖")
    text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16).eval()

    print("Loading image adapter 🖼️")
    image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
    image_adapter.load_state_dict(torch.load( "image_adapter.pt", map_location="cpu", weights_only=True))
    image_adapter.eval().to("cuda")

    return clip_processor, clip_model, tokenizer, text_model, image_adapter

@torch.no_grad()
def stream_chat(input_images: List[Image.Image], batch_size: int, pbar: tqdm, models: tuple) -> List[str]:
    clip_processor, clip_model, tokenizer, text_model, image_adapter = models
    torch.cuda.empty_cache()
    all_captions = []

    for i in range(0, len(input_images), batch_size):
        batch = input_images[i:i+batch_size]
        
        try:
            images = clip_processor(images=batch, return_tensors='pt', padding=True).pixel_values.to('cuda')
        except ValueError as e:
            print(f"Error processing image batch: {e}")
            print("Skipping this batch and continuing...")
            continue

        with torch.amp.autocast_mode.autocast('cuda', enabled=True):
            vision_outputs = clip_model(pixel_values=images, output_hidden_states=True)
            image_features = vision_outputs.hidden_states[-2]
            embedded_images = image_adapter(image_features).to(dtype=torch.bfloat16)

        prompt = tokenizer.encode(VLM_PROMPT, return_tensors='pt')
        prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda')).to(dtype=torch.bfloat16)
        embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64)).to(dtype=torch.bfloat16)

        inputs_embeds = torch.cat([
            embedded_bos.expand(embedded_images.shape[0], -1, -1),
            embedded_images,
            prompt_embeds.expand(embedded_images.shape[0], -1, -1),
        ], dim=1).to(dtype=torch.bfloat16)

        input_ids = torch.cat([
            torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long).expand(embedded_images.shape[0], -1),
            torch.zeros((embedded_images.shape[0], embedded_images.shape[1]), dtype=torch.long),
            prompt.expand(embedded_images.shape[0], -1),
        ], dim=1).to('cuda')

        attention_mask = torch.ones_like(input_ids)

        generate_ids = text_model.generate(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=300,
            do_sample=True,
            top_k=10,
            temperature=0.5,
        )

        generate_ids = generate_ids[:, input_ids.shape[1]:]

        for ids in generate_ids:
            caption = tokenizer.decode(ids[:-1] if ids[-1] == tokenizer.eos_token_id else ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            caption = caption.replace('<|end_of_text|>', '').replace('<|finetune_right_pad_id|>', '').strip()
            all_captions.append(caption.strip().replace('\n', ' ').replace(', ', ' '))

        if pbar:
            pbar.update(len(batch))

    return all_captions

def process_directory(input_dir: Path, output_dir: Path, batch_size: int, models: tuple):
    output_dir.mkdir(parents=True, exist_ok=True)
    image_files = [f for f in input_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]
    images_to_process = [f for f in image_files if not (output_dir / f"{f.stem}.txt").exists()]

    if not images_to_process:
        print("No new images to process.")
        return

    with tqdm(total=len(images_to_process), desc="Processing images", unit="image") as pbar:
        for i in range(0, len(images_to_process), batch_size):
            batch_files = images_to_process[i:i+batch_size]
            batch_images = [Image.open(f).convert('RGB') for f in batch_files]

            captions = stream_chat(batch_images, batch_size, pbar, models)
            
            for file, caption in zip(batch_files, captions):
                with open(output_dir / f"{file.stem}.txt", 'w', encoding='utf-8') as f:
                    caption = "sakuranomiya maika, " + caption
                    f.write(caption)

            for img in batch_images:
                img.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and generate captions.")
    parser.add_argument("input", nargs='+', help="Input image file or directory (or multiple directories)")
    parser.add_argument("--output", help="Output directory (optional)")
    parser.add_argument("--bs", type=int, default=4, help="Batch size (default: 4)")
    return parser.parse_args()

def main():
    args = parse_arguments()
    input_paths = [Path(input_path) for input_path in args.input]
    batch_size = args.bs
    models = load_models()

    for input_path in input_paths:
        if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
            output_path = input_path.with_suffix('.txt')
            print(f"Processing single image 🎞️: {input_path.name}")
            with tqdm(total=1, desc="Processing image", unit="image") as pbar:
                captions = stream_chat([Image.open(input_path).convert('RGB')], 1, pbar, models)
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(captions[0])
            print(f"Output saved to {output_path}")
        elif input_path.is_dir():
            output_path = Path(args.output) if args.output else input_path
            print(f"Processing directory 📁: {input_path}")
            print(f"Output directory 📦: {output_path}")
            print(f"Batch size 🗄️: {batch_size}")
            process_directory(input_path, output_path, batch_size, models)
        else:
            print(f"Invalid input: {input_path}")
            print("Skipping...")

    if not input_paths:
        print("Usage:")
        print("For single image: python app.py [image_file] [--bs batch_size]")
        print("For directory (same input/output): python app.py [directory] [--bs batch_size]")
        print("For directory (separate input/output): python app.py [directory] --output [output_directory] [--bs batch_size]")
        print("For multiple directories: python app.py [directory1] [directory2] ... [--output output_directory] [--bs batch_size]")
        sys.exit(1)

if __name__ == "__main__":
    main()
