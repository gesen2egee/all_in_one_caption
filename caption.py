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

CLIP_PATH = "google/siglip-so400m-patch14-384"
VLM_PROMPT = "A descriptive caption for this image:\n"
MODEL_PATH = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"
CHECKPOINT_PATH = "https://huggingface.co/spaces/fancyfeast/joy-caption-pre-alpha/resolve/main/wpkklhc6/image_adapter.pt"

if not os.path.exists("image_adapter.pt"):
    print("Downloading image_adapter.pt...")
    response = requests.get(CHECKPOINT_PATH)
    with open("image_adapter.pt", "wb") as f:
        f.write(response.content)

aes_model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
aes_model = aes_model.to(torch.bfloat16).cuda()

class ImageAdapter(nn.Module):
    def __init__(self, input_features: int, output_features: int):
        super().__init__()
        self.linear1 = nn.Linear(input_features, output_features)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(output_features, output_features)
    
    def forward(self, vision_outputs: torch.Tensor):
        x = self.linear1(vision_outputs)
        x = self.activation(x)
        x = self.linear2(x)
        return x

# Load CLIP
print("Loading CLIP")
clip_processor = AutoProcessor.from_pretrained(CLIP_PATH)
clip_model = AutoModel.from_pretrained(CLIP_PATH)
clip_model = clip_model.vision_model
clip_model.eval()
clip_model.requires_grad_(False)
clip_model.to("cuda")

# Tokenizer
print("Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=False)
assert isinstance(tokenizer, PreTrainedTokenizer) or isinstance(tokenizer, PreTrainedTokenizerFast), f"Tokenizer is of type {type(tokenizer)}"

# LLM
print("Loading LLM")
text_model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16)
text_model.eval()

# Image Adapter
print("Loading image adapter")
image_adapter = ImageAdapter(clip_model.config.hidden_size, text_model.config.hidden_size)
image_adapter.load_state_dict(torch.load("image_adapter.pt", map_location="cpu"))
image_adapter.eval()
image_adapter.to("cuda")

@torch.no_grad()
def stream_chat(input_image: Image.Image):
    torch.cuda.empty_cache()

    # Preprocess image
    image = clip_processor(images=input_image, return_tensors='pt').pixel_values
    image = image.to('cuda')

    # Tokenize the prompt
    prompt = tokenizer.encode(VLM_PROMPT, return_tensors='pt', padding=False, truncation=False, add_special_tokens=False)

    # Embed image
    with torch.amp.autocast_mode.autocast('cuda', enabled=True):
        vision_outputs = clip_model(pixel_values=image, output_hidden_states=True)
        image_features = vision_outputs.hidden_states[-2]
        embedded_images = image_adapter(image_features)
        embedded_images = embedded_images.to('cuda')

    # Embed prompt
    prompt_embeds = text_model.model.embed_tokens(prompt.to('cuda'))
    assert prompt_embeds.shape == (1, prompt.shape[1], text_model.config.hidden_size), f"Prompt shape is {prompt_embeds.shape}, expected {(1, prompt.shape[1], text_model.config.hidden_size)}"
    embedded_bos = text_model.model.embed_tokens(torch.tensor([[tokenizer.bos_token_id]], device=text_model.device, dtype=torch.int64))

    # Construct prompts
    inputs_embeds = torch.cat([
        embedded_bos.expand(embedded_images.shape[0], -1, -1),
        embedded_images.to(dtype=embedded_bos.dtype),
        prompt_embeds.expand(embedded_images.shape[0], -1, -1),
    ], dim=1)

    input_ids = torch.cat([
        torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long),
        torch.zeros((1, embedded_images.shape[1]), dtype=torch.long),
        prompt,
    ], dim=1).to('cuda')
    attention_mask = torch.ones_like(input_ids)

    generate_ids = text_model.generate(input_ids, inputs_embeds=inputs_embeds, attention_mask=attention_mask, max_new_tokens=300, do_sample=True, top_k=10, temperature=0.5, suppress_tokens=None)

    # Trim off the prompt
    generate_ids = generate_ids[:, input_ids.shape[1]:]
    if generate_ids[0][-1] == tokenizer.eos_token_id:
        generate_ids = generate_ids[:, :-1]

    caption = tokenizer.batch_decode(generate_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

    return caption.strip().replace('\n', ' ').replace(', ', ' ')

def get_aesthetic_tag(image):
    def aesthetic_tag(image):
        pixel_values = (
            preprocessor(images=image, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .to('cuda')
        )
        with torch.inference_mode():
            score = aes_model(pixel_values).logits.squeeze().float().cpu().numpy()
        if score >= 5.5:
            return "aesthetic"
        elif score >= 4.5:
            return "good"
        else:
            return "rough"
    tag, _ = anime_completeness(image)
    if tag == "polished":
        tag = aesthetic_tag(image)
    return tag

def resize_image(image_path, max_size=448):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")		
    if max(image.width, image.height) > max_size:
        if image.width > image.height:
            new_width = max_size
            new_height = int(max_size * image.height / image.width)
        else:
            new_height = max_size
            new_width = int(max_size * image.width / image.height)
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image

def process_image(image_path, args, wd_caption):
    image = resize_image(image_path)
    joy_caption = stream_chat(image)
    tags_text = f"{joy_caption}, the image is composed of the following: {wd_caption}"
	
    parent_folder = Path(image_path).parent.name
    if args.folder_name and "_" in parent_folder and parent_folder.split("_")[0].isdigit():
        tags_text = parent_folder.split('_')[1].replace('_', ' ').strip().lower() + ', ' + tags_text

    tag_file_path = os.path.splitext(image_path)[0] + ".txt"
    with open(tag_file_path, 'w', encoding='utf-8') as f:
        f.write(tags_text.lower())

def find_and_process_images(directory, args):
    directory = directory.replace('\\', '/')
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]
    wd_captions = {}
    
    for root, dirs, files in os.walk(directory):
        image_paths = []
        tag_dict = {}
        for ext in extensions:
            for file in files:
                if fnmatch.fnmatchcase(file, ext) or fnmatch.fnmatchcase(file, ext.upper()):
                    image_paths.append(os.path.join(root, file))

        for image_path in tqdm(image_paths, desc=f"處理圖片 {root}"):
            try:
                image = resize_image(image_path)
                rating, features, _, _ = get_wd14_tags(image, character_threshold=0.6, general_threshold=0.27, drop_overlap=True, fmt=('rating', 'general', 'character', 'embedding'))      
                wd_caption = tags_to_text(features, use_escape=False, use_spaces=True)
                wd_captions[image_path] = wd_caption + f", rating:{max(rating, key=rating.get)}" + f", {get_aesthetic_tag(image)}"
                tags = wd_caption.split(', ')
                for tag in tags:
                    tag_dict[tag] = tag_dict.get(tag, 0) + 1
                tag_dict['caption_count'] = tag_dict.get('caption_count', 0) + 1
                
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()

        del_tag = [tag for tag, count in tag_dict.items() if tag != 'caption_count' and count > tag_dict['caption_count'] * 0.7]
        print(del_tag)
        
        for image_path in tqdm(image_paths, desc="應用過濾標籤"):
            try:
                wd_caption = wd_captions[image_path]
                tags = wd_caption.split(', ')
                filtered_tags = [tag for tag in tags if not any(d_tag in tag for d_tag in del_tag) or not args.del_tag]
                wd_caption = ', '.join(filtered_tags)
                process_image(image_path, args, wd_caption)
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作txt中的第一個詞")
    parser.add_argument("--del_tag", action="store_true", help="自動刪除子目錄中的wd tag多數標( > 70%)")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()
    find_and_process_images(args.directory, args)
