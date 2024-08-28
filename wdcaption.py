from pathlib import Path
from PIL import Image
import torch
import os
import requests
import traceback
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from imgutils.tagging import get_wd14_tags, tags_to_text
from imgutils.validate import anime_completeness
from tqdm import tqdm
import fnmatch
import argparse

aes_model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
aes_model = aes_model.to(torch.bfloat16).cuda()

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
    tags_text = f"{wd_caption}"
	
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
                wd_caption = tags_to_text(features, use_escape=False, use_spaces=True) + f", {max(rating, key=rating.get)}" + f", {get_aesthetic_tag(image)}"
                wd_captions[image_path] = tags = wd_caption.split(', ')                
                for tag in tags:
                    tag_dict[tag] = tag_dict.get(tag, 0) + 1
                tag_dict['caption_count'] = tag_dict.get('caption_count', 0) + 1
                
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()

        del_tag = [tag for tag, count in tag_dict.items() if tag != 'caption_count' and count > tag_dict['caption_count'] * 0.5]
        keep_tag = [tag for tag, count in sorted(tag_dict.items(), key=lambda item: item[1], reverse=True) if tag not in del_tag and tag != 'caption_count']
        
        for i, tag in enumerate(keep_tag):
            tag_count = sum([1 for caption_tags in wd_captions.values() if tag in caption_tags])
            if tag_count < 3 or tag in del_tag:
                continue
            for other_tag in keep_tag[i+1:]:
                other_tag_count = sum([1 for caption_tags in wd_captions.values() if tag in caption_tags and other_tag in caption_tags])
                if other_tag_count < 4 or other_tag in del_tag:
                    continue
                if other_tag_count >= tag_count * 0.85:
                    del_tag.append(other_tag)
                    print(f"{other_tag} added to del_tag, it appears in captions which is more than 90% of {tag}")
        print(f"del_tag: {del_tag}")
        
        for image_path in image_paths:
            try:
                tags = wd_captions[image_path]
                filtered_tags = [tag for tag in tags if not any(d_tag in tag for d_tag in del_tag) or not args.del_tag]
                wd_caption = ', '.join(filtered_tags)
                process_image(image_path, args, wd_caption)
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作txt中的第一個詞")
    parser.add_argument("--del_tag", action="store_true", help="自動刪除子目錄中的wd tag多數標( > 50%)")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()
    find_and_process_images(args.directory, args)
