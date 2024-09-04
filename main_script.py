import os
import re
import argparse
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import re
import random
import traceback
import json
import fnmatch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from imgutils.tagging import get_wd14_tags, tags_to_text
from imgutils.validate import anime_completeness
from transformers import AutoProcessor, AutoModelForCausalLM
import traceback
from datetime import datetime, timedelta

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = 'yayayaaa/florence-2-large-ft-moredetailed'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(device).half()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def read_select_tags(file_path):
    select_tags = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                clean_tag = line.strip()
                if clean_tag:
                    select_tags.add(clean_tag)
                    
    except FileNotFoundError:
        print(f"找不到檔案: {file_path}")
    except Exception as e:
        print(f"發生錯誤: {e}")
    return select_tags

select_tags = read_select_tags('select.txt')

def read_tag_categories(file_path):
    df = pd.read_csv(file_path)
    tag_dict = {}
    for index, row in df.iterrows():
        category = row.iloc[0]
        tags = ','.join([str(row.iloc[-3]), str(row.iloc[-2]), str(row.iloc[-1])])
        tags_list = [tag.strip() for tag in tags.split(',') if tag.strip()]
        for tag in tags_list:
            if tag not in tag_dict:
                tag_dict[tag] = category
    return tag_dict

tag_categories_dict = read_tag_categories('tag_categories.csv')

def read_tags(file_path):
    df = pd.read_excel(file_path)
    tags_dict = dict(zip(df.iloc[:, 0].str.strip(), df.iloc[:, 1].str.strip()))
    return tags_dict
    
tags_dict = read_tags('tags2.xlsx')

def get_attribute(image_path, attribute, default_value_func):
    if image_path in image_meta:
        if attribute not in image_meta[image_path]:
            image_meta[image_path][attribute] = default_value_func()
    return image_meta[image_path][attribute]

def get_data(image_path, attribute, default_value_func):
    if image_path in image_data:
        if attribute not in image_data[image_path]:
            image_data[image_path][attribute] = default_value_func()
    return image_data[image_path][attribute]

def categorize_and_combine(tags):
    category_dict = {}
    for tag in tags:
        category = tag_categories_dict.get(tag, "Uncategorized")
        if category in category_dict:
            category_dict[category].append(tag)
        else:
            category_dict[category] = [tag]
    return [' '.join(tags) for tags in category_dict.values()]

def transform_caption(caption):
    tags = [tag.strip() for tag in caption.split(',')]
    transformed_tags, original_tags = [], []
    for tag in tags:
        if tags_dict.get(tag, "") and tag in select_tags:
            if tag not in transformed_tags:
                transformed_tags.append(f'{tag} means {tags_dict.get(tag, "").lower().replace(".", "")}')
    #    else:
    #        if tag not in original_tags:
    #            original_tags.append(tag)
    #if 'solo' in caption:
    #    original_tags = categorize_and_combine(original_tags)
    combined_tags = transformed_tags    
    return ', '.join(combined_tags)

def run_example(task_prompt, image, text_input=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)
    bboxes = []
    # 將inputs轉換為fp16
    inputs["pixel_values"] = inputs["pixel_values"].half()
    
    with torch.cuda.amp.autocast():
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
        )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height)
    )
    
    if task_prompt == '<DENSE_REGION_CAPTION>':
        dense_labels = parsed_answer['<DENSE_REGION_CAPTION>']['labels']
        caption = ', '.join([label for label in dense_labels if label.count(' ') > 1])
    elif task_prompt == '<CAPTION_TO_PHRASE_GROUNDING>':
        bboxes = parsed_answer['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
        num_persons = len(bboxes)
        num_persons_word = p.number_to_words(num_persons)
        if num_persons > 1:
            caption = f'{num_persons_word} person' if num_persons < 6 else '6+persons'
        else:
            caption = ''
    else:
        caption = parsed_answer[task_prompt].replace('\n', ' ').replace(', ', ' ')
        caption = caption.replace('.', ',')
        caption = caption.replace('  ', ' ')
        caption = caption.strip()
        if caption[-1]==',':
            return caption[:-1], bboxes
    return caption, bboxes

def generate_special_text(image_path, args, features=None, chars=None):
    """
    根據 features, image_path 和 parent_folder 生成 special_text。
    """
    def has_reverse_name(name_set, name):
        """
        檢查 name_set 中是否存在 name 的相反名稱（中間有一個空格）。
        """
        name_parts = name.split()
        if len(name_parts) == 2:
            reverse_name = f"{name_parts[1]} {name_parts[0]}"
            if reverse_name in name_set:
                return True
        return False
    base_file_name = os.path.splitext(image_path)[0]
    boorutag_path = None
    boorutag = ""
    artisttag = ""
    styletag = None
    chartag_from_folder = ""
    concept_tag = ""
    # 查找 boorutag 文件路徑
    for ext in ['.jpg.boorutag', '.png.boorutag']:
        potential_path = base_file_name + ext
        if os.path.exists(potential_path):
            boorutag_path = potential_path
            break

    chartags = set()

    # 獲取 parent_folder 並添加 chartag_from_folder
    parent_folder = Path(image_path).parent.name
    if args.folder_name and "_" in parent_folder and parent_folder.split("_")[0].isdigit():
        chartag_from_folder = parent_folder.split('_')[1].replace('_', ' ').strip().lower()
        chartags.add(chartag_from_folder)            
            
    # 處理 boorutag 文件內容
    if boorutag_path:
        try:
            with open(boorutag_path, 'r', encoding='cp950') as file:
                lines = file.readlines()
                first_line = lines[0]
                first_line_cleaned = re.sub(r'\(.*?\)', '', first_line)
                for tag in first_line_cleaned.split(','):
                    cleaned_tag = tag.replace('\\', '').replace('_', ' ').strip()
                    if not has_reverse_name(chartags, cleaned_tag):
                        chartags.add(cleaned_tag)
                if len(lines) >= 19:
                    artisttag = lines[6].strip()
                    boorutag = lines[18].strip()
                    boorutag_tags = drop_overlap_tags(boorutag.split(', '))
                    boorutag_tags_cleaned = [tag for tag in boorutag_tags if tag.replace(' ', '_') not in features.keys()]
                    boorutag = ', ' + ', '.join(boorutag_tags_cleaned)                
        except Exception as e:
            # 讀取文件或處理過程中發生錯誤
            pass

    # 處理 chars.keys()
    if chars:
        for key in chars.keys():
            cleaned_key = re.sub(r'\(.*?\)', '', key).replace('\\', '').replace('_', ' ').strip()
            if not has_reverse_name(chartags, cleaned_key):
                chartags.add(cleaned_key)

    # 將 chartags 轉換為列表並隨機打亂
    chartags = list(chartags)
    random.shuffle(chartags)

    if chartag_from_folder and features and ("solo" in features or "solo_focus" in features):
        return f"{chartag_from_folder}", ', '.join(chartags), boorutag, artisttag

    if len(chartags) > 3:
        chartags = []
    
    if not chartag_from_folder and features and ("solo" in features or "solo_focus" in features):
        return f"{' '.join(chartags)}" if chartags else "", ', '.join(chartags), boorutag, artisttag

    return f"{', '.join(chartags)}", ', '.join(chartags), boorutag, artisttag
    
def process_image(image_path, folder_chartag, args):
    """
    處理單個圖片，獲取標籤並存儲。修改以支持多進程數據傳遞。
    """

    def resize_image(image_path, max_size=448):
        """
        縮小圖像使其最大邊不超過 max_size，返回縮小後的圖像數據
        """
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

    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')

    # 檢查文件最後修改時間，如果在一周內則略過
    if tag_file_path.exists():
        last_modified_time = datetime.fromtimestamp(tag_file_path.stat().st_mtime)
        if datetime.now() - last_modified_time < timedelta(days=args.continue_caption):
            print(f"Skipping {tag_file_path} as it was modified within the last week.")
            return None, None, 'skipped'   
    try:
        image = resize_image(image_path)
        rating, features, chars = get_wd14_tags(image, character_threshold=0.6, general_threshold=0.2682, drop_overlap=True)
        wd14_caption = tags_to_text(features, use_escape=False, use_spaces=True)
        special_text, chartags, boorutag, artisttag = generate_special_text(image_path, args, features, chars)
        ratingtag = max(rating, key=rating.get)
        wd14_caption = wd14_caption + ', ' + boorutag
        wd14_caption = transform_caption(wd14_caption)
        print(wd14_caption)
        more_detailed_caption, _ = run_example('<MORE_DETAILED_CAPTION>', image) 
        special_text = ' '.join([text.strip() for text in special_text.split(',') if text.strip()])
        tags_text = (
            f"{special_text}, {more_detailed_caption}, {wd14_caption}"
        )     
        with open(tag_file_path, 'w', encoding='utf-8') as f:
            f.write(tags_text.lower())

    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        traceback.print_exc()

def find_and_process_images(directory, args):
    directory = directory.replace('\\', '/')
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]
    all_final_scores = []
    for root, dirs, files in os.walk(directory):
        folder_chartag = {}
        image_paths = []
        image_infos_list = []
        folder_final_scores = []
        for ext in extensions:
            for file in files:
                if fnmatch.fnmatchcase(file, ext) or fnmatch.fnmatchcase(file, ext.upper()):
                    image_paths.append(os.path.join(root, file))

        for image_path in tqdm(image_paths, desc=f"處理圖片 {root}"):
            try:
                process_image(image_path, folder_chartag, args)  
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--continue_caption", type=int, default=0, help="忽略n天內打的標")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()
    
    find_and_process_images(args.directory, args)
