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
from imgutils.tagging import get_wd14_tags, tags_to_text
from transformers import AutoProcessor, AutoModelForCausalLM
import traceback
from datetime import datetime, timedelta
from imgutils.validate import anime_completeness, is_ai_created
from imgutils.metrics import laplacian_score
from imgutils.ocr import detect_text_with_ocr
import os
from transformers import AutoProcessor, SiglipModel
import shutil
os.environ['ONNX_MODE'] = 'gpu'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = 'MiaoshouAI/Florence-2-large-PromptGen-v1.5' #'yayayaaa/florence-2-large-ft-moredetailed' 
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(device).half()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def get_aesthetic_tag(image):
    tag, score = anime_completeness(image, model_name="caformer_s36_v2-beta")
    if tag != "monochrome" :
        if laplacian_score(image) < 500:
            tag = "blurry"                
        elif is_ai_created(image):
            tag = "ai created"
        else:    
            texts = detect_text_with_ocr(image)
            if len(texts) > 1:
                tag = "text"
    if tag in ["monochrome", "text", "blurry", "rough", "ai created"]:  
        return tag
    else:
        return ""

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
        caption = parsed_answer[task_prompt].replace('\n', ' ').replace(',', ' ')
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
        if name in name_set:
            return True
        name_parts = name.split(' ')
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
    name_from_folder = ""
    concept_tag = ""
    # 查找 boorutag 文件路徑
    for ext in ['.jpg.boorutag', '.png.boorutag']:
        potential_path = base_file_name + ext
        if os.path.exists(potential_path):
            boorutag_path = potential_path
            break

    chartags = set()

    # 獲取 parent_folder 並添加 name_from_folder
    parent_folder = Path(image_path).parent.name
    if args.folder_name and "_" in parent_folder and parent_folder.split("_")[0].isdigit():
        name_from_folder = parent_folder.split('_')[1].replace('_', ' ').strip().lower()
            
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
                    artisttag = lines[6].strip().replace(' artstlye', '')
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

    if name_from_folder and features and ("solo" in features or "solo_focus" in features):
        return f"{name_from_folder}", ', '.join(chartags), boorutag, artisttag

    if has_reverse_name(chartags, name_from_folder):
        name_from_folder = ""

    if len(chartags) > 3:
        chartags = []

    if not name_from_folder and features and ("solo" in features or "solo_focus" in features):
        return ", ".join(filter(None, [name_from_folder, ' '.join(chartags)])), ', '.join(chartags), boorutag, artisttag

    return ", ".join(filter(None, [name_from_folder, ' and '.join(chartags)])), ', '.join(chartags), boorutag, artisttag
    
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

def process_features(features: dict) -> (dict, str):
    patterns_to_keep = [
        r'^anime.*$', r'^monochrome$', r'^.*background$', r'^comic$', r'^greyscale$', r'^sketch$', 
        r'^traditional_media$', r'^parody$', r'^.*cover$', r'^.*_theme$', r'^.*realistic$', 
        r'^multiple_views$', r'^.*blurry.*$', r'^.*koma$', r'^watermark$', r'^pixel.*$',
        r'^oekaki$', r'^3d$', r'^.*chart$', r'^letterboxed$', r'^variations$', r'^.*mosaic.*$', 
        r'^.*censor.*$', r'^.*_name$', r'^signature$', r'^.*_username$', r'^.*text.*$', r'^.*_bubble$', 
        r'^omake$', r'^column.*$', r'^.*_(medium)$', r'^manga$', r'^lineart$', r'^.*logo$'
        #r'^(from_side|from_behind|from_above|from_below)$', r'^(close_up|dutch_angle|downblouse|downpants|pantyshot|upskirt|atmospheric_perspective|fisheye|panorama|perspective|pov|rotated|sideways|upside_down|vanishing_point|straight-on)$', r'^(face|cowboy_shot|portrait|upper_body|lower_body|feet_out_of_frame|full_body|wide_shot|very_wide_shot|cut_in|cropped_legs|head_out_of_frame|cropped_torso|cropped_arms|cropped_shoulders|profile|group_profile)$', r'^(armpit_focus|ass_focus|back_focus|breast_focus|eye_focus|foot_focus|hand_focus|hip_focus|navel_focus|pectoral_focus|thigh_focus|soft_focus|solo_focus)$'
    ]
    keep_tags_set = set()
    patterns_to_keep.extend([r'^holding_.*$'])
    keys = list(features.keys())
    keys_to_delete = []

    for key in keys:
        for pattern in patterns_to_keep:
            regex = re.compile(pattern)
            if regex.match(key):
                keep_tags_set.add(key.replace('_', ' '))
                keys_to_delete.append(key)

    lying_conditions = ['on_stomach', 'on_back', 'on_side']
    if 'lying' in keys and any(cond in keys for cond in lying_conditions):
        for cond in lying_conditions:
            if cond in keys:
                features[f'lying_{cond}'] = 0
                keys_to_delete.append(cond)
        keys_to_delete.append('lying')

    boygirl_tags = [tag for tag in keys if tag in {'multiple_girls', '1girl', 'multiple_boys', '1boy'}]
    if boygirl_tags:
        feature_key = ' '.join(sorted(boygirl_tags))
        features[feature_key] = 0
        for tag in boygirl_tags:
            keys_to_delete.append(tag)           

    for key in keys_to_delete:
        if key in features:
            del features[key]

    keep_tags = keep_tags_set

    return features, keep_tags

def process_image(image_path, args, wd_caption, special_text, wd_features, threshold, aesthetic_tag):
    """
    處理單個圖片，獲取標籤並存儲。修改以支持多進程數據傳遞。
    """
    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')

    # 檢查文件最後修改時間，如果在一周內則略過
    if tag_file_path.exists():
        last_modified_time = datetime.fromtimestamp(tag_file_path.stat().st_mtime)
        if datetime.now() - last_modified_time < timedelta(days=args.continue_caption):
            print(f"Skipping {tag_file_path} as it was modified within the last week.")
            return None, None, 'skipped'   
    try:
        image = resize_image(image_path)
        features, keep_tags = process_features(wd_features)
        if threshold > 0.8:
            tags_text = f"{special_text}, {wd_caption}"
        elif threshold > 0.6:
            tags_text = f"{special_text}"
        elif threshold > 0.4:
            caption, _ = run_example('<CAPTION>', image) 
            tags_text = f"{special_text} {caption}, {wd_caption}"
        elif threshold > 0.2:
            caption, _ = run_example('<DETAILED_CAPTION>', image) 
            tags_text = f"{special_text} {caption}, {wd_caption}"
        else:
            caption, _ = run_example('<MORE_DETAILED_CAPTION>', image) 
            tags_text = f"{special_text} {caption}, {wd_caption}"
        tags = list(set(tag.strip() for tag in tags_text.split(',') if tag not in keep_tags))
        random.shuffle(tags)
        tags_text = ', '.join(filter(None, [aesthetic_tag] + list(keep_tags) + tags))
        with open(tag_file_path, 'w', encoding='utf-8') as f:
            f.write(tags_text.lower())
            
        try:
            npz_pattern = f"{Path(image_path).stem}*_te.npz"
            for npz_file in Path(image_path).parent.glob(npz_pattern):
                if npz_file.exists():
                    os.remove(npz_file)
        except Exception as e:
            print(f"發生錯誤: {e}")
    
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        traceback.print_exc()

def difference(core_embeddings, embedding):
    core_mean = np.mean(core_embeddings, axis=0)  # 取核心嵌入的平均
    diff = np.linalg.norm(core_mean - embedding)  # 計算L2距離
    return diff

def move_images_to_folders(src_folder, normalized_differences):
    for image_path, norm_diff in normalized_differences.items():
        # 將差異值取小數點後一位，並作為資料夾名稱
        norm_diff = norm_diff / 2
        tag = f"{norm_diff:.1f}"
        target_folder = Path(src_folder) / tag
        target_folder.mkdir(exist_ok=True)
        target_image_path = target_folder / Path(image_path).name

        # 移動圖片到以差異為標籤的資料夾中
        shutil.move(str(image_path), str(target_image_path))
        print(f"移動圖片 {image_path} 到 {target_image_path}")

def find_and_process_images(directory, args):
    directory = directory.replace('\\', '/')
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]
    special_texts, wd_captions, wd_features, aesthetic_tags = {}, {}, {}, {}
    for root, dirs, files in os.walk(directory):
        image_paths = []
        tag_dict = {}
        embeddings, core_embeddings = {}, []
        del_tag = []
        differences = []
        for ext in extensions:
            for file in files:
                if fnmatch.fnmatchcase(file, ext) or fnmatch.fnmatchcase(file, ext.upper()):
                    image_paths.append(os.path.join(root, file))

        for image_path in tqdm(image_paths, desc=f"處理圖片 {root}"):
            try:
                image = resize_image(image_path)
                rating, features, chars, embedding = get_wd14_tags(image, character_threshold=0.6, general_threshold=0.27, drop_overlap=True, fmt=('rating', 'general', 'character', 'embedding'))      
                wd_caption = tags_to_text(features, use_escape=False, use_spaces=True)
                special_text, _, boorutag, artisttag = generate_special_text(image_path, args, features, chars)
                aesthetic_tag = get_aesthetic_tag(image)
                aesthetic_tags[image_path] = aesthetic_tag
                special_texts[image_path] = special_text
                tags = list(set((wd_caption + ', ' + boorutag).split(', ')))
                wd_captions[image_path] = ', '.join(tags)
                wd_features[image_path] = features
                for tag in tags:
                    tag_dict[tag] = tag_dict.get(tag, 0) + 1
                tag_dict['caption_count'] = tag_dict.get('caption_count', 0) + 1
                embeddings[image_path] = embedding
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()
        if args.del_tag:
            del_tag = [tag for tag, count in tag_dict.items() if tag != 'caption_count' and count > tag_dict['caption_count'] * 0.5]
            print(del_tag)
            
        for image_path in image_paths:
            if image_path in embeddings:
                core_embeddings.append(embeddings[image_path])
                
        if core_embeddings:
            core_embeddings = np.array(core_embeddings) 
            for image_path in tqdm(image_paths, desc="計算差異值"):
                try:
                    diff = difference(core_embeddings, embeddings[image_path])
                    differences.append((image_path, diff))  # 保存 (image_path, diff)
                except Exception as e:
                    print(f"處理圖片 {image_path} 時出錯: {e}")
        
            sorted_differences = sorted(differences, key=lambda x: x[1]) 
            n = len(sorted_differences)
            normalized_differences = {}
            for i, (image_path, diff) in enumerate(sorted_differences):
                norm_diff = 1 - (i / (n - 1)) if n > 1 else 1
                #if norm_diff > 0.6:
                #    if "simple_background" in wd_features[image_path]:
                #        norm_diff = norm_diff - 0.15
                #    else:
                #        norm_diff = norm_diff + 0.15
                #    if "solo" not in wd_features[image_path]:
                #        norm_diff = norm_diff - 0.15
                normalized_differences[image_path] = norm_diff
            #move_images_to_folders(directory, normalized_differences)

        for image_path in tqdm(image_paths, desc="打自然語言標"):
            try:
                wd_caption = wd_captions[image_path]
                tags = wd_caption.split(', ')
                threshold = normalized_differences[image_path]
                features = wd_features[image_path]
                less_tag = []
                for tag in features.keys():
                    if features[tag] < threshold * 0.85:
                        less_tag.append(tag)
                filtered_tags = [tag for tag in tags if not any(d_tag in tag for d_tag in less_tag + del_tag)]
                wd_caption =  ', '.join(filtered_tags)
                process_image(image_path, args, wd_caption, special_texts[image_path], features, threshold, aesthetic_tags[image_path])
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()

        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--del_tag", action="store_true", help="自動刪除子目錄中的wd tag多數標( > 50%)")
    parser.add_argument("--continue_caption", type=int, default=0, help="忽略n天內打的標")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()
    
    find_and_process_images(args.directory, args)
