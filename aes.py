import subprocess
import sys
from pathlib import Path
import torch
import shutil
import re
from tqdm import tqdm
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image
from imgutils.metrics import get_aesthetic_score, anime_dbaesthetic, laplacian_score
from imgutils.validate import anime_completeness_score
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
import os

# 設定 GPU 模式
os.environ['ONNX_MODE'] = 'gpu'
model, preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
model = model.to(torch.bfloat16).cuda()

def process_image(image_path):    
    weight_file_path = Path(image_path).with_suffix('.weight')
    if weight_file_path.exists(): 
        with open(weight_file_path, 'r') as f:
            first_line = f.readline().strip()
            try:
                score = float(first_line)
                return score
            except ValueError:
                print(f"Warning: Could not convert {first_line} to float.")

    image = Image.open(image_path).convert("RGB")
    pixel_values = (
        preprocessor(images=image, return_tensors="pt")
        .pixel_values.to(torch.bfloat16)
        .cuda()
    )
    
    with torch.inference_mode():
        aes_score = min(max(model(pixel_values).logits.squeeze().float().cpu().numpy() - 3, 0) / 3, 1)
    
    anime_score = min(get_aesthetic_score(image), 1)
    db_confidence = anime_dbaesthetic(image, fmt='confidence')
    db_score = min(1 - max(db_confidence["worst"], db_confidence["low"], db_confidence["normal"]), 1)
    anime_completeness = anime_completeness_score(image, model_name="caformer_s36_v2-beta")
    completeness_score = min(1 - max(anime_completeness["rough"], anime_completeness["monochrome"]), 1)
    
    lapl_score = laplacian_score(image)
    # 計算最後的分數
    mean_score = (db_score + aes_score + anime_score + completeness_score) / 4
    max_score = max(db_score , aes_score , anime_score)
    min_score = min(db_score , aes_score , anime_score)
    lapl_score = max(1 - (max((500 - lapl_score), 0) / 1500), min_score)
    # 計算最後的分數
    score = lapl_score * ((min_score + mean_score + (max_score * 2)) / 4)
    return score

def get_aesthetic_tag(score):
    tag = str(math.floor(score))
    return tag
    
def save_score_to_file(image_path, score):
    # 創建與圖片同名的 .weight 文件並保存分數
    weight_file_path = image_path.with_suffix('.weight')
    with open(weight_file_path, 'w') as f:
        f.write(f'{score}')

def process_single_image(image_path):
    weight_file_path = Path(image_path).with_suffix('.weight')
    #if weight_file_path.exists(): 
    #    return
    score = process_image(image_path)
    save_score_to_file(image_path, score)
    tag = get_aesthetic_tag(score * 10)
    src_folder = image_path.parent
    target_folder = src_folder / tag
    target_folder.mkdir(parents=True, exist_ok=True) 
    target_image_path = target_folder / image_path.name
    shutil.move(str(image_path), str(target_image_path))
    booru_tag_file = Path(str(image_path) + ".boorutag")
    if booru_tag_file.exists():
        shutil.move(str(image_path) + ".boorutag", str(target_image_path) + ".boorutag")
        
    txt_file = image_path.with_suffix('.txt')
    if txt_file.exists():
        shutil.move(str(txt_file), str(target_image_path.with_suffix('.txt')))

    # 生成 .weight 文件路徑並移動
    weight_file = image_path.with_suffix('.weight')
    if weight_file.exists():
        shutil.move(str(weight_file), str(target_image_path.with_suffix('.weight')))

def process_all_images(base_path, max_workers=4):
    # 定義支持的圖像擴展名
    extensions = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.WEBP", "*.BMP"]
    
    # 遍歷所有子資料夾並收集所有圖片路徑
    image_paths = []
    for ext in extensions:
        image_paths.extend(base_path.rglob(ext))  # 使用 rglob 來遞歸收集所有子資料夾中的圖片
    
    # 如果沒有圖片，則直接返回
    if not image_paths:
        print("No images found.")
        return

    # 使用多進程處理所有圖片
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_image, image_path): image_path for image_path in image_paths}
        for future in tqdm(as_completed(futures), total=len(image_paths), desc="Processing images"):
            try:
                future.result()  # 捕獲任何潛在的錯誤
            except Exception as exc:
                print(f"Error processing {futures[future]}: {exc}")

def main():
    base_path = Path.cwd()  # 當前工作目錄
    process_all_images(base_path, max_workers=2)  # 對所有子資料夾和圖片進行處理

if __name__ == "__main__":
    main()
