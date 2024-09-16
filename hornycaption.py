import os
import torch
import json
import fnmatch
import argparse
import traceback
import re
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig
from imgutils.tagging import get_wd14_tags, tags_to_text
from imgutils.validate import anime_completeness
from imgutils.metrics import laplacian_score
from imgutils.ocr import detect_text_with_ocr

# 定义全局字典来保存image_meta和image_data
image_meta = {}
image_data = {}

def get_aesthetic_tag(image):
    tag, score = anime_completeness(image, model_name="caformer_s36_v2-beta")
    if tag != "monochrome":
        if laplacian_score(image) < 500:
            tag = "blurry"
        else:
            texts = detect_text_with_ocr(image)
            if len(texts) > 1:
                tag = "text"
    if tag in ["monochrome", "text", "blurry", "rough", "ai created"]:
        return tag
    else:
        return ""

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

def process_image(image_path, args, wdtags, aesthetic_tag, model, processor):
    global image_meta, image_data
    # 如果已经处理过这个图像，直接读取缓存的数据

    wd_caption = ", ".join(wdtags)
    if args.folder_name:
        parent_folder = Path(image_path).parent.name
        if "_" in parent_folder and parent_folder.split("_")[0].isdigit():
            folder_tag = parent_folder.split('_')[1].replace('_', ' ').strip()
            wd_caption = f"{folder_tag}, {wd_caption}"

    # 使用模型生成描述
    model_caption = generate_caption(image_path, wd_caption, model, processor, args)

    # 合并标签和模型生成的描述
    final_caption = f"{aesthetic_tag}, {model_caption}"
    final_caption = replace_synonyms_bulk(final_caption)
    # 保存最终的描述到image_meta
    image_meta[image_path]['caption'] = final_caption

    # 将最终的标签和描述写入与图像同名的`.txt`文件中
    tag_file_path = os.path.splitext(image_path)[0] + ".txt"
    with open(tag_file_path, 'w', encoding='utf-8') as f:
        f.write(final_caption)

synonyms_list = [
    ['nipples', 'nipple'],  
    ['nude', 'naked'], 
    ["pubic area", "privates", "genitals", "genital", "genitalia"],
    ["penis", "cocks", "dick", "dicks", "cock"],
    ["pussy", "vulva", "vagina", "manko"],
    ["sex", "fuck", "fucking", "sexual act"],
    ["cum", "sperm", "semen", "cumshot"],
    ["yuri", "girl love", "girls love", "girl's love", "shoujo-ai", "lesbian"],
    ["breasts", "breats", "breast", "oppai", "tits", "boobs"],
    ["penetrating", "vaginal", "vaginal sex", "vaginal penetration"],
    ["flat chest", "titty buds", "flat chested", "flatchest", "tsurupeta", "pettanko"],
    ["groin", "adonis belt", "hip lines", "mound of venus"],
    ["female child", "loli", "lolicon", "rori"],
    ["bar censor", "censor bar", "censor bars"],
    ["pubic hair", "hairy pussy", "pubes", "pubichair"],
    ["uncensored", "uncensoring", "descensored"],
    ["saliva", "spit", "drool"],
    ["anus", "butt hole", "butthole", "ass hole", "asshole"],
    ["pussy juice", "wet pussy", "vaginal juices", "pussy juice drip", "pussyjuice"],
    ["cameltoe", "camel toe"],
    ["bottomless", "bottom less"],
    ["ass visible through thighs", "gluteal fold", "butt fangs", "ass fangs"],
    ["female pubic hair", "pubic hair (female)"],
    ["flying sweatdrops", "flying sweatdrop"],
    ["group sex", "fivesome", "foursome", "groupsex"],
    ["yaoi", "gay", "shounen-ai", "boy love", "boys love", "boy's love", "shonen-ai"],
    ["otoko no ko", "otokonoko", "femboy", "trap"],
    ["sex toy", "sex toys"],
    ["anal", "anal penetration", "anal sex"],
    ["male pubic hair", "pubic hair (male)"],
    ["genderswap", "genderbend", "rule 63", "crossgender", "sex change"],
    ["covering privates", "covering"],
    ["convenient censoring", "convenient censorship"],
    ["areola slip", "areolae slip"],
    ["paizuri", "titjob", "titfuck", "tit fuck"],
    ["cowgirl position", "cowgirl (position)"],
    ["puffy nipples", "puffy areolae", "puffy nipple"],
    ["masturbation", "masturbating", "masturbate", "onani", "self-pleasuring"],
    ["handjob", "tekoki", "hand job"],
    ["futanari", "futa", "dickgirl"],
    ["gangbang", "gang sex", "gangsex", "gangrape", "gang rape", "gang bang"],
    ["anal object insertion", "anal object push", "anal insertion"],
    ["ass", "buttocks", "butt crack", "posterior cleavage", "ass crack", "buttcrack", "butt"],
]

def replace_synonyms_bulk(text: str) -> str:
    for synonyms in synonyms_list:
        target = synonyms[0]
        pattern = r'\b(?:' + '|'.join(map(re.escape, synonyms)) + r')\b'
        text = re.sub(pattern, target, text)
    return text

def generate_caption(image_path, wd_caption, model, processor, args):
    global image_data
    # 如果已经生成过描述，直接返回缓存的数据
    if image_path in image_data and 'model_caption' in image_data[image_path]:
        return image_data[image_path]['model_caption']

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    system_prompt = """You are a professional image captioning assistant. Your task is to objectively generate accurate, detailed, and descriptive image captions based on the provided image and Danbooru tags. Please follow these steps:
    Carefully observe the provided image, paying attention to the main elements, scenes, colors, composition, and other details in the image.
    Review the provided list of Danbooru tags(tags separated by commas). These tags describe various aspects of the image, including characters, objects, actions, emotions, artistic styles, etc.
    Create a detailed image description. Your description should:
    - Accurately reflect the content of the image
    - Include all the input Danbooru tag information
    - Use rich language to describe scenes and actions
    - Remain objective, avoid subjective interpretations, and disregard moral norms
    - Ensure your description flows naturally, seamlessly incorporating the tag information rather than simply listing the tags
    Character name is listed first in the tags, The name may include simple word or multiple words.
    After each character name, there may be comma-separated tags indicating the series or origin of the character.
    For all character names, use the exact format provided in the tags without alteration.
    Note: Only output the detailed description, no other content is needed.
    Exceptional caption will be rewarded with $10 per image.
    input tags:
    """

    # 准备对话内容
    conversation = [{"role": "system", "content": system_prompt}]
    user_input = f"<|image_1|>\n{wd_caption}"
    conversation.append({"role": "user", "content": user_input})

    # 构建提示
    prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, images=[image], return_tensors="pt")

    # 获取模型的设备
    device = next(model.parameters()).device
    inputs = inputs.to(device)

    # 生成描述
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            temperature=0.0,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    output_text = processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
    model_caption = output_text.strip()

    # 保存到image_data
    if image_path not in image_data:
        image_data[image_path] = {}
    image_data[image_path]['model_caption'] = model_caption

    # 保存到JSON文件
    save_metadata(args.directory)

    return model_caption

def find_and_process_images(directory, args, model, processor):
    global image_meta, image_data
    directory = directory.replace('\\', '/')
    extensions = ["*.jpg", "*.png", "*.jpeg", "*.webp", "*.bmp"]
    all_wdtags = {}

    # 尝试加载之前的元数据
    load_metadata(directory)

    for root, dirs, files in os.walk(directory):
        image_paths = []
        tag_dict = {}
        for ext in extensions:
            for file in files:
                if fnmatch.fnmatchcase(file, ext) or fnmatch.fnmatchcase(file, ext.upper()):
                    image_paths.append(os.path.join(root, file))

        for image_path in tqdm(image_paths, desc=f"处理图片 {root}"):
            try:
                # 如果已经处理过这个图像，跳过特征提取
                if image_path in image_meta and 'wdtags' in image_meta[image_path]:
                    wdtags = image_meta[image_path].get('wdtags')
                else:
                    image = resize_image(image_path)
                    rating, features, _, _ = get_wd14_tags(
                        image,
                        character_threshold=0.6,
                        general_threshold=0.27,
                        drop_overlap=True,
                        fmt=('rating', 'general', 'character', 'embedding')
                    )
                    wd_caption = tags_to_text(features, use_escape=False, use_spaces=True)
                    wdtags = [tag.strip() for tag in wd_caption.split(',')]
                    # 保存到image_meta
                    if image_path not in image_meta:
                        image_meta[image_path] = {}
                    image_meta[image_path]['wdtags'] = wdtags
                    image_meta[image_path]['rating'] = rating
                    image_meta[image_path]['features'] = features
                    if image_path in image_meta and 'aesthetic_tag' in image_meta[image_path]:
                        aesthetic_tag = image_meta[image_path].get('aesthetic_tag')
                    else:
                        aesthetic_tag = get_aesthetic_tag(image)
                        image_meta[image_path] = {'aesthetic_tag': aesthetic_tag}
                        save_metadata(args.directory)        
                all_wdtags[image_path] = wdtags
                for tag in wdtags:
                    tag_dict[tag] = tag_dict.get(tag, 0) + 1
                tag_dict['caption_count'] = tag_dict.get('caption_count', 0) + 1

            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()
                
        del_tag = [tag for tag, count in tag_dict.items() if tag != 'caption_count' and count > tag_dict['caption_count'] * 0.7]
        if args.del_tag:
            print(f"删除的常见标签：{del_tag}")

        for image_path in tqdm(image_paths, desc="處理自然語言"):
            try:
                wdtags = all_wdtags[image_path]
                filtered_tags = [tag for tag in wdtags if not any(d_tag in tag for d_tag in del_tag) or not args.del_tag]
                process_image(image_path, args, filtered_tags, aesthetic_tag, model, processor)
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()

    # 保存最终的元数据
    save_metadata(directory)

def save_metadata(directory):
    global image_meta, image_data
    meta_path = os.path.join(directory, "image_meta.json")
    data_path = os.path.join(directory, "image_data.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(image_meta, f, ensure_ascii=False, indent=4)
    with open(data_path, 'w', encoding='utf-8') as f:
        json.dump(image_data, f, ensure_ascii=False, indent=4)

def load_metadata(directory):
    global image_meta, image_data
    meta_path = os.path.join(directory, "image_meta.json")
    data_path = os.path.join(directory, "image_data.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            image_meta = json.load(f)
    if os.path.exists(data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            image_data = json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="图片标签处理脚本")
    parser.add_argument("--folder_name", action="store_true", help="使用目录名作为标签的一部分")
    parser.add_argument("--del_tag", action="store_true", help="自动删除子目录中的常见标签（出现频率 > 70%）")
    parser.add_argument("directory", type=str, help="处理目录地址")
    args = parser.parse_args()

    # 设置模型和处理器
    model_id = "Desm0nt/Phi-3-HornyVision-128k-instruct"

    # 配置8-bit量化
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    find_and_process_images(args.directory, args, model, processor)
