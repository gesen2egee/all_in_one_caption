import subprocess
import os
import sys
import shutil
import platform
import requests
import argparse
from io import BytesIO
from pathlib import Path
from glob import glob
from tqdm import tqdm
from PIL import Image
from datetime import datetime, timedelta
from transformers import AutoProcessor, AutoModelForCausalLM, CLIPProcessor, CLIPModel
import requests
import copy
import inflect
import re
import random
import torch
import torch.nn.functional as F
from model import longclip
import ftfy
import onnxruntime
import fnmatch
from imgutils.tagging import get_wd14_tags, tags_to_text, drop_blacklisted_tags, drop_basic_character_tags, drop_overlap_tags
from imgutils.validate import anime_dbrating, anime_completeness
import traceback
import json
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
import faiss
import numpy as np
import pandas as pd


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = 'yayayaaa/florence-2-large-ft-moredetailed'
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval().to(device).half()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
p = inflect.engine()
clip_model, clip_preprocess = longclip.load("./checkpoints/Long-ViT-L-14-GmP-ft-state_dict.pt", device=device)
aes_model, aes_preprocessor = convert_v2_5_from_siglip(
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)
aes_model = aes_model.to(torch.bfloat16).to(device)

char_tags = {
    'long hair', 'short hair', 'blue eyes', 'large breasts', 'blonde hair', 'brown hair', 'black hair', 'hair ornament', 'red eyes', 'hat', 'bow', 'animal ears', 'ribbon', 'hair between eyes', 'very long hair', 'twintails', 'medium breasts', 'brown eyes', 'green eyes', 'blue hair', 'purple eyes', 'tail', 'yellow eyes', 'white hair', 'pink hair', 'grey hair', 'ahoge', 'braid', 'hair ribbon', 'purple hair', 'ponytail', 'multicolored hair', 'sidelocks', 'hair bow', 'earrings', 'red hair', 'small breasts', 'hairband', 'horns', 'wings', 'green hair', 'glasses', 'pointy ears', 'hairclip', 'medium hair', 'fang', 'dark skin', 'cat ears', 'blunt bangs', 'hair flower', 'pink eyes', 'hair bun', 'mole', 'hair over one eye', 'rabbit ears', 'orange hair', 'black eyes', 'two-tone hair', 'streaked hair', 'huge breasts', 'halo', 'red bow', 'twin braids', 'side ponytail', 'animal ear fluff', 'red ribbon', 'aqua eyes', 'dark-skinned female', 'parted bangs', 'two side up', 'v-shaped eyebrows', 'grey eyes', 'orange eyes', 'cat tail', 'symbol-shaped pupils', 'eyelashes', 'lips', 'black headwear', 'mole under eye', 'fox ears', 'maid headdress', 'shiny skin', 'fake animal ears', 'black bow', 'single braid', 'neck ribbon', 'black ribbon', 'gradient hair', 'double bun', 'floating hair', 'aqua hair', 'colored skin', 'swept bangs', 'facial hair', 'heterochromia', 'white headwear', 'blue bow', 'fox tail', 'witch hat', 'low twintails', 'one side up', 'headband', 'horse ears', 'beret', 'wavy hair', 'fangs', 'headphones', 'hair intakes', 'facial mark', 'thick eyebrows', 'horse girl', 'headgear', 'muscular male', 'heart-shaped pupils', 'bob cut', 'drill hair', 'sunglasses', 'dark-skinned male', 'light brown hair', 'wolf ears', 'black hairband', 'eyepatch', 'scrunchie', 'white bow', 'demon girl', 'cat girl', 'mob cap', 'magical girl', 'eyes visible through hair', 'demon horns', 'single hair bun', 'high ponytail', 'x hair ornament', 'fox girl', 'blue ribbon', 'grabbing another\'s breast', 'antenna hair', 'hat ribbon', 'crown', 'pink bow', 'spiked hair', 'bat wings', 'ear piercing', 'slit pupils', 'bright pupils', 'monster girl', 'rabbit tail', 'tassel', 'head wings', 'short twintails', 'messy hair', 'horse tail', 'straight hair', 'feathered wings', 'hat bow', 'multiple tails', 'extra ears', 'eyewear on head', 'demon tail', 'dog ears', 'pale skin', 'red headwear', 'white ribbon', 'between breasts', 'colored inner hair', 'hair over shoulder', 'skin fang', 'mole under mouth', 'side braid', 'third eye', 'scar on face', 'baseball cap', 'beard', 'blue headwear', 'peaked cap', 'glowing eyes', 'white pupils', 'semi-rimless eyewear', 'low ponytail', 'twin drills', 'yellow bow', 'wolf tail', 'eyeshadow', 'french braid', 'no headwear', 'tokin hat', 'crossed bangs', 'black wings', 'green bow', 'single horn', 'dragon horns', 'drinking glass', 'hair scrunchie', 'santa hat', 'pink ribbon', 'half updo', 'freckles', 'demon wings', 'topless male', 'single earring', 'low-tied long hair', 'white skin', 'hair rings', 'mature male', 'unworn headwear', 'mole on breast', 'black-framed eyewear', 'short ponytail', 'purple bow', 'round eyewear', 'angel wings', 'goggles on head', 'braided ponytail', 'red-framed eyewear', 'curly hair', 'raised eyebrows', 'hat ornament', 'dragon girl', 'faceless male', 'asymmetrical hair', 'dog tail', 'yellow ribbon', 'top hat', 'sun hat', 'furry female', 'white hairband', 'asymmetrical bangs', 'fake tail', 'blood on face', 'star hair ornament', 'under-rim eyewear', 'white wings', 'mature female', 'multicolored eyes', 'colored eyelashes', 'rabbit girl', 'hoop earrings', 'bouncing breasts', 'unworn hat', 'tentacle hair', 'eyebrows hidden by hair', 'green headwear', 'wolf girl', 'light blue hair', 'mini hat', 'military hat', 'brown headwear', 'dragon tail', 'striped bow', 'tress ribbon', 'pink lips', 'short eyebrows', 'scar across eye', 'mustache', 'folded ponytail', 'dog girl', 'furry male', 'blue skin', 'heart hair ornament', 'muscular female', 'red hairband', 'hime cut', 'mouse ears', 'bandaid on face', 'nurse cap', 'purple ribbon', 'butterfly hair ornament', 'straw hat', 'green ribbon', 'visor cap', 'orange bow', 'stud earrings', 'licking lips', 'bags under eyes', 'low wings', 'long bangs', 'eyeliner', 'red lips', 'fake horns', 'back bow', 'crown braid', 'tail ornament', 'hanging breasts', 'sailor hat', 'hair behind ear', 'cabbie hat', 'flipped hair', 'single side bun', 'absurdly long hair', 'frog hair ornament', 'on head', 'fairy wings', 'star-shaped pupils', 'bird wings', 'hair over eyes', 'cow ears', 'glass', 'food-themed hair ornament', 'pink headwear', 'wrist scrunchie', 'black horns', 'headdress', 'feather hair ornament', 'tinted eyewear', 'ringed eyes', 'mask on head', 'covered eyes', 'horn ornament', 'cow horns', 'mini crown', 'very short hair', 'blue hairband', 'green skin', 'blue halo', 'tiger ears', 'symbol in eye', 'wet hair', 'purple headwear', 'flat cap', 'wine glass', 'snake hair ornament', 'cone hair bun', 'curled horns', 'ice wings', 'bald', 'mechanical halo', 'red horns', 'animal hat', 'raccoon ears', 'pink halo', 'unworn eyewear', 'lolita hairband', 'star earrings', 'crescent hair ornament', 'mouse tail', 'leg ribbon', 'garrison cap', 'white eyes', 'deep skin', 'frilled bow', 'tilted headwear', 'animal on head', 'grey skin', 'ear ornament', 'asymmetrical wings', 'two tails', 'facial tattoo', 'crescent hat ornament', 'rolling eyes', 'toned male', 'no pupils', 'glowing eye', 'fish tail', 'constricted pupils', 'split-color hair', 'leaf hair ornament', 'rabbit hair ornament', 'red skin', 'chest hair', 'leaf on head', 'goat horns', 'necktie between breasts', 'raccoon tail', 'multicolored skin', 'polka dot bow', 'ears through headwear', 'purple skin', 'heart earrings', 'double-parted bangs', 'dark blue hair', 'big hair', 'frilled hairband', 'hair over breasts', 'blank eyes', 'lion ears', 'sparkling eyes', 'tiger tail', 'cow girl', 'huge ahoge', 'tassel earrings', 'star hat ornament', 'braided bun', 'assertive female', 'grey headwear', 'mini top hat', 'arm ribbon', 'braided bangs', 'bear ears', 'shark tail', 'red halo', 'red eyeshadow', 'sheep horns', 'insect wings', 'rimless eyewear', 'bow hairband', 'skin-covered horns', 'yellow halo', 'anchor hair ornament', 'navel hair', 'yellow hairband', 'no eyes', 'ear bow', 'gigantic breasts', 'extra eyes', 'long braid', 'jphones', 'large bow', 'tail ribbon', 'bird ears', 'pink skin', 'cat boy', 'shark girl', 'mouse girl', 'arthropod girl', 'fur hat', 'fur-trimmed headwear', 'raised eyebrow', 'black skin', 'frilled hat', 'striped ribbon', 'waist bow', 'super crown', 'low twin braids', 'crazy eyes', 'cat hair ornament', 'blue wings', 'naked ribbon', 'butterfly wings', 'multiple hair bows', 'demon boy', 'sagging breasts', 'dress bow', 'red scrunchie', 'dragon wings', 'forked eyebrows', 'armpit hair', 'footwear bow', 'purple hairband', 'multiple wings', 'wrist ribbon', 'v over eye', 'red pupils', 'pirate hat', 'towel on head', 'orange headwear', 'bow-shaped hair', 'against glass', 'leg hair', 'mini wings', 'multiple horns', 'carrot hair ornament', 'long eyelashes', 'backwards hat', 'black tail', 'red headband', 'tiger girl', 'mechanical wings', 'white horns', 'musical note hair ornament', 'unaligned breasts', 'orange ribbon', 'heart-shaped eyewear', 'small horns', 'uneven eyes', 'lion tail', 'dangle earrings', 'print bow', 'dog boy', 'raccoon girl', 'blue scrunchie', 'lion girl', 'opaque glasses', 'robot ears', 'christmas ornaments', 'biting own lip', 'framed breasts', 'wizard hat', 'cat ear headphones', 'quad tails', 'bandage over one eye', 'sheep ears', 'arms under breasts', 'diagonal bangs', 'wing hair ornament', 'perky breasts', 'bone hair ornament', 'striped tail', 'cuts', 'medical eyepatch', 'braided hair rings', 'multicolored wings', 'rectangular eyewear', 'purple wings', 'squirrel ears', 'ear ribbon', 'black headband', 'multiple earrings', 'single hair intake', 'sheep girl', 'updo', 'bat hair ornament', 'goggles on headwear', 'horned headwear', 'white scrunchie', 'red eyeliner', 'black scrunchie', 'white headband', 'blue-framed eyewear', 'squirrel tail', 'horn bow', 'green hairband', 'horizontal pupils', 'stained glass', 'wolf boy', 'horseshoe ornament', 'chef hat', 'black lips', 'fox boy', 'multi-tied hair', 'slime girl', 'animal ear piercing', 'shark hair ornament', 'bird girl', 'gold earrings', 'tassel hair ornament', 'feather hair', 'puckered lips', 'orange hairband', 'ankle ribbon', 'flower earrings', 'grey horns', 'crescent earrings', 'yellow pupils', 'drill sidelocks', 'pink scrunchie', 'strap between breasts', 'winged hat', 'ghost tail', 'porkpie hat', 'parted hair', 'squirrel girl', 'police hat', 'over-rim eyewear', 'diagonal-striped bow', 'shower head', 'monkey tail', 'energy wings', 'wide ponytail', 'snowflake hair ornament', 'yellow scrunchie', 'brown ribbon', 'jackal ears', 'bandaged head', 'high side ponytail', 'blue lips', 'clover hair ornament', 'diamond-shaped pupils', 'long pointy ears', 'frilled ribbon', 'broken glass', 'flame-tipped tail', 'turning head', 'tiger boy', 'hair horns', 'skin fangs', 'deer ears', 'looking over eyewear', 'pink-framed eyewear', 'feather earrings', 'broken horn', 'laurel crown', 'large hat', 'flaming eye', 'pom pom hair ornament', 'grey bow', 'disembodied head', 'narrowed eyes', 'no eyewear', 'yellow skin', 'orange scrunchie', 'aqua ribbon', 'large tail', 'averting eyes', 'dreadlocks', 'character hair ornament', 'mechanical horns', 'grey-framed eyewear', 'star halo', 'cocktail glass', 'striped horns', 'multiple moles', 'curtained hair', 'cat hat', 'green lips', 'shako cap', 'buzz cut', 'dragon boy', 'alternate headwear', 'asymmetrical horns', 'short bangs', 'orange-tinted eyewear', 'cracked skin', 'yellow-framed eyewear', 'bandage on face', 'snake tail', 'thigh ribbon', 'afro', 'white-framed eyewear', 'd-pad hair ornament', 'tri tails', 'spread wings', 'school hat', 'tall female', 'bisexual female', 'cone horns', 'pink pupils', 'hair through headwear', 'mechanical tail', 'prehensile hair', 'patchwork skin', 'blue eyeshadow', 'drop earrings', 'veiny breasts', 'two-tone ribbon', 'bear hair ornament', 'bowl hat', 'gold hairband', 'spider girl', 'red-tinted eyewear', 'eyebrow cut', 'animal ear headwear', 'goat ears', 'single hair ring', 'fish hair ornament', 'dixie cup hat', 'leopard ears', 'skull earrings', 'party hat', 'blue horns', 'brushing hair', 'plaid headwear', 'white tail', 'brown hairband', 'blood from eyes', 'fiery hair', 'green halo', 'dyed bangs', 'two-tone eyes', 'wrinkled skin', 'bat ears', 'black halo', 'upturned eyes', 'bowl cut', 'bear girl', 'blue headband', 'yellow wings', 'fish girl', 'fake wings', 'x-shaped pupils', 'fake facial hair', 'flower ornament', 'pillbox hat', 'circle cut', 'yellow horns', 'body hair', 'hair ears', 'bow earrings', 'no wings', 'doughnut hair bun', 'green-framed eyewear', 'magnifying glass', 'eyewear on headwear', 'brown horns', 'plant girl', 'pink eyeshadow', 'multiple braids', 'magatama earrings', 'brown-framed eyewear', 'blue-tinted eyewear', 'cow boy', 'spiked tail', 'purple eyeshadow', 'body freckles', 'multicolored bow', 'heart tail', 'large wings', 'triangle earrings', 'rabbit boy', 'horns through headwear', 'purple-tinted eyewear', 'unusually open eyes', 'sunflower hair ornament', 'lizard tail', 'multicolored horns', 'arm between breasts', 'two-tone headwear', 'panda ears', 'fake mustache', 'expressive hair', 'purple tail', 'drawing bow', 'object through head', 'pink wings', 'blue pupils', 'transparent wings', 'purple horns', 'phoenix crown', 'artificial eye', 'grey ribbon', 'striped headwear', 'goat girl', 'tulip hat', 'crystal hair', 'aqua headwear', 'arched bangs', 'broken halo', 'mechanical ears', 'brown wings', 'leopard tail', 'grey halo', 'no eyebrows', 'notched ear', 'monkey ears', 'pink-tinted eyewear', 'fiery horns', 'uneven horns', 'jaguar ears', 'purple halo', 'sphere earrings', 'bat girl', 'candy hair ornament', 'tapir tail', 'dark halo', 'ruffling hair', 'diving mask on head', 'triangle hair ornament', 'mechanical eye', 'huge bow', 'robot girl', 'sleeve bow', 'rabbit-shaped pupils', 'dice hair ornament', 'button eyes',  'prehensile tail', 'multicolored headwear', 'green wings', 'solid eyes', 'thick lips', 'compass rose halo', 'brown tail', 'strawberry hair ornament', 'food-themed earrings', 'split ponytail', 'two-tone bow', 'neck tassel', 'lion boy', 'two-tone hairband', 'gradient skin', 'polka dot headwear', 'purple scrunchie', 'glowing wings', 'crystal earrings', 'liquid hair', 'orange skin', 'cetacean tail', 'glowing hair', 'smokestack hair ornament', 'panties on head', 'crocodilian tail', 'long tail', 'pearl earrings', 'glowing horns', 'red tail', 'print headwear', 'egg hair ornament', 'side drill', 'blue tail', 'huge eyebrows', 'hair wings', 'snake hair', 'thick eyelashes', 'swim cap', 'grey tail', 'choppy bangs', 'aviator sunglasses', 'pill earrings', 'no tail', 'pink tail', 'owl ears', 'pointy breasts', 'hat over one eye', 'full beard', 'bandaid hair ornament', 'footwear ribbon', 'grey hairband', 'coin hair ornament', 'bucket hat', 'alpaca ears', 'yellow tail', 'low-tied sidelocks', 'weasel ears', 'wrist bow', 'grey wings', 'pursed lips', 'no eyepatch', 'deer girl', 'white headdress', 'green tail', 'wing ornament', 'mismatched eyebrows', 'sleeve ribbon', 'purple-framed eyewear', 'rainbow hair', 'hedgehog ears', 'sideways hat', 'flower on head', 'coke-bottle glasses', 'fish boy', 'orange tail', 'hard hat', 'hair on horn', 'ribbon-trimmed headwear', 'multiple heads', 'flower over eye', 'yellow-tinted eyewear', 'otter ears', 'dashed eyes', 'low-braided long hair', 'arm above head', 'lace-trimmed hairband', 'four-leaf clover hair ornament', 'potara earrings', 'detached hair', 'cephalopod eyes', 'long beard', 'camouflage headwear', 'japari bun', 'star ornament', 'striped hairband', 'hat with ears', 'bunching hair', 'ears visible through hair', 'green scrunchie', 'thick mustache', 'diamond hairband', 'polka dot scrunchie', 'cherry hair ornament', 'bear tail', 'jaguar tail', 'v-shaped eyes', 'rabbit hat', 'thick beard', 'hugging tail', 'no mole', 'green-tinted eyewear', 'ornament', 'diamond hair ornament', 'wavy eyes', 'shell hair ornament', 'heart-shaped eyes', 'chain headband', 'planet hair ornament', 'pearl hair ornament', 'multicolored hairband', 'drop-shaped pupils', 'polka dot ribbon', 'ribbon braid', 'alternate wings', 'hollow eyes', 'unworn eyepatch',  'spaceship hair ornament', 'bowler hat', 'green eyeshadow', 'pumpkin hair ornament', 'spiked hairband', 'flower in eye', 'magical boy', 'behind-the-head headphones', 'plaid ribbon', 'skull ornament', 'bear boy', 'holly hair ornament', 'uneven twintails', 'folded hair', 'pig ears', 'metal skin', 'pumpkin hat', 'cut bangs', 'mole under each eye', 'clock eyes', 'reptile girl', 'hair between breasts', 'alternate hair ornament', 'licking ear', 'braiding hair', 'hexagon hair ornament', 'tri braids', 'animal ear hairband', 'solid circle pupils', 'penis to breast', 'frog girl', 'curly eyebrows', 'star-shaped eyewear', 'fiery wings', 'orange headband', 'scratching head', 'bloodshot eyes', 'green horns', 'green headband', 'single head wing', 'animal head', 'bulging eyes', 'deer tail', 'weasel girl', 'brown lips', 'lifebuoy ornament', 'frilled headwear', 'cable tail', 'safety glasses', 'leopard girl', 'wing ears', 'spade hair ornament', 'white halo', 'weasel tail', 'propeller hair ornament', 'wide oval eyes', 'otter tail', 'pom pom earrings', 'checkered bow', 'fruit hat ornament', 'starfish hair ornament', 'aqua hairband', 'crystal wings', 'object head', 'multicolored tail', 'gradient wings', 'giant male', 'purple pupils', 'torn wings', 'head on head', 'moose ears', 'pointy hat', 'hair over one breast', 'forked tail', 'lightning bolt hair ornament', 'undone neck ribbon', 'hedgehog tail', 'lop rabbit ears', 'sparse chest hair', 'pink horns', 'pokemon ears', 'ankle bow', 'bird boy', 'bandaid on head', 'implied extra ears', 'hat tassel', 'fruit on head', 'starry hair', 'sparkle hair ornament', 'long ribbon', 'rice hat', 'washing hair', 'anchor earrings', 'asymmetrical sidelocks', 'mini witch hat', 'unworn hair ornament', 'heart hair', 'arthropod boy', 'detached ahoge', 'large ears', 'aviator cap', 'monkey boy', 'female service cap', 'moth girl', 'glove bow', 'bangs', 'shiny hair', 'light purple hair', 'oni horns', 'pillow hat', 'polos crown', 'light green hair', 'monocle hair ornament', 'dark green hair', 'pouty lips', 'bunny-shaped pupils', 'bunny hatester cap', 'detached wings', 'solid oval eyes', 'cube hair ornament', 'heart ahoge', 'cross-shaped pupils', 'cross hair ornament', 'pointy hair', 'very dark skin', 'aqua bow', 'front ponytail', 'pink hairband', 'skull hair ornament', 'side braids', 'tail bow', 'cross earrings', 'horn ribbon', 'cow tail', 'floppy ears', 'two-tone skin', 'plaid bow', 'purple lips', 'single sidelock', 'solid circle eyes', 'yellow headwear', 'faceless female', 'single wing', 'brown bow', 'medium bangs', 'red wings', 'monster boy', 'mismatched pupils', 'cowboy hat', 'flower-shaped pupils', 'bird tail', 'gradient eyes', 'bursting breasts', 'animal ear head', 'hair bobbles', 'prosthetic leg', 'centaur'
}

clothing_tags = {
    'nude', 'topless', 'bottomless', 'crotchless', 'one breast out', 'breasts out', 'penis', 'anal ball wear', 'futanari', 'wings', 'stomach', 'anus', 'pussy', 'cameltoe', 'testicles', 'tanlines', 'sailor collar', 'anal object insertion', 'monoglove', 'nipple piercing','no panties', 'shirt', 'skirt', 'long sleeves', 'hair ornament', 'gloves', 'dress', 'thighhighs', 'hat', 'bow', 'navel', 'ribbon', 'cleavage', 'jewelry', 'bare shoulders', 'underwear', 'jacket', 'school uniform', 'collarbone', 'white shirt', 'panties', 'swimsuit', 'hair ribbon', 'short sleeves', 'hair bow', 'pantyhose', 'earrings', 'bikini', 'pleated skirt', 'frills', 'hairband', 'boots', 'open clothes', 'necktie', 'detached sleeves', 'shorts', 'japanese clothes', 'shoes', 'sleeveless', 'black gloves', 'alternate costume', 'collared shirt', 'choker', 'barefoot', 'socks', 'glasses', 'pants', 'serafuku', 'puffy sleeves', 'hairclip', 'belt', 'black thighhighs', 'elbow gloves', 'midriff', 'white gloves', 'bowtie', 'hood', 'black skirt', 'hair flower', 'official alternate costume', 'wide sleeves', 'miniskirt', 'fingerless gloves', 'black footwear', 'kimono', 'white dress', 'off shoulder', 'necklace', 'striped clothes', 'nail polish', 'star \(symbol\)', 'bag', 'black dress', 'scarf', 'cape', 'white thighhighs', 'bra', 'armor', 'vest', 'open jacket', 'halo', 'apron', 'red bow', 'white panties', 'leotard', 'coat', 'black jacket', 'high heels', 'collar', 'sweater', 'bracelet', 'uniform', 'red ribbon', 'crop top', 'black shirt', 'puffy short sleeves', 'blue skirt', 'black pantyhose', 'neckerchief', 'sleeves past wrists', 'fur trim', 'see-through', 'wrist cuffs', 'maid', 'strapless', 'zettai ryouiki', 'clothing cutout', 'black headwear', 'plaid', 'torn clothes', 'one-piece swimsuit', 'sash', 'maid headdress', 'sleeveless shirt', 'short shorts', 'bare arms', 'sleeveless dress', 'ascot', 'black panties', 'cosplay', 'kneehighs', 'bare legs', 'thigh strap', 'black bow', 'covered navel', 'hoodie', 'neck ribbon', 'black ribbon', 'detached collar', 'tattoo', 'black choker', 'dress shirt', 'buttons', 'open shirt', 'sideboob', 'bell', 'military', 'mask', 'skindentation', 'capelet', 'bodysuit', 'blue dress', 'black pants', 'no bra', 'black bikini', 'white headwear', 'red skirt', 'blue bow', 'turtleneck', 'underboob', 'witch hat', 'highleg', 'military uniform', 'headband', 'black shorts', 'beret', 'side-tie bikini bottom', 'brown footwear', 'halterneck', 'chain', 'playboy bunny', 'headphones', 'piercing', 'white jacket', 'white socks', 'blush stickers', 'chinese clothes', 'white bikini', 'no shoes', 'plaid skirt', 'thigh boots', 'white footwear', 'headgear', 'sandals', 'floral print', 'garter straps', 'short dress', 'sunglasses', 'obi', 'red dress', 'hood down', 'frilled dress', 'cleavage cutout', 'white skirt', 'blue shirt', 'hair tubes', 'ring', 'bound', 'blue jacket', 'black socks', 'black hairband', 'eyepatch', 'scrunchie', 'white bow', 'formal', 'mob cap', 'cardigan', 'backpack', 'frilled skirt', 'tank top', 'blazer', 'suspenders', 'helmet', 'suit', 'feathers', 'x hair ornament', 'underwear only', 'blue ribbon', 'frilled sleeves', 'school swimsuit', 'cross', 'hat ribbon', 'denim', 'crown', 'knee boots', 'pink bow', 'red necktie', 'tiara', 'juliet sleeves', 'polka dot', 'black nails', 'ear piercing', 'wing collar', 'lingerie', 'animal print', 'red shirt', 'undressing', 'striped thighhighs', 'blue sailor collar', 'sneakers', 'black leotard', 'white border', 't-shirt', 'tassel', 'gem', 'red footwear', 'white apron', 'bondage', 'red bowtie', 'hair bobbles', 'lipstick', 'green skirt', 'goggles', 'shoulder armor', 'brooch', 'black bra', 'fishnets', 'loafers', 'crescent', 'towel', 'single thighhigh', 'pink dress', 'strapless leotard', 'hat bow', 'grey shirt', 'black necktie', 'no pants', 'eyewear on head', 'bike shorts', 'hooded jacket', 'armband', 'casual', 'revealing clothes', 'red headwear', 'gauntlets', 'white ribbon', 'rope', 'sheath', 'china dress', 'ribbon trim', 'pink panties', 'adapted costume', 'multicolored clothes', 'wristband', 'hakama', 'blouse', 'puffy long sleeves', 'veil', 'red jacket', 'red nails', 'lace trim', 'waist apron', 'skirt set', 'pelvic curtain', 'strapless dress', 'baseball cap', 'string bikini', 'striped panties', 'blue headwear', 'bridal gauntlets', 'cloak', 'peaked cap', 'highleg leotard', 'red neckerchief', 'purple dress', 'side-tie panties', 'semi-rimless eyewear', 'white pantyhose', 'jingle bell', 'hand fan', 'grey skirt', 'front-tie top', 'bow panties', 'buckle', 'clothes writing', 'pom pom \(clothes\)', 'micro bikini', 'yellow bow', 'maid apron', 'sleeves past fingers', 'hood up', 'corset', 'neck bell', 'blue nails', 'skin tight', 'o-ring', 'hakama skirt', 'black belt', 'lace', 'no headwear', 'tokin hat', 'white sleeves', 'cropped jacket', 'bikini top only', 'brown gloves', 'restrained', 'red gloves', 'mary janes', 'spikes', 'blue bikini', 'side slit', 'black coat', 'camisole', 'strap slip', 'armlet', 'green bow', 'hair scrunchie', 'sleeves rolled up', 'gold trim', 'blue necktie', 'santa hat', 'black sailor collar', 'pink skirt', 'single glove', 'pink ribbon', 'white sailor collar', 'zipper', 'open coat', 'blue shorts', 'pencil skirt', 'pink shirt', 'pendant', 'ribbed sweater', 'topless male', 'high heel boots', 'track jacket', 'single earring', 'frilled apron', 'asymmetrical legwear', 'sweater vest', 'cross-laced footwear', 'headset', 'black vest', 'frilled bikini', 'beads', 'pocket', 'vertical-striped clothes', 'green dress', 'unworn headwear', 'frilled shirt collar', 'black-framed eyewear', 'brown pantyhose', 'thong', 'red bikini', 'purple bow', 'long skirt', 'high-waist skirt', 'round eyewear', 'blue footwear', 'crossdressing', 'white bra', 'cuffs', 'gym uniform', 'purple skirt', 'yellow shirt', 'goggles on head', 'black bowtie', 'red-framed eyewear', 'epaulettes', 'ribbon-trimmed sleeves', 'santa costume', 'high collar', 'brown jacket', 'denim shorts', 'brown skirt', 'hat ornament', 'panties under pantyhose', 'buruma', 'white pants', 'school bag', 'nontraditional miko', 'pouch', 'thighband pantyhose', 'black serafuku', 'dual wielding', 'red scarf', 'fur collar', 'green jacket', 'sailor dress', 'robe', 'garter belt', 'white shorts', 'pauldrons', 'competition swimsuit', 'yellow ribbon', 'lolita fashion', 'top hat', 'sun hat', 'white hairband', 'watch', 'blood on face', 'blue one-piece swimsuit', 'turtleneck sweater', 'star hair ornament', 'white kimono', 'sports bra', 'under-rim eyewear', 'grey jacket', 'shiny clothes', 'white coat', 'striped shirt', 'impossible clothes', 'jeans', 'circlet', 'belt buckle', 'shoulder bag', 'green shirt', 'partially fingerless gloves', 'sarashi', 'striped bikini', 'crystal', 'shawl', 'bandaged arm', 'hairpin', 'hoop earrings', 'sheathed', 'unworn hat', 'blue bowtie', 'green headwear', 'yukata', 'mini hat', 'white leotard', 'purple shirt', 'military hat', 'breastplate', 'pajamas', 'brown headwear', 'black sleeves', 'blue pants', 'bespectacled', 'shirt tucked in', 'striped bow', 'tress ribbon', 'mouth mask', 'handbag', 'blue panties', 'animal hood', 'hugging object', 'collared dress', 'scar across eye', 'backless outfit', 'meme attire', 'tabard', 'long dress', 'sportswear', 'torn pantyhose', 'fishnet pantyhose', 'ofuda', 'off-shoulder dress', 'forehead mark', 'front-tie bikini top', 'pink footwear', 'heart hair ornament', 'lab coat', 'panties aside', 'black one-piece swimsuit', 'red hairband', 'hair bell', 'skirt hold', 'wedding dress', 'blue kimono', 'bandaid on face', 'nurse cap', 'purple ribbon', 'bloomers', 'butterfly hair ornament', 'red cape', 'hat flower', 'hair tie', 'blue gloves', 'arrow \(symbol\)', 'purple nails', 'miko', 'pasties', 'straw hat', 'green ribbon', 'bandana', 'black bodysuit', 'blue leotard', 'visor cap', 'winter uniform', 'covering own mouth', 'orange bow', 'drawstring', 'yellow ascot', 'red vest', 'stud earrings', 'blindfold', 'fur-trimmed jacket', 'brown belt', 'off-shoulder shirt', 'pink jacket', 'center frills', 'shrug \(clothing\)', 'panties around one leg', 'black cape', 'pink bra', 'criss-cross halter', 'anklet', 'center opening', 'white sweater', 'headpiece', 'gag', 'suspender skirt', 'highleg panties', 'beanie', 'grabbing from behind', 'spaghetti strap', 'bandeau', 'white scarf', 'pink bikini', 'folding fan', 'underbust', 'back bow', 'white one-piece swimsuit', 'emblem', 'adjusting eyewear', 'double-breasted', 'brown thighhighs', 'eighth note', 'gakuran', 'geta', 'sailor hat', 'nun', 'black collar', 'tabi', 'name tag', 'cabbie hat', 'microskirt', 'pinafore dress', 'anchor symbol', 'arm warmers', 'petticoat', 'frog hair ornament', 'arm tattoo', 'sailor shirt', 'bodystocking', 'highleg swimsuit', 'frilled shirt', 'tracen school uniform', 'habit', 'body fur', 'yellow jacket', 'unbuttoned', 'winter clothes', 'bikini under clothes', 'wristwatch', 'lowleg', 'bound wrists', 'summer uniform', 'gothic lolita', 'nipple slip', 'cow print', 'brown shirt', 'lace-up boots', 'cheerleader', 'checkered clothes', 'grey pants', 'food-themed hair ornament', 'brown dress', 'swimsuit under clothes', 'taut clothes', 'hitodama', 'witch', 'vambraces', 'overalls', 'pom pom \(cheerleading\)', 'pink headwear', 'wrist scrunchie', 'black sweater', 'mittens', 'blue thighhighs', 'military vehicle', 'hair stick', 'ankle boots', 'layered sleeves', 'toeless legwear', 'print bikini', 'food in mouth', 'headdress', 'breast pocket', 'print kimono', 'feather hair ornament', 'gohei', 'tinted eyewear', 'blood on clothes', 'wedding ring', 'bangle', 'cable', 'zipper pull tab', 'purple bikini', 'unzipped', 'sundress', 'pleated dress', 'purple jacket', 'crotch seam', 'armored dress', 'armored boots', 'mask on head', 'midriff peek', 'red kimono', 'black kimono', 'ninja', 'purple gloves', 'grey dress', 'bridal garter', 'tube top', 'bare back', 'horn ornament', 'halloween costume', 'open fly', 'mini crown', 'nurse', 'white capelet', 'animal costume', 'shoulder cutout', 'naked shirt', 'half gloves', 'blue hairband', 'brown pants', 'red choker', 'strap gap', 'japanese armor', 'male underwear', 'black hoodie', 'uneven legwear', 'blue halo', 'falling petals', 'blue vest', 'haori', 'office lady', 'holster', 'waitress', 'thighlet', 'earmuffs', 'red thighhighs', 'purple thighhighs', 'open cardigan', 'yellow bikini', 'two-tone dress', 'bow bra', 'naval uniform', 'bridal veil', 'purple headwear', 'flat cap', 'pilot suit', 'paw gloves', 'tight clothes', 'randoseru','red panties', 'blue neckerchief', 'rigging', 'ear covers', 'triangular headpiece', 'snake hair ornament', 'star print', 'paw print', 'pink kimono', 'purple panties', 'fishnet thighhighs', 'heart of string', 'shibari', 'red hakama', 'muneate', 'yellow dress', 'sarong', 'red shorts', 'slippers', 'mismatched legwear', 'mechanical halo', 'highleg bikini', 'bracer', 'yellow neckerchief', 'covering crotch', 'cropped shirt', 'hakama short skirt', 'sleeve cuffs', 'red ascot', 'hip vent', 'animal hat', 'sack', 'arm strap', 'navel cutout', 'kita high school uniform', 'bare pectorals', 'scar on cheek', 'white cape', 'pink halo', 'white choker', 'partially unbuttoned', 'unworn eyewear', 'shoulder tattoo', 'condom wrapper', 'lolita hairband', 'facepaint', 'knee pads', 'backless dress', 'green nails', 'casual one-piece swimsuit', 'torn thighhighs', 'print shirt', 'animal nose', 'tied shirt', 'layered dress', 'string panties', 'bandaged leg', 'sleeveless turtleneck', 'bikini skirt', 'red leotard', 'layered skirt', 'leggings', 'green bikini', 'star earrings', 'crescent hair ornament', 'scabbard', 'brown coat', 'pantyhose under shorts', 'tasuki', 'ribbon choker', 'lace-trimmed panties', 'lace-trimmed legwear', 'leg ribbon', 'tate eboshi', 'plugsuit', 'hat feather', 'garrison cap', 'short kimono', 'magatama', 'parasol', 'fur-trimmed coat', 'leather', 'asymmetrical clothes', 'lace-trimmed bra', 'head wreath', 'shimenawa', 'frilled bow', 'tilted headwear', 'single bare shoulder', 'animal on head', 'yellow skirt', 'sample watermark', 'purple footwear', 'blue bra', 'raglan sleeves', 'harness', 'clothes around waist', 'tiger print', 'blue scarf', 'grey footwear', 'asymmetrical gloves', 'ear ornament', 'o-ring top', 'asymmetrical wings', 'naked towel', 'o-ring bikini', 'forehead jewel', 'stitches', 'white collar', 'facial tattoo', 'purple kimono', 'headphones around neck', 'pantyhose pull', 'white tank top', 'cross necklace', 'french kiss', 'crescent hat ornament', 'crop top overhang', 'short over long sleeves', 'black scarf', 'open kimono', 'oversized clothes', 'black neckerchief', 'two-sided fabric', 'strap', 'fur-trimmed sleeves', 'platform footwear', 'sakuragaoka high school uniform', 'one-piece tan', 'white hoodie', 'striped dress', 'orange shirt', 'blue cape', 'pectoral cleavage', 'white belt', 'leaf hair ornament', 'handcuffs', 'red pants', 'navel piercing', 'rabbit hair ornament', 'spiked collar', 'shackles', 'neck ring', 'pink bowtie', 'brown sweater', 'pants pull', 'yellow necktie', 'fox mask', 'belt pouch', 'spiked bracelet', 'unworn shoes', 'red coat', 'black tank top', 'grey shorts', 'head scarf', 'greaves', 'leaf on head', 'necktie between breasts', 'micro shorts', 'slingshot swimsuit', 'camouflage', 'east asian architecture', 'arm cannon', 'polka dot bow', 'striped necktie', 'naked apron', 'red capelet', 'green vest', 'bandaid on leg', 'white bowtie', 'fur-trimmed gloves', 'no shirt', 'loose socks', 'body writing', 'badge', 'multicolored jacket', 'shorts under skirt', 'heart earrings', 'black capelet', 'bead necklace', 'babydoll', 'arm guards', 'green necktie', 'ooarai school uniform', 'brown shorts', 'undershirt', 'frilled hairband', 'grey sweater', 'orange skirt', 'see-through sleeves', 'hooded cloak', 'bonnet', 'green shorts', 'huge weapon', 'blue choker', 'skirt pull', 'brown cardigan', 'gym shirt', 'earphones', 'bra lift', 'blue coat', 'bead bracelet', 'fundoshi', 'iron cross', 'cutoffs', 'white ascot', 'smoking pipe', 'cow girl', 'bikini pull', 'hachimaki', 'white bodysuit', 'plaid vest', 'pink gloves', 'shoulder pads', 'argyle clothes', 'tassel earrings', 'drink can', 'bikini armor', 'black sports bra', 'yellow bowtie', 'pink thighhighs', 'star hat ornament', 'latex', 'purple bowtie', 'torn shirt', 'animal collar', 'grey headwear', 'sweater dress', 'yin yang', 'bobby socks', 'grey gloves', 'blue sweater', 'diamond \(shape\)', 'fringe trim', 'green footwear', 'striped pantyhose', 'mini top hat', 'arm garter', 'aqua necktie', 'frilled panties', 'off-shoulder sweater', 'arm ribbon', 'fur-trimmed capelet', 'broom riding', 'untied bikini', 'police', 'red collar', 'ear blush', 'object on head', 'multicolored dress', 'single shoe', 'bikini bottom only', 'dirty', 'red halo', 'yellow footwear', 'green kimono', 'vertical-striped thighhighs', 'red sweater', 'heart brooch', 'rimless eyewear', 'bow hairband', 'male swimwear', 'frilled bra', 'blue gemstone', 'chest tattoo', 'green panties', 'grey thighhighs', 'zouri', 'collared jacket', 'race queen', 'fur-trimmed dress', 'yellow halo', 'traditional bowtie', 'anchor hair ornament', 'yellow hairband', 'ear bow', 'unworn panties', 'fur-trimmed cape', 'long coat', 'ball gag', 'v-neck', 'chest jewel', 'unworn skirt', 'extra arms', 'topknot', 'blue hoodie', 'sailor senshi uniform', 'cum string', 'between fingers', 'sleeveless jacket', 'green pants', 'white neckerchief', 'superhero', 'single sock', 'brown vest', 'print panties', 'waist cape', 'shirt pull', 'green gloves', 'hooded coat', 'clothed male nude female', 'jester cap', 'open vest', 'plaid shirt', 'weapon on back', 'monocle', 'black blindfold', 'pink sweater', 'track suit', 'pink choker', 'pillow hug', 'cube hair ornament', 'frilled collar', 'floating object', 'black suit', 'nightgown', 'frilled choker', 'striped bowtie', 'white necktie', 'jumpsuit', 'faulds', 'police uniform', 'uwabaki', 'bride', 'blood on hands', 'downblouse', 'yellow nails', 'kariginu', 'sepia', 'cross hair ornament', 'weapon over shoulder', 'competition school swimsuit', 'over-kneehighs', 'aqua bow', 'leotard under clothes', 'lowleg panties', 'pink hairband', 'sweater lift', 'taut shirt', 'bandaid on nose', 'skull hair ornament', 'red bodysuit', 'heart print', 'heart cutout', 'plaid scarf', 'side braids', 'open hoodie', 'charm \(object\)', 'red bra', 'purple bra', 'chest harness', 'adjusting headwear', 'short necktie', 'blood splatter', 'cross earrings', 'coat on shoulders', 'halter dress', 'tokiwadai school uniform', 'horn ribbon', 'multiple rings', 'dog tags', 'two-tone skin', 'plaid bow', 'dougi', 'chaldea uniform', 'leg warmers', 'loincloth', 'frilled thighhighs', 'bra pull', 'pinstripe pattern', 'purple necktie', 'military jacket', 'meat', 'forehead protector', 'two-tone shirt', 'yellow headwear', 'lanyard', 'otonokizaka school uniform', 'leg tattoo', 'sleeves past elbows', 'heart pasties', 'bound legs', 'white camisole', 'thighhighs under boots', 'belt collar', 'brown bow', 'thigh holster', 'bubble skirt', 'striped socks', 'blue butterfly', 'waistcoat', 'cowboy hat', 'print dress', 'asymmetrical sleeves', 'polka dot panties', 'lapels', 'blue bodysuit', 'knight', 'torn pants', 'star in eye', 'print pantyhose', 'grey pantyhose', 'gradient eyes', 'cow print bikini', 'thong bikini', 'bra visible through clothes', 'flag print', 'unmoving pattern', 'white bloomers', 'full armor', 'panty peek', 'popped collar', 'chest sarashi', 'wet panties', 'yellow gloves', 'pearl necklace', 'large bow', 'multicolored nails', 'old school swimsuit', 'no socks', 'green bowtie', 'flower knot', 'blue sleeves', 'bird ears', 'stirrup legwear', 'maebari', 'orange jacket', 'purple leotard', 'partially unzipped', 'aiguillette', 'dolphin shorts', 'vertical-striped shirt', 'fur hat', 'obijime', 'fur-trimmed headwear', 'bra strap', 'strap pull', 'orange dress', 'blue buruma', 'interface headset', 'pink necktie', 'toeless footwear', 'crotchless', 'suit jacket', 'striped scarf', 'mismatched gloves', 'grey vest', 'blue socks', 'medium skirt', 'argyle legwear', 'frilled hat', 'tengu-geta', 'striped skirt', 'showgirl skirt', 'striped ribbon', 'waist bow', 'quiver', 'red buruma', 'falling leaves', 'closed umbrella', 'button gap', 'black cloak', 'fedora', 'cable knit', 'grabbing own ass', 'backboob', 'single leg pantyhose', 'grey cardigan', 'elbow pads', 'print thighhighs', 'winter coat', 'yellow sweater', 'serval print', 'orange flower', 'see-through cleavage', 'back cutout', 'two-tone jacket', 'single elbow glove', 'cat hair ornament', 'yellow shorts', 'cake slice', 'orange bikini', 'torn dress', 'black ascot', 'print skirt', 'multiple hair bows','track pants', 'bikini bottom aside', 'towel around neck', 'blood on weapon', 'dress bow', 'pink neckerchief', 'red scrunchie', 'grey hoodie', 'business suit', 'baggy pants', 'metal collar', 'unworn mask', 'bandaged hand', 'white vest', 'hose', 'frilled hair tubes', 'impossible shirt', 'unbuttoned shirt', 'scar on chest', 'blue capelet', 'hairpods', 'footwear bow', 'pocket watch', 'purple hairband', 'body markings', 'lightning bolt symbol', 'wrist ribbon', 'single sleeve', 'surgical mask', 'unconventional maid', 'open collar', 'bustier', 'brown bag', 'pirate hat', 'red rope', 'anal tail', 'sleeveless kimono', 'leather jacket', 'trench coat', 'orange bowtie', 'pink cardigan', 'planted sword', 'orange headwear', 'loose necktie', 'bodypaint', 'brown scarf', 'green sailor collar', 'pink scarf', 'red sailor collar', 'fur-trimmed boots', 'multicolored skirt', 'gym shorts', 'carrot hair ornament', 'power armor', 'covered collarbone', 'silk', 'sailor', 'heart necklace', 'frilled gloves', 'yellow panties', 'blue ascot', 'stomach tattoo', 'underboob cutout', 'leaf print', 'arm belt', 'backwards hat', 'cat cutout', 'polka dot bikini', 'crescent pin', 'red headband', 'strapless bikini', 'red sleeves', 'earclip', 'earpiece', 'checkered skirt', 'white robe', 'reverse outfit', 'thorns', 'grey coat', 'cat lingerie', 'purple pantyhose', 'pince-nez', 'musical note hair ornament', 'frilled pillow', 'torn skirt', 'thong leotard', 'two-tone gloves', 'red belt', 'heart choker', 'orange ribbon', 'black bag', 'scar on arm', 'heart-shaped eyewear', 'policewoman', 'ribbon-trimmed legwear', 'circle', 'hooded capelet', 'blue belt', 'bat print', 'hair beads', 'demon slayer uniform', 'soccer uniform', 'striped jacket', 'pink shorts', 'grey sailor collar', 'purple vest', 'virgin killer sweater', 'improvised gag', 'reverse bunnysuit', 'white feathers', 'dangle earrings', 'food print', 'bike shorts under skirt', 'print bow', 'yellow scarf', 'tight pants', 'number tattoo', 'kiseru', 'sleeves pushed up', 'cat hood', 'print gloves', 'soul gem', 'glowing weapon', 'blue scrunchie', 'leotard aside', 'bikini tan', 'opaque glasses', 'kunai', 'pubic hair peek', 'no legwear', 'bandaids on nipples', 'idol clothes', 'orange necktie', 'christmas ornaments', 'pink leotard', 'multiple belts', 'sideless outfit', 'flip-flops', 'bandaid on cheek', 'wizard hat', 'aran sweater', 'goth fashion', 'single detached sleeve', 'scar on nose', 'cat ear headphones', 'employee uniform', 'plaid dress', 'tomoe \(symbol\)', 'black armor', 'jacket partially removed', 'asymmetrical footwear', 'red sash', 'vertical-striped dress', 'orange nails', 'leopard print', 'rose print', 'costume', 'green gemstone', 'bandage over one eye', 'breast curtains', 'unworn bra', 'purple bodysuit', 'ribbed shirt', 'blue pantyhose', 'spacesuit', 'pumps', 'two-tone skirt', 'swimsuit aside', 'wing hair ornament', 'green cape', 'spread toes', 'maid bikini', 'bone hair ornament', 'tail through clothes', 'green leotard', 'green scarf', 'furisode', 'presenting armpit', 'jacket around waist', 'green sweater', 'multiple crossover', 'gloved handjob', 'medical eyepatch', 'purple choker', 'swim trunks', 'obiage', 'frilled one-piece swimsuit', 'kote', 'coattails', 'braided hair rings', 'non-humanoid robot', 'orange footwear', 'yellow kimono', 'multicolored wings', 'pantylines', 'blue hakama', 'rectangular eyewear', 'slave', 'skirt suit', 'aqua bowtie', 'feather boa', 'o-ring bottom', 'purple wings', 'american flag legwear', 'clothes grab', 'letterman jacket', 'breast tattoo', 'chain necklace', 'single kneehigh', 'ear ribbon', 'barbell piercing', 'black headband', 'green coat', 'mole on thigh', 'multiple earrings', 'domino mask', 'layered clothes', 'purple pants', 'grey panties', 'kanzashi', 'wet swimsuit', 'egyptian', 'white wrist cuffs', 'frilled kimono', 'eyepatch bikini', 'bag charm', 'fur-trimmed hood', 'bat hair ornament', 'diagonal-striped clothes', 'goggles on headwear', 'burn scar', 'neck tattoo', 'paradis military uniform', 'open dress', 'white scrunchie', 'capri pants', 'bra peek', 'button badge', 'rudder footwear', 'red eyeliner', 'wiffle gag', 'black scrunchie', 'white headband', 'blue-framed eyewear', 'nightcap', 'skates', 'bird on head', 'american flag dress', 'layered bikini', 'diamond button', 'print bowtie', 'black camisole', 'trefoil', 'black cardigan', 'horn bow', 'naked sweater', 'pink apron', 'okobo', 'gas mask', 'green hairband', 'two-tone swimsuit', 'rei no himo', 'yoga pants', 'yellow cardigan', 'black robe', 'horseshoe ornament', 'hand over own mouth', 'basketball \(object\)', 'green bra', 'hagoromo', 'carrying person', 'tie clip', 'chef hat', 'santa dress', 'high-waist shorts', 'green thighhighs', 'uranohoshi school uniform', 'orange gloves', 'black mask', 'heart tattoo', 'rabbit hood', 'four-leaf clover', 'excalibur \(fate/stay night\)', 'tunic', 'rabbit print', 'purple sleeves', 'dragon print', 'spider web print', 'sleeveless sweater', 'ankle socks', 'earbuds', 'grey socks', 'sleepwear', 'qingdai guanmao', 'medal', 'slime girl', 'shark hair ornament', 'shibari over clothes', 'ribbed dress', 'gold earrings', 'two-tone bikini', 'lace panties', 'tassel hair ornament', 'chained', 'loose belt', 'triangle', 'food on head', 'mandarin collar', 'soldier', 'sailor bikini', 'striped bra', 'orange hairband', 'ankle ribbon', 'flower earrings', 'strappy heels', 'torn sleeves', 'plunging neckline', 'flats', 'red pantyhose', 'blue serafuku', 'white cloak', 'torn bodysuit', 'crescent earrings', 'purple umbrella', 'harem outfit', 'pink scrunchie', 'hands in opposite sleeves', 'strap between breasts', 'winged hat', 'crotch rope', 'shirt tug', 'meiji schoolgirl uniform', 'purple cape', 'wa maid', 'undersized clothes', 'orange bodysuit', 'paw shoes', 'cross scar', 'suspender shorts', 'see-through dress', 'eye mask', 'female pov', 'gold chain', 'microdress', 'blue cardigan', 'porkpie hat', 'breast slip', 'clitoral hood', 'white sports bra', 'thumb ring', 'spade \(shape\)', 'police hat',  'white cardigan', 'over-rim eyewear', 'diagonal-striped bow', 'covering own eyes', 'heart-shaped pillow', 'hanfu', 'hugging doll', 'unworn helmet', 'multi-strapped bikini bottom', 'multicolored swimsuit', 'snowflake hair ornament', 'purple shorts', 'covered abs', 'blue overalls', 'snowflake print', 'head-mounted display', 'rubber boots', 'checkered scarf', 'g-string', 'breastless clothes', 'black apron', 'purple scarf', 'purple coat', 'american flag bikini', 'striped pants', 'jirai kei', 'black corset', 'yellow scrunchie', 'brown ribbon', 'whistle around neck', 'heart in mouth', 'side cutout', 'bandaged head', 'mismatched bikini', 'black feathers', 'red socks', 'neck ruff', 'feather trim', 'pirate', 'white nails', 'tongue piercing', 'shoulder spikes', 'holding panties', 'o-ring choker', 'clover hair ornament', 'evening gown', 'scar on forehead', 'duffel bag', 'pink bodysuit', 'alternate legwear', 'chinese knot', 'frilled ribbon', 'orange shorts', 'talisman', 'enpera', 'two-footed footjob', 'torn cape', 'holding condom', 'bow bikini', 'fashion', 'orange kimono', 'red armband', 'strapless shirt', 'black hakama', 'holding sign', 'shimakaze \(kancolle\) \(cosplay\)', 'pink hoodie', 'gem uniform \(houseki no kuni\)', 'asticassia school uniform', 'puffy detached sleeves', 'aqua dress', 'egyptian clothes', 'uneven gloves', 'medium dress', 'fur coat', 'looking over eyewear', 'bodysuit under clothes', 'open shorts', 'flower wreath', 'skirt tug', 'tailcoat', 'pink-framed eyewear', 'blue apron', 'chest belt', 'year of the tiger', 'red cloak', 'year of the rabbit', 'lace-trimmed dress', 'feather earrings', 'holding sack', 'raincoat', 'santa bikini', 'brown sweater vest', 'laurel crown', 'large hat', 'yugake', 'holding cat', 'pith helmet', 'brown capelet', 'open-chest sweater', 'pom pom hair ornament', 'arabian clothes', 'seamed legwear', 'ear tag', 'grey bow', 'chemise', 'aqua skirt', 'pant suit', 'hooded cape', 'single ear cover', 'wrestling outfit', 'holding paintbrush', 'cat ear panties', 'red umbrella', 'fur scarf', 'pink sleeves', 'safety pin', 'flower \(symbol\)', 'zero suit', 'bomber jacket', 'no eyewear', 'patterned clothing', 'naoetsu high school uniform', 'brown kimono', 'rice bowl', 'hexagram', 'purple sweater', 'high-waist pants', 'tuxedo', 'overskirt', 'chain leash', 'crossed bandaids', 'bit gag', 'orange scrunchie', 'negligee', 'green choker', 'cyberpunk', 'borrowed clothes', 'power symbol', 'diagonal-striped necktie', 'back tattoo', 'brown sailor collar', 'black leggings', 'hand on hilt', 'aqua ribbon', 'ainu clothes', 'yellow vest', 'smiley face', 'red hoodie', 'white suit', 'pink pants', 'bandaid on arm', 'wreath', 'carrot necklace', 'character hair ornament', 'grey-framed eyewear', 'ribbon-trimmed skirt', 'sun symbol', 'see-through legwear', 'bare hips', 'uneven sleeves', 'steam censor', 'belly chain', 'star halo', 'white male underwear', 'samurai', 'turtleneck dress', 'ankle cuffs', 'untied panties', 'aqua bikini', 'holding handheld game console', 'striped sleeves', 'cat hat', 'aqua shirt', 'cat print', 'red bag', 'kitauji high school uniform', 'st\. gloriana\'s school uniform', 'black sash', 'frilled capelet', 'character print', 'shako cap', 'diadem', 'impossible bodysuit', 'nose piercing', 'orange choker', 'boxers', 'holding clipboard', 'brown cape', 'holding syringe', 'torn shorts', 'very long sleeves', 'turban', 'transparent umbrella', 'skirt around one leg', 'butterfly print', 'mask pull', 'naked jacket', 'red one-piece swimsuit', 'oni mask', 'denim skirt', 'spandex', 'ribbed legwear', 'notched lapels', 'fanny pack', 'tam o\' shanter', 'green hoodie', 'white sash', 'magatama necklace', 'star choker', 'single pauldron', 'purple sailor collar', 'diving mask', 'crotchless panties', 'strapless bra', 'vertical-striped skirt', 'orange-tinted eyewear', 'insignia', 'yellow-framed eyewear', 'legwear garter', 'feather-trimmed sleeves', 'traditional nun', 'ooarai military uniform', 'naked coat', 'plaid bikini', 'studded belt', 'bandage on face', 'starry sky print', 'shared scarf', 'pendant choker', 'impossible leotard', 'pink bag', 'korean clothes', 'prayer beads', 'nontraditional playboy bunny', 'dirty clothes', 'holding pencil', 'shuuchiin academy school uniform', 'thigh ribbon', 'fur-trimmed legwear', 'oversized shirt', 'holding lantern', 'two-sided cape', 'assault visor', 'open skirt', 'tennis uniform', 'shark hood', 'boxing gloves', 'plaid bowtie', 'glowing sword', 'holding stick', 'panty straps', 'white-framed eyewear', 'briefs', 'multicolored bodysuit', 'team rocket', 'turnaround', 'black wristband', 'breast bondage', 'd-pad hair ornament', 'champion\'s tunic \(zelda\)', 'sweatband', 'latex bodysuit', 'yellow choker', 'skull mask', 'grey kimono', 'vertical-striped pantyhose', 'little busters! school uniform', 'blood stain', 'school hat', 'gown', 'sweater around waist', 'red armor', 'sleeping on person', 'lace bra', 'tactical clothes', 'impossible dress', 'hikarizaka private high school uniform', 'hands on headwear', 'u u', 'spiked armlet', 'holding whip', 'millennium cheerleader outfit \(blue archive\)', 'see-through leotard', 'green neckerchief', 'o-ring thigh strap', 'frilled socks', 'swim briefs', 'leotard pull', 'grey necktie', 'breast curtain', 'holstered', 'puffy shorts', 'cherry blossom print', 'helm', 'blue sash', 'two-sided jacket', 'crotch plate', 'plaid necktie', 'kappougi', 'multicolored legwear', 'orange scarf', 'backless leotard', 'multicolored gloves', 'holding swim ring', 'multiple condoms', 'loose clothes', 'horned helmet', 'mechanical tail', 'mini-hakkero', 'stomach cutout', 'naked sheet', 'skull print', 'seigaiha', 'chaps', 'bird on hand', 'poke ball print', 'black male underwear', 'happy tears', 'emoji', 'nijigasaki academy school uniform', 'clothed female nude female', 'tape gag', 'st\. gloriana\'s military uniform', 'constellation print', 'leg belt', 'see-through skirt', 'sports bikini', 'tracen training uniform', 'star necklace', 'fine fabric emphasis', 'open pants', 'fishnet top', 'drop earrings', 'multicolored bikini', 'barcode tattoo', 'sleeve garter', 'heart o-ring', 'pink vest', 'two-tone ribbon', 'bear hair ornament', 'chest strap', 'bowl hat', 'tight shirt', 'brown necktie', 'pencil dress', 'gold hairband', 'yellow hoodie', 'condom packet strip', 'white bag', 'red-tinted eyewear', 'animal ear headwear', 'collared coat', 'volleyball uniform', 'budget sarashi', 'anzio school uniform', 'collared cape', 'grey scarf', 'yellow thighhighs', 'cloud print', 'green sleeves', 'yellow bra', 'fish hair ornament', 'poncho', 'dixie cup hat', 'tankini', 'purple gemstone', 'bondage outfit', 'scar on neck', 'lip piercing', 'checkered kimono', 'clover print', 'ushanka', 'lycoris uniform', 'holding game controller', 'untucked shirt', 'pink socks', 'black buruma', 'mars symbol', 'winged helmet', 'skull earrings', 'side-tie leotard', 'party hat', 'green apron', 'gusset', 'gold necklace', 'mouth veil', 'polka dot dress', 'puffy pants', 'plaid headwear', 'space helmet', 'brown hairband', 'sidepec', 'strawberry print', 'leather belt', 'butler', 'pokemon on head', 'claw ring', 'super robot', 'frilled cuffs', 'two-tone bowtie', 'baseball uniform', 'single gauntlet', 'taut dress', 'holding brush', 'black halo', 'checkered necktie', 'three-dimensional maneuver gear', 'tangzhuang', 'cropped vest', 'utility belt', 'white serafuku', 'fur-trimmed cloak', 'straitjacket', 'blue headband', 'gold bracelet', 'pink coat', 'black undershirt', 'stiletto heels', 'poke ball symbol', 'cum on legs', 'polka dot bra', 'holding baseball bat', 'naked cape', 'orange thighhighs', 'thigh cutout', 'ankle lace-up', 'open bra', 'ribbon-trimmed clothes', 'polo shirt', 'blue bag', 'purple belt', 'strap-on', 'red bandana', 'blue collar', 'gorget', 'white veil', 'belt bra', 'yellow armband', 'holding leaf', 'flower ornament', 'german clothes', 'fur-trimmed skirt', 'shoulder boards', 'flame print', 'cupless bra', 'holding shoes', 'hooded sweater', 'arm wrap', 'multicolored shirt', 'pillbox hat', 'brown socks', 'single fingerless glove', 'plaid pants', 'holding helmet', 'claw \(weapon\)', 'yellow belt', 'pink sailor collar', 'homurahara academy school uniform', 'red hood', 'grey sleeves', 'pocky kiss', 'unworn bikini top', 'striped gloves', 'hair ears', 'bow earrings', 'fur-trimmed kimono', 'cropped hoodie', 'bandaid on hand', 'biker clothes', 'sticker on face', 'pink pajamas', 'green-framed eyewear', 'bandaged neck', 'pacifier', 'striped kimono', 'crescent facial mark', 'x', 'blue cloak', 'stitched face', 'sweatpants', 'shoulder strap', 'eyewear on headwear', 'cowboy western', 'pink collar', 'respirator', 'unworn boots', 'ribbon bondage', 'male playboy bunny', 'thigh belt', 'shoelaces', 'kibito high school uniform', 'purple capelet', 'yellow bag', 'bodice', 'pink eyeshadow', 'holding pillow', 'dress tug', 'pink belt', 'reindeer costume', 'ribbon-trimmed collar', 'hakama pants', 'snap-fit buckle', 'chef', 'pink one-piece swimsuit', 'gold armor', 'magatama earrings', 'holding balloon', 'brown-framed eyewear', 'blue-tinted eyewear', 'moon \(ornament\)', 'buttoned cuffs', 'cow boy', 'micro panties', 'viewer holding leash', 'wiping tears', 'priest', 'purple eyeshadow', 'yellow sash', 'sword over shoulder', 'holding scissors', 'brown cloak', 'multicolored bow', 'romper', 'diamond cutout', 'kuromorimine school uniform', 'single strap', 'shinsengumi', 'single pantsleg', 'bird on shoulder', 'yasogami school uniform', 'gold bikini', 'grey belt', 'black garter straps', 'undone necktie', 'orange sailor collar', 'ankle strap', 'holding needle', 'triangle earrings', 'bow choker', 'striped shorts', 'platform heels', 'delinquent', 'ribbed sleeves', 'animal hug', 'dress flower', 'embellished costume', 'thighhighs pull', 'hooded robe', 'purple-tinted eyewear', 'venus symbol', 'yellow pants', 'heart button', 'sunflower hair ornament', 'hawaiian shirt', 'plate armor', 'bruise on face', 'sleeveless turtleneck leotard', '39', 'plaid jacket', 'lace-trimmed sleeves', 'orange neckerchief', 'pointless condom', 'drinking straw in mouth', 'diving suit', 'dirndl', 'sakuramon', 'holding water gun', 'two-tone headwear', 'brown bowtie', 'ribbon in mouth', 'frilled shorts', 'green bodysuit', 'tricorne', 'handkerchief', 'spiked club', 'cloth gag', 'harem pants', 'naked kimono', 'vibrator under panties', 'leather gloves', 'sleeveless hoodie', 'naked hoodie', 'multicolored coat', 'tribal', 'colored shoe soles', 'bow legwear', 'sparkler', 'mustache stubble', 'greco-roman clothes', 'butterfly on hand', 'turtleneck leotard', 'gradient clothes', 'sleep mask', 'hakurei reimu \(cosplay\)', 'ass cutout', 'latex gloves', 'bath yukata', 'year of the dragon', 'santa boots', 'bear print', 'gold choker', 'open robe', 'drawing bow', 'icho private high school uniform', 'ginkgo leaf', 'scar on stomach', 'loose bowtie', 'grey bikini', 'unworn sandals', 'yellow coat', 'white armor', 'forked tongue', 'eyewear strap', 'print bra', 'pentacle', 'shimaidon \(sex\)', 'blue armor', 'pink pantyhose', 'kigurumi', 'happi', 'duffel coat', 'pants rolled up', 'unworn gloves', 'short jumpsuit', 'grey ribbon', 'volleyball \(object\)', 'deerstalker', 'red apron', 'star facial mark', 'broken chain', 'grey sports bra', 'orange pants', 'tulip hat', 'untying', 'orange pantyhose', 'ajirogasa', 'wrist guards', 'grey bra', 'ballerina', 'full-length zipper', 'novel cover', 'cross print', 'masturbation through clothes', 'black garter belt', 'purple one-piece swimsuit', 'green capelet', 'holding fishing rod', 'two-tone footwear', 'overcoat', 'dark penis', 'key necklace', 'winged footwear', 'brown apron', 'high kick', 'pink-tinted eyewear', 'holding cane', 'crescent print', 'mask around neck', 'brown hoodie', 'print jacket', 'jaguar ears', 'lace-trimmed skirt', 'open belt', 'fishnet gloves', 'naked bandage', 'back-seamed legwear', 'cocktail dress', 'two-tone bodysuit', 'brown bikini', 'torn jeans', 'holding vegetable', 'purple hoodie', 'sunflower field', 'animal ear legwear', 'holding hose', 'new school swimsuit', 'sphere earrings', 'hamaya', 'low neckline', 'yellow apron', 'green bag', 'hatsune miku \(cosplay\)', 'ribbed bodysuit', 'impossible swimsuit', 'cum on self', 'triangle print', 'sunscreen', 'boxer briefs', 'striped sweater', 'candy hair ornament', 'kesa', 'gradient legwear', 'holding jacket', 'mismatched sleeves', 'scooter', 'kimono skirt', 'orange ascot', 'tooth necklace', 'purple neckerchief', 'double fox shadow puppet', 'aqua panties', 'sideless shirt', 'leather boots', 'goatee stubble', 'hand tattoo', 'ballet slippers', 'camouflage jacket', 'kimono pull', 'combat helmet', 'grey neckerchief', 'tapir tail', 'single horizontal stripe', 'white bird', 'glomp', 'diving mask on head', 'gradient dress', 'pointy footwear', 'blood on knife', 'torn scarf', 'kouhaku nawa', 'spiked choker', 'sword on back', 'kiyosumi school uniform', 'holding stylus', 'arrow through heart', 'scar on leg', 'sobu high school uniform', 'onmyouji', 'huge bow', 'nippleless clothes', 'aqua jacket', 'circle skirt', 'sleeve bow', 'no gloves', 'pearl bracelet', 'orange hoodie', 'hooded cardigan', 'pink capelet', 'yellow bodysuit', 'two-tone sports bra', 'combat boots', 'rabbit-shaped pupils', 'yin yang orb', 'dice hair ornament', 'fish print', 'polka dot swimsuit', 'ninja mask', 'overall shorts', 'holding ladle', 'sweaty clothes', 'shorts under dress', 'fur-trimmed footwear', 'shiny legwear', 'drum set', 'eden academy school uniform', 'eyewear hang', 'star brooch', 'kirin \(armor\)', 'expressive clothes', 'burnt clothes', 'ribbon-trimmed dress', 'multicolored headwear', 'duster', 'necktie grab', 'wetsuit', 'cross tie', 'belt boots', 'sharp toenails', 'camouflage pants', 'ribbed leotard', 'torn leotard', 'pinching sleeves', 'strawberry hair ornament', 'food-themed earrings', 'white umbrella', 'holding ice cream', 'torn panties', 'green socks', 'clothes', 'plaid panties', 'mole above mouth', 'riding pokemon', 'athletic leotard', 'headlamp', 'sword behind back', 'grey bodysuit', 'fur-trimmed shorts', 'frilled leotard', 'jingasa', 'brown corset', 'bird mask', 'orange panties', 'hat tip', 'tarot \(medium\)', 'denim jacket', 'two-tone hairband', 'wig', 'square 4koma', 'brown panties', 'holding gohei', 'anchor print', 'white snake', 'polka dot headwear', 'white garter straps', 'frilled ascot', 'colored shadow', 'yellow sleeves', 'age regression', 'shark costume', 'cutout above navel', 'purple scrunchie', 'torn gloves', 'two-tone legwear', 'motorcycle helmet', 'high-waist pantyhose', 'mummy costume', 'orange sweater', 'mahjong tile', 'unitard', 'torn jacket', 'bikesuit', 'upshorts', 'papakha', 'lace-trimmed gloves', 'silver trim', 'scarf over mouth', 'lace choker', 'collared vest', 'tented shirt', 'ghost costume', 'animal on lap', 'ballet', 'penis peek', 'crystal earrings', 'double w', 'bicorne', 'holding saucer', 'multicolored footwear', 'kourindou tengu costume', 'red border', 'pink border', 'detective', 'multicolored kimono', 'drawing sword', 'vampire costume', 'shell bikini', 'brown leotard', 'pink ascot', 'breast cutout', 'two-tone leotard', 'holding violin', 'stole', 'cetacean tail', 'holding envelope', 'sparkle print', 'yellow leotard', 'frog print', 'yellow butterfly', 'pink camisole', 'panties on head', 'lapel pin', 'loungewear', 'nearly naked apron', 'long tail', 'green hakama', 'santa gloves', 'kodona', 'pearl earrings', 'blue border', 'boobplate', 'heart collar', 'training bra', 'arm armor', 'purple socks', 'white mask', 'fourth east high school uniform', 'polka dot legwear', 'uchikake', 'surrounded by penises', 'print headwear', 'pouring onto self', 'egg hair ornament', 'kamiyama high school uniform \(hyouka\)', 'baggy clothes', 'kine', 'yellow cape', 'native american', 'hanten \(clothes\)', 'buruma pull', 'holding bucket', 'adjusting legwear', 'lace gloves', 'side drill', 'sideburns stubble', 'tube dress', 'blue sports bra', 'strap lift', 'scar on mouth', 'nejiri hachimaki', 'gathers', 'covering one eye', 'rook \(chess\)', 'glowing butterfly', 'thighhighs over pantyhose', 'wakizashi', 'swim cap', 'fur cape', 'grey capelet', 'stained panties', 'aviator sunglasses', 'pill earrings', 'blue robe', 'prison clothes', 'aqua footwear', 'drying hair', 'unzipping', 'pinstripe shirt', 'hat over one eye', 'full beard', 'bishop \(chess\)', 'bandaid hair ornament', 'huge moon', 'hanbok', 'loose shirt', 'year of the rat', 'footwear ribbon', 'tearing clothes', 'white butterfly', 'grey hairband', 'ornate ring', 'coin hair ornament', 'holding tablet pc', 'bucket hat', 'gold footwear', 'tutu', 'holding popsicle', 'between pectorals', 'orange vest', 'alpaca ears', 'holding ribbon', 'floating scarf', 'mole on cheek', 'crotch cutout', 'single epaulette', 'heart facial mark', 'cropped sweater', 'messenger bag', 'weasel ears', 'cowboy boots', 'wrist bow', 'upshirt', 'in cup', 'brown sleeves', 'clothes between breasts', 'swimsuit cover-up', 'double vertical stripe', 'covering ass', 'kissing hand', 'armpit cutout', 'white hood', 'brown choker', 'chin strap', 'gladiator sandals', 'mole on stomach', 'single boot', 'red tank top', 'black umbrella', 'blue tunic', 'wrist wrap', 'single wrist cuff', 'kepi', 'white headdress', 'wet dress', 'hooded track jacket', 'orange sleeves', 'brown collar', 'two-tone cape', 'hooded bodysuit', 'red mask', 'body armor', 'red mittens', 'torn swimsuit', 'purple sash', 'satin', 'alice \(alice in wonderland\) \(cosplay\)', 'cat ear legwear', 'saiyan armor', 'white mittens', 'grey cape', 'frilled sailor collar', 'side slit shorts', 'pants tucked in', 'condom belt', 'cross choker', 'black sweater vest', 'rider belt', 'multicolored cape', 'girthy penis', 'yellow socks', 'fold-over boots', 'pink hakama', 'naked overalls', 'spit take', 'leg wrap', 'mochi trail', 'sleeve ribbon', 'blood on arm', 'tied jacket', 'cum in nose', 'blue tank top', 'two-sided dress', 'holding beachball', 'clothes between thighs', 'purple-framed eyewear', 'jockstrap', 'lowleg pants', 'flying kick', 'tight dress', 'no jacket', 'holding jewelry', 'frilled camisole', 'unworn coat', 'see-through jacket', 'pink cape', 'sideways hat', 'holding megaphone', 'string bra', 'huge testicles', 'unworn dress', 'holding letter', 'coke-bottle glasses', 'open bodysuit', 'holding behind back', 'holding chocolate', 'studded bracelet', 'aqua gloves', 'star pasties', 'shuka high school uniform', 'multicolored scarf', 'test plugsuit', 'levitation', 'houndstooth', 'head chain', 'yellow tank top', 'polka dot skirt', 'radiation symbol', 'chalice', 'adidas', 'bandaid on forehead', 'vertical-striped jacket', 'leather pants', 'hard hat', 'cardigan around waist', 'vertical-striped bikini', 'torn bodystocking', 'shoulder cannon', 'purple ascot', 'breast padding', 'white tiger', 'arachne', 'cross pasties', 'holding money', 'two-tone hoodie', 'kimono lift', 'nipple clamps', 'latex legwear', 'grey tank top', 'back-print panties', 'barefoot sandals \(jewelry\)', 'green pantyhose', 'heart maebari', 'male maid', 'arm cuffs', 'floral print kimono', 'fake nails', 'ribbon-trimmed headwear', 'clown', 'joestar birthmark', 'taimanin suit', 'jaguar print', 'adjusting necktie', 'lightsaber', 'jeweled branch of hourai', 'multi-strapped panties', 'medallion', 'holding notebook', 'pool of blood', 'yellow raincoat', 'flower over eye', 'cardigan vest', 'bridal legwear', 'yellow-tinted eyewear', 'striped hoodie', 'naked scarf', 'dudou', 'green tunic', 'otter ears', 'purple hakama', 'green tank top', 'hand under shirt', 'skirt basket', 'white romper', 'sitting backwards', 'youtou high school uniform', 'vietnamese dress', 'lace-trimmed hairband', 'hoodie lift', 'bear costume', 'green belt', 'crotchless pantyhose', 'wringing clothes', 'holding branch', 'shorts around one leg', 'aqua neckerchief', 'holding remote control', 'nose ring', 'jacket pull', 'polka dot shirt', 'underbutt', 'holding skull', 'four-leaf clover hair ornament', 'potara earrings', 'grey leotard', 'print necktie', 'parka', 'shell necklace', 'holding sex toy', 'blue bandana', 'apron lift', 'long beard', 'orange belt', 'animal slippers', 'camouflage headwear', 'penguin hood', 'crocs', 'jacket over swimsuit', 'rope belt', 'polar bear', 'shoulder sash', 'fur boots', 'checkered sash', 'yellow sweater vest', 'purple cardigan', 'anchor necklace', 'striped hairband', 'brown bra', 'sailor senshi', 'bike shorts under shorts', 'hat with ears', 'puff and slash sleeves', 'stitched mouth', 'half mask', 'print sleeves', 'green scrunchie', 'thick mustache', 'argyle sweater', 'hospital gown', 'onesie', 'green armband', 'polka dot scrunchie', 'double \\m/', 'two-tone coat', 'cherry hair ornament', 'sukajan', 'platform boots', 'floating weapon', 'wa lolita', 'striped one-piece swimsuit', 'cat costume', 'jaguar tail', 'rabbit hat', 'thick beard', 'yellow border', 'martial arts belt', 'bib', 'fur-trimmed collar', 'sports bra lift', 'surcoat', 'single thigh boot', 'strawberry panties', 'high tops', 'sitting on table', 'plaid bra', 'sleeve grab', 'panty lift', 'blood bag', 'ankle wrap', 'male underwear pull', 'print hoodie', 'green-tinted eyewear', 'dress swimsuit', 'flower brooch', 'cum in container', 'cross-laced legwear', 'popped button', 'blue shawl', 'butterfly brooch', 'white sarong', 'green one-piece swimsuit', 'grey serafuku', 'lace-trimmed thighhighs', 'orange cape', 'american flag print', 'skirt flip', 'ehoumaki', 'chain headband', 'holding frying pan', 'orange leotard', 'sling bikini top', 'adapted uniform', 'kabuto \(helmet\)', 'planet hair ornament', 'hair color connection', 'patchwork clothes', 'hat on back', 'watermelon slice', 'holding teapot', 'pants under skirt', 'unworn bikini bottom', 'popsicle in mouth', 'milky way', 'multicolored hairband', 'drop-shaped pupils', 'skull necklace', 'purple serafuku', 'mitre', 'frilled jacket', 'penis on ass', 'aqua bra', 'blue pajamas', 'anchor choker', 'polka dot ribbon', 'halter shirt', 'red sports bra', 'nudist', 'naked tabard', 'sideless kimono', 'single knee pad', 'long shirt', 'multiple scars', 'penis in panties', 'cross-laced slit', 'card parody', 'orange socks', 'cream on face', 'sam browne belt', 'satin panties', 'embroidery', 'blue sarong', 'pink umbrella', 'buruma aside', 'genderswap \(otf\)', 'blue umbrella', 'legband', 'musical note print', 'holding wrench', 'unworn eyepatch', 'hooded dress', 'floating book', 'rabbit costume', 'skeleton print', 'wataboushi', 'st\. theresa\'s girls academy school uniform', 'pinstripe suit', 'bowler hat', 'pegasus knight uniform \(fire emblem\)', 'green eyeshadow', 'pumpkin hair ornament', 'bandaged wrist', 'holding swimsuit', 'spiked hairband', 'coat dress', 'jester', 'stopwatch', 'shoulder belt', 'holding footwear', 'holding toy', 'panties under buruma', 'food art', 'hugging book', 'brown border', 'half-skirt', 'orange jumpsuit', 'midriff sarashi', 'red track suit', 'grey suit', 'hooded vest', 'scylla', 'bathrobe', 'coif', 'bikini shorts', 'bow skirt', 'side-tie peek', 'tweaking own nipple', 'bralines', 'blue camisole', 'striped coat', 'pelt', 'unfastened', 'greek toe', 'black armband', 'adjusting panties', 'vertical-striped socks', 'plaid ribbon', 'vertical-striped panties', 'print sarong', 'cloth', 'holding test tube', 'band uniform', 'checkered shirt', 'lowleg skirt', 'fur-trimmed shirt', 'german flag bikini', 'lightning bolt print', 'holding mop', 'blue tabard', 'holly hair ornament', 'exercise ball', 'lillian girls\' academy school uniform', 'covering one breast', 'vertical-striped pants', 'blood on leg', 'stained clothes', 'high-low skirt', 'christmas stocking', 'tengu mask', 'pumpkin hat', 'hand wraps', 'belt skirt', 'silver dress', 'lace-trimmed choker', 'brown mittens', 'shiny and normal', 'blue hood', 'naked cloak', 'one-piece thong', 'black bandeau', 'orange goggles', 'fishnet socks', 'purple collar', 'flower choker', 'elbow sleeve', 'holding heart', 'pocky in mouth', 'grey apron', 'jiangshi costume', 'mizu happi', 'rubber gloves', 'red cardigan', 'holding coin', 'mole under each eye', 'clothes theft', 'simulated fellatio', 'holding microphone stand', 'clock eyes', 'holding chain', 'wrong foot', 'converse', 'thong aside', 'walking on liquid', 'knight \(chess\)', 'pelvic curtain lift', 'mutual hug', 'brown neckerchief', 'kerchief', 'red suit', 'red robe', 'strapless bottom', 'wing brooch', 'diagonal-striped bowtie', 'holding drumsticks', 'aqua kimono', 'vertical-striped kimono', 'stitched arm', 'pink sash', 'cuff links', 'checkered dress', 'ornate border', 'animal ear hairband', 'grey bowtie', 'clothed male nude male', 'toe cleavage', 'yellow camisole', 'crotch zipper', 'shirt overhang', 'animal on hand', 'holding shirt', 'unworn shorts', 'riding bicycle', 'star-shaped eyewear', 'orange headband', 'scouter', 'long toenails', 'holding cake', 'cargo pants', 'frilled umbrella', 'glitter', 'holding suitcase', 'green headband', 'micro bra', 'motosu school uniform', 'brown serafuku', 'single head wing', 'year of the dog', 'covered clitoris', 'panda hood', 'taut swimsuit', 'purple butterfly', 'aqua leotard', 'little red riding hood \(grimm\) \(cosplay\)', 'year of the pig', 'fur cuffs', 'glowing hand', 'panties under shorts', 'maple leaf print', 'exploding clothes', 'right-over-left kimono', 'holding creature', 'stiletto \(weapon\)', 'sock pull', 'clawed gauntlets', 'print mug', 'camisole lift', 'frilled headwear', 'cable tail', 'red male underwear', 'exposed pocket', 'two-sided coat', 'safety glasses', 'holding fish', 'front slit', 'flippers', 'kariyushi shirt', 'knives between fingers', 'broken sword', 'policeman', 'spade hair ornament', 'male underwear peek', 'leotard peek', 'neck garter', 'weasel tail', 'blue suit', 'holding photo', 'dissolving clothes', 'holding pole', 'jacket lift', 'holding shovel', 'backless swimsuit', 'tickling armpits', 'low-cut armhole', 'propeller hair ornament', 'fake magazine cover', 'holding cross', 'otter tail', 'taut leotard', 'o-ring swimsuit', 'wind turbine', 'pom pom earrings', 'checkered bow', 'multiple hairpins', 'studded choker', 'red bandeau', 'single garter strap', 'fruit hat ornament', 'ski goggles', 'holding briefcase', 'brown sash', 'layered kimono', 'o-ring belt', 'striped vest', 'green cardigan', 'multicolored stripes', 'aqua hairband', 'plate carrier', 'bear hood', 'holding bra', 'detached leggings', 'paw print pattern', 'body switch', 'multicolored tail', 'walker \(robot\)', 'down jacket', 'rabbit on head', 'giant male', 'holding scroll', 'pink tank top', 'yellow one-piece swimsuit', 'white bandeau', 'black tube top', 'scoop neck', 'female goblin', 'temari ball', 'red wine', 'yellow pantyhose', 'bandaged fingers', 'ahoge wag', 'black hood', 'black veil', 'head on head', 'leaf background', 'hakama shorts', 'moose ears', 'fishnet bodysuit', 'pointy hat', 'fur jacket', 'bandaid on neck', 'holding surfboard', 'bridal lingerie', 'hat belt', 'overall skirt', 'sweater pull', 'holding map', 'disguise', 'knife sheath', 'rotary phone', 'pantyhose under swimsuit', 'pawn \(chess\)', 'unworn goggles', 'sky lantern', 'frontless outfit', 'armored leotard', 'shoulder plates', 'ribbed thighhighs', 'forked tail', 'lightning bolt hair ornament', 'undone neck ribbon', 'shoulder guard', 'lop rabbit ears', 'cassock', 'metamoran vest', 'normal suit', 'checkered legwear', 'see-through swimsuit', 'holding necklace', 'panties over pantyhose', 'orange bra', 'adjusting scarf', 'layered shirt', 'bird on arm', 'paint on clothes', 'scar on hand', 'blue outline', 'unworn bikini', 'pink sports bra', 'tape on nipples', 'adjusting buruma', 'side-tie shirt', 'torn coat', 'rash guard', 'poke ball \(legends\)', 'ankle bow', 'covering own ears', 'mtu virus', 'bandaid on head', 'fur-trimmed bikini', 'hat tassel', 'argyle cutout', 'cross-laced skirt', 'fruit on head', 'suspenders slip', 'cow costume', 'multicolored leotard', 'white garter belt', 'holding toothbrush', 'toga', 'holding lipstick tube', 'multi-strapped bikini top', 'white wristband', 'purple robe', 'turtleneck jacket', 'rice hat', 'shared earphones', 'mole on arm', 'holding mirror', 'corsage', 'black outline', 'anchor earrings', 'wrapped candy', 'gingham', 'sweet lolita', 'side-tie skirt', 'print scarf', 'green collar', 'sweater tucked in', 'front-print panties', 'square neckline', 'bear panties', 'mini witch hat', 'holding key', 'holding torch', 'holding plectrum', 'white tube top', 'unworn hair ornament', 'holding magnifying glass', 'single off shoulder', 'torn cloak', 'heart hair', 'shirt around waist', 'sailor swimsuit \(idolmaster\)', 'detached ahoge', 'ankle garter', 'year of the rooster', 'singlet', 'sailor collar lift', 'aviator cap', 'aqua shorts', 'holding newspaper', 'female service cap', 'ankleband', 'black babydoll', 'multiple bracelets', 'front zipper swimsuit', 'kin-iro mosaic high school uniform', 'holding bell', 'blue male underwear', 'side cape', 'glove bow', 'green serafuku', 'claw foot bathtub', 'ribbed socks', 'dress shoes', 'vertical-striped shorts', 'blue sweater vest', 'fur-trimmed thighhighs', 'streetwear', 'vertical stripes', 'labcoat', 'argyle', 'print legwear', 'tight', 'legwear under shorts', 'skirt removed', 'panties removed', 'multi-strapped bikini', 'diagonal stripes', 'clothes removed', 'bikini lift', 'gothic', 'frilled swimsuit', 'bunny print', 'qing guanmao', 'matching outfit', 'borrowed garments', 'beltbra', 'bikini top removed', 'nike', 'traditional clothes', 'boots removed', 'power suit \(metroid\)', 'sandals removed', 'clog sandals', 'multiple straps', 'socks removed', 'catholic', 'barefoot sandals', 'dress removed', 'strapless swimsuit', 'sling', 'bunny hat', 'beltskirt', 'greek clothes', 'military helmet', 'hardhat', 'bikini bottom removed', 'yamakasa', 'necktie removed'
}

clip_labels = [
    'hdr', 'accent light', 'adorable', 'adventurous', 'aggressive', 'airport', 'alert', 'alley', 'alternate reality', 'ambient', 'amusement park', 'ancient', 'angry', 'animated', 'anxious', 'aquarium', 'arctic', 'arena', 'arrogant', 'artificial light', 'artistic', 'astral', 'attractive', 'augmented reality', 'avant-garde', 'awesome', 'awful', 'back light', 'back lit', 'bar', 'baroque', 'battlefield', 'bay', 'beach', 'beautiful', 'belligerent', 'best', 'bio-punk', 'bizarre', 'black tone', 'blue tone', 'blurry', 'bohemian', 'bokeh', 'boring', 'bounced light', 'bridge', 'bright', 'broken', 'bus station', 'busy', 'cafe', 'calm', 'campground', 'candlelit', 'canyon', 'castle', 'cave', 'celestial', 'cemetery', 'charming', 'cheerful', 'church', 'cityscape', 'classic', 'clean', 'clever', 'cliff', 'clinic', 'clumsy', 'coastal', 'cold', 'color graded', 'colorful', 'colorful light', 'comfortable', 'confused', 'construction site', 'contrasty', 'cool light', 'coral reef', 'cosmic', 'cosmic horror', 'countryside', 'creepy', 'crowded', 'cruel', 'curious', 'cute', 'cyberpunk', 'damaged', 'dangerous', 'dark fantasy', 'dark tone', 'daylight', 'deep', 'delicate', 'delicious', 'depressed', 'desert', 'determined', 'diesel-punk', 'different', 'diffused light', 'dim', 'dirty', 'disgusting', 'dock', 'dramatic', 'dramatic light', 'dreamlike', 'dreamy', 'dry', 'dull', 'dusty', 'dystopian', 'edgy', 'eerie', 'elegant', 'emotional', 'empty', 'enchanting', 'enthusiastic', 'epic fantasy', 'ethereal', 'ethnic', 'excellent', 'exciting', 'expensive', 'extraterrestrial', 'fabulist', 'fabulous', 'fairground', 'fancy', 'fantastic', 'fantastical', 'fantasy', 'farm', 'fast', 'fearful', 'fearless', 'field', 'fill light', 'filthy', 'floodlight', 'fluorescent', 'foolish', 'forest', 'forgetful', 'fortress', 'friendly', 'funny', 'futuristic', 'galactic', 'garden', 'gentle', 'geometric', 'ghostly', 'glacier', 'glamorous', 'gleaming', 'glimmering', 'glorious', 'glossy', 'glowing', 'golden hour', 'gorgeous', 'graceful', 'greedy', 'green tone', 'grouchy', 'grumpy', 'hallucinatory', 'handsome', 'happy', 'harbor', 'hard light', 'harsh', 'haunting', 'healthy', 'heavy', 'high resolution', 'high-key', 'highway', 'hilarious', 'historical', 'horrible', 'hospital', 'hot', 'hotel', 'huge', 'humorous', 'hungry', 'hypnotic', 'incandescent', 'industrial', 'innocent', 'inquisitive', 'intelligent', 'intense', 'interesting', 'intergalactic', 'introspective', 'irresistible', 'island', 'jolly', 'joyful', 'juicy', 'jumpy', 'jungle', 'key light', 'kind', 'kooky', 'lake', 'landscape', 'legendary', 'lens flare', 'library', 'low resolution', 'low-key', 'luminescent', 'luminous', 'macabre', 'magical', 'magical realism', 'magnificent', 'majestic', 'marketplace', 'matte', 'meadow', 'melancholic', 'merry', 'metaphysical', 'minimalist', 'modern', 'monochromatic light', 'monument', 'moody', 'moonlit', 'mosque', 'mountainous', 'museum', 'muted', 'mysterious', 'mystical', 'mythical', 'mythopoeic', 'nano-punk', 'nasty', 'natural', 'natural light', 'naturalistic', 'neat', 'neon', 'nervous', 'nice', 'nightclub', 'noisy', 'nostalgic', 'numerous', 'oasis', 'obedient', 'obnoxious', 'old', 'optimistic', 'opulent', 'orchard', 'organic', 'otherworldly', 'outstanding', 'over exposed', 'pagoda', 'palace', 'parallel universe', 'park', 'pastel', 'peaceful', 'peninsula', 'phantasmagoric', 'pier', 'pink tone', 'pixelated', 'plaza', 'polished', 'pond', 'popular', 'post-apocalyptic', 'precious', 'pretty', 'psychedelic', 'quick', 'quiet', 'radiant', 'rainforest', 'rare', 'red tone', 'reflective', 'remarkable', 'resort', 'restaurant', 'retro', 'retro-futuristic', 'rich', 'rim light', 'riverbank', 'romanesque', 'romantic', 'royal', 'rude', 'ruins', 'rural', 'rustic', 'saturated', 'savannah', 'scary', 'school', 'science fiction', 'seascape', 'secretive', 'selfish', 'sepia tone', 'serene', 'serious', 'shadowy', 'sharp', 'sharp focus', 'shiny', 'shocking', 'short', 'shy', 'side light', 'silhouetted', 'silly', 'sincere', 'skyscraper', 'slim', 'slow', 'soft', 'soft focus', 'soft light', 'somber', 'sparkling', 'speculative', 'spicy', 'splendid', 'spooky', 'spotlight', 'square', 'stadium', 'steampunk', 'streetscape', 'strong', 'suburban', 'successful', 'sunlit', 'supernatural', 'surreal', 'surrealistic', 'swamp', 'sweet', 'synagogue', 'talented', 'tall', 'techno-thriller', 'temple', 'tense', 'terrible', 'terrific', 'theater', 'thick', 'thin', 'time travel', 'tired', 'top light', 'train station', 'tranquil', 'transcendent', 'transdimensional', 'tropical', 'tunnel', 'twilight', 'ugly', 'under exposed', 'under light', 'underwater', 'unearthly', 'unique', 'university', 'unnerving', 'urban', 'urban fantasy', 'utopian', 'valley', 'vibrant', 'vineyard', 'vintage', 'virtual reality', 'visionary', 'visionary fiction', 'vivid', 'volcano', 'warm', 'waterfall', 'weak', 'wealthy', 'weird fiction', 'well lit', 'whimsical', 'wild', 'wilderness', 'wise', 'wonderful', 'worried', 'young', 'youthful', 'zany', 'zealous', 'zestful', 'zippy', 'zoo'
]

people_tags = [
    r'^1girl$', r'^1boy$', r'^69$', r'^absolutely_everyone$', r'^after_kiss$', r'^age_comparison$', r'^age_difference$', r'^age_progression$', r'^angel_and_devil$', r'^anilingus$', r'^ankle_grab$', r'^anti-aircraft$', r'^armpit_sex$', r'^arms_around_neck$', r'^arms_around_waist$', r'^arm_around_back$', r'^arm_around_neck$', r'^arm_around_shoulder$', r'^arm_around_waist$', r'^arm_held_back$', r'^arm_hug$', r'^ass-to-ass$', r'^asymmetrical_docking$', r'^back-to-back$', r'^band$', r'^behind_another$', r'^black_vs_white$', r'^bound_together$', r'^boy_on_top$', r'^boy_sandwich$', r'^breastfeeding$', r'^breasts_on_head$', r'^breast_envy$', r'^grabbing_another\'s_breast$', r'^breast_smother$', r'^breast_sucking$', r'^buttjob$', r'^caressing_testicles$', r'^carrying_person$', r'^chart$', r'^chasing$', r'^cheating_\(relationship\)$', r'^cheek-to-cheek$', r'^chikan$', r'^child_carry$', r'^child_on_child$', r'^circle_formation$', r'^clog_sandals$', r'^clone$', r'^clothed_female_nude_female$', r'^clothed_female_nude_male$', r'^clothed_male_nude_female$', r'^clothed_sex$', r'^coffee_cup$', r'^collage$', r'^colored_text$', r'^column_lineup$', r'^comforting$', r'^cooperative_fellatio$', r'^cooperative_paizuri$', r'^copyright$', r'^costume_switch$', r'^couple$', r'^cousins$', r'^covering_another\'s_eyes$', r'^covering_another\'s_mouth$', r'^covering_mouth$', r'^cowgirl_position$', r'^cross-section$', r'^cuddling$', r'^cum_in_nose$', r'^cum_overflow$', r'^cunnilingus$', r'^cute_$', r'^dark_penis$', r'^deepthroat$', r'^deep_penetration$', r'^disembodied_limb$', r'^disembodied_penis$', r'^doggystyle$', r'^double_handjob$', r'^dressing_another$', r'^dual_persona$', r'^duckling$', r'^duel$', r'^ear_biting$', r'^ejaculating_while_penetrated$', r'^ejaculation$', r'^emotionless_sex$', r'^everyone$', r'^evolutionary_line$', r'^expression_chart$', r'^eye_contact$', r'^face-to-face$', r'^facepalm$', r'^face_to_breasts$', r'^facing_another$', r'^fellatio$', r'^female_child$', r'^femdom$', r'^fff_threesome$', r'^ffm_threesome$', r'^fighting$', r'^finger_biting$', r'^finger_in_another\'s_mouth$', r'^finger_to_another\'s_mouth$', r'^flashback$', r'^flat_chest_grab$', r'^fleeing$', r'^footjob$', r'^foot_worship$', r'^forehead-to-forehead$', r'^french_kiss$', r'^friends$', r'^frilled_swimsuit$', r'^frottage$', r'^full_nelson$', r'^fume$', r'^furry_with_furry$', r'^furry_with_non-furry$', r'^futa_on_male$', r'^futa_with_female$', r'^futa_with_futa$', r'^futa_with_male$', r'^gangbang$', r'^girl_on_top$', r'^girl_sandwich$', r'^glansjob$', r'^glomp$', r'^gloved_handjob$', r'^grabbing$', r'^grabbing_another\'s_ass$', r'^grabbing_another\'s_breast$', r'^grabbing_another\'s_chin$', r'^grabbing_another\'s_hair$', r'^grabbing_from_behind$', r'^greek_clothes$', r'^griffin_$', r'^grinding$', r'^groom$', r'^groping$', r'^group_hug$', r'^group_picture$', r'^group_sex$', r'^guided_breast_grab$', r'^guided_penetration$', r'^guiding_hand$', r'^hairjob$', r'^handjob$', r'^handshake$', r'^hands_on_another\'s_cheeks$', r'^hands_on_another\'s_chest$', r'^hands_on_another\'s_face$', r'^hands_on_another\'s_head$', r'^hands_on_another\'s_hips$', r'^hands_on_another\'s_shoulders$', r'^hands_on_another\'s_thighs$', r'^hands_on_shoulders$', r'^hand_grab$', r'^hand_in_another\'s_hair$', r'^hand_on_another\'s_arm$', r'^hand_on_another\'s_ass$', r'^hand_on_another\'s_back$', r'^hand_on_another\'s_cheek$', r'^hand_on_another\'s_chest$', r'^hand_on_another\'s_chin$', r'^hand_on_another\'s_ear$', r'^hand_on_another\'s_face$', r'^hand_on_another\'s_hand$', r'^hand_on_another\'s_head$', r'^hand_on_another\'s_hip$', r'^hand_on_another\'s_leg$', r'^hand_on_another\'s_neck$', r'^hand_on_another\'s_shoulder$', r'^hand_on_another\'s_stomach$', r'^hand_on_another\'s_thigh$', r'^hand_on_another\'s_waist$', r'^happy_sex$', r'^harem$', r'^headpat$', r'^heads_together$', r'^head_between_breasts$', r'^head_grab$', r'^head_on_another\'s_shoulder$', r'^head_on_chest$', r'^heart_hands_duo$', r'^heckler_$', r'^height_difference$', r'^hetero$', r'^holding_another\'s_arm$', r'^holding_another\'s_foot$', r'^holding_another\'s_hair$', r'^holding_another\'s_leg$', r'^holding_another\'s_wrist$', r'^holding_hair$', r'^holding_hands$', r'^holding_pokemon$', r'^holomyth$', r'^hoop_piercing$', r'^horn_grab$', r'^hug$', r'^hug_from_behind$', r'^humping$', r'^imminent_fellatio$', r'^imminent_kiss$', r'^imminent_penetration$', r'^imminent_vaginal$', r'^implied_fingering$', r'^implied_futanari$', r'^implied_kiss$', r'^in-franchise_crossover$', r'^incest$', r'^infinity$', r'^instant_loss$', r'^internal_cumshot$', r'^interracial$', r'^interspecies$', r'^invisible_man$', r'^in_the_face$', r'^irrumatio$', r'^jealous$', r'^josou_seme$', r'^just_the_tip$', r'^kabedon$', r'^kanshou_$', r'^kiss$', r'^kissing_cheek$', r'^kissing_forehead$', r'^kissing_hand$', r'^kissing_neck$', r'^kissing_penis$', r'^lap_pillow$', r'^leaning_on_person$', r'^left-to-right_manga$', r'^legwear_under_shorts$', r'^leg_between_thighs$', r'^leg_grab$', r'^leg_lock$', r'^licking_another\'s_face$', r'^licking_armpit$', r'^licking_foot$', r'^licking_nipple$', r'^licking_penis$', r'^lifted_by_another$', r'^lifting_another\'s_clothes$', r'^lifting_person$', r'^light_blue_background$', r'^lineup$', r'^locked_arms$', r'^lolidom$', r'^looking_at_another$', r'^looking_at_penis$', r'^lying_on_lap$', r'^lying_on_person$', r'^massage$', r'^matching_outfits$', r'^matching_outfits$', r'^mating_press$', r'^missionary$', r'^misunderstanding$', r'^mixed-sex_bathing$', r'^mixed_bathing$', r'^mmf_threesome$', r'^mmm_threesome$', r'^mod3_\(girls\'_frontline\)$', r'^molestation$', r'^motherly$', r'^mouse$', r'^mtu_virus$', r'^multiple_4koma$', r'^multiple_boys$', r'^multiple_crossover$', r'^multiple_drawing_challenge$', r'^multiple_girls$', r'^multiple_others$', r'^multiple_penises$', r'^multiple_persona$', r'^multiple_riders$', r'^multiple_views$', r'^multitasking$', r'^mutual_hug$', r'^mutual_masturbation$', r'^netorare$', r'^nipple-to-nipple$', r'^noses_touching$', r'^nursing_handjob$', r'^odd_one_out$', r'^onee-loli$', r'^onee-shota$', r'^onii-shota$', r'^on_person$', r'^oral$', r'^orgy$', r'^out_of_frame$', r'^overflow$', r'^paizuri$', r'^paizuri_under_clothes$', r'^penises_touching$', r'^penis_awe$', r'^penis_grab$', r'^penis_on_ass$', r'^penis_on_face$', r'^penis_size_difference$', r'^people$', r'^perpendicular_paizuri$', r'^person_on_head$', r'^phone_screen$', r'^picture_\(object\)$', r'^piggyback$', r'^pikmin_\(creature\)$', r'^pointing_at_another$', r'^pokemon_on_head$', r'^pokemon_on_shoulder$', r'^pokephilia$', r'^pov_crotch$', r'^pov_hands$', r'^prank$', r'^princess_carry$', r'^print_legwear$', r'^prone_bone$', r'^protecting$', r'^pulled_by_another$', r'^pulling_another\'s_clothes$', r'^pushing$', r'^pushing_away$', r'^reach-around$', r'^remembering$', r'^reverse_cowgirl_position$', r'^reverse_suspended_congress$', r'^reverse_upright_straddle$', r'^rhodes_island_logo$', r'^riding_pokemon$', r'^rotational_symmetry$', r'^rough_sex$', r'^sailor_senshi$', r'^same-sex_bathing$', r'^sandwiched$', r'^see-through_swimsuit$', r'^selfcest$', r'^sequential$', r'^sex$', r'^sextuplets$', r'^sexual_coaching$', r'^sex_from_behind$', r'^shared_bathing$', r'^shared_clothes$', r'^shared_earphones$', r'^shared_food$', r'^shared_object_insertion$', r'^shared_scarf$', r'^shared_speech_bubble$', r'^shared_umbrella$', r'^shimaidon_\(sex\)$', r'^shiny_and_normal$', r'^shoulder_carry$', r'^siblings$', r'^side-by-side$', r'^sisters$', r'^sitting_on_bench$', r'^sitting_on_face$', r'^sitting_on_lap$', r'^sitting_on_person$', r'^sitting_on_shoulder$', r'^size_difference$', r'^slapping$', r'^sleeping_on_person$', r'^sleeve_grab$', r'^sling$', r'^solo_focus$', r'^spitroast$', r'^spitting$', r'^spit_take$', r'^spooning$', r'^square_4koma$', r'^squatting_cowgirl_position$', r'^standing_sex$', r'^starter_pokemon_trio$', r'^stealth_sex$', r'^still_life$', r'^straddling$', r'^straddling_paizuri$', r'^strangling$', r'^strap-on$', r'^surprise_kiss$', r'^surrounded_by_penises$', r'^suspended_congress$', r'^symmetrical_docking$', r'^tail_around_leg$', r'^tail_feathers$', r'^take_your_pick$', r'^teacher_and_student$', r'^teamwork$', r'^team_9$', r'^testicle_grab$', r'^testicle_sucking$', r'^thigh_grab$', r'^thigh_sex$', r'^threesome$', r'^time_paradox$', r'^torso_grab$', r'^tribadism$', r'^triplets$', r'^turnaround$', r'^twincest$', r'^twins$', r'^two-footed_footjob$', r'^two-handed_handjob$', r'^ugly_man$', r'^undressing_another$', r'^upright_straddle$', r'^uterus$', r'^vaginal$', r'^variations$', r'^walk-in$', r'^window_shade$', r'^wrestling$', r'^yaoi$', r'^yuri$', r'^:>=$'
    ] 

view_labels = [
    "portrait", "upper body", "lower body", "cowboy shot", "feet out of frame",
    "full body", "wide shot", "very wide shot", "close-up", "cut-in", "split crop",
    "profile", "from behind", "from side", "upside-down"
]
preson_labels = ['focus on one person', 'two persons', 'three persons', 'four persons', 'five persons', 'many persons', 'lots of people']
    
text_features_dict = {}
lebel_word = ""
clip_word = ""
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
        if tags_dict.get(tag, "") and tag in select_tags and 'CLOTHES' not in tag_categories_dict.get(tag, ""):
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

def get_aesthetic_tag(image):
    def aesthetic_tag(image):
        pixel_values = (
            aes_preprocessor(images=image, return_tensors="pt")
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
        if not args.not_char:
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
        return f"{'focus on ' if 'solo_focus' in features else ''}one person {chartag_from_folder}", ', '.join(chartags), boorutag, artisttag

    if len(chartags) > 3:
        chartags = []
    
    if not chartag_from_folder and features and ("solo" in features or "solo_focus" in features):
        return f"{'focus on ' if 'solo_focus' in features else ''}one person {' '.join(chartags)}" if chartags else "", ', '.join(chartags), boorutag, artisttag

    return f"{', '.join(chartags)}", ', '.join(chartags), boorutag, artisttag
    
def calculate_best_labels(image, short_caption, long_caption, image_path): 
    def contains_color(tag: str) -> bool:
        colors = {'red', 'orange', 'yellow', 'green', 'blue', 'aqua', 'purple', 'brown', 'pink', 'black', 'white', 'grey', 'dark ', 'light ', 'blonde'}
        return any(color in tag for color in colors)

    def cluster_labels(label_scores, is_solo, persontag, image=None, num_clusters=5):
        labels = [label for label, _ in label_scores]
        text_features = [text_features_dict[label].cpu().numpy().reshape(-1) for label in labels]

        text_features = np.array(text_features)
        text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)

        kmeans = faiss.Kmeans(text_features.shape[1], num_clusters, niter=20, verbose=True)
        kmeans.train(text_features)
        _, cluster_assignments = kmeans.index.search(text_features, 1)

        clusters = {i: [] for i in range(num_clusters)}
        for (label, score), cluster in zip(label_scores, cluster_assignments):
            clusters[cluster[0]].append((label, score))

        for i in list(clusters.keys()):
            cluster_labels = clusters[i]
            cluster_labels.sort(key=lambda x: x[1], reverse=True)

        sorted_clusters = sorted(clusters.values(), key=lambda cluster_labels: np.mean([score for _, score in cluster_labels if score != float('-inf')]), reverse=True)

        thresholds = np.linspace(0, 1, num=11)
        selected_labels = [""] * 11
        for i, threshold in enumerate(thresholds):
            selected_labels_clust = []
            for cluster_labels in sorted_clusters:
                num_to_select = max(1, int(len(cluster_labels) * threshold))
                selected_labels_clust.append(' '.join([label.replace(clip_word, "").replace(lebel_word, "") for label, _ in cluster_labels[:num_to_select]]))
                
            selected_labels[i] = ', '.join(selected_labels_clust)
        return selected_labels        

    def find_best_pair(image_features, labels, init_text=None):
        best_score = float('-inf')
        best_pair = []

        for i, label1 in enumerate(labels):
            for j, label2 in enumerate(labels):
                if i != j:
                    combined_text = f'{init_text} {label1}, {label2}' if init_text else f'{label1}, {label2}'
                    text_tensor = longclip.tokenize([combined_text]).to(device)
                    with torch.no_grad():
                        text_features = clip_model.encode_text(text_tensor)
                        text_features = F.normalize(text_features, dim=-1)
                        logits_per_image = (image_features @ text_features.T).item()
                    if logits_per_image > best_score:
                        best_score = logits_per_image
                        best_pair = [label1, label2]

        return best_pair, best_score

    def find_best_combined_text(image_features, labels, init_text=None, _iter=10):
        best_labels = []
        best_score = float('-inf')
        last_score = float('-inf')
        i = 0
        while len(labels) > 1:
            best_pair, pair_score = find_best_pair(image_features, labels, init_text)
            if not best_pair or pair_score <= last_score or i == _iter:
                break
            last_score = pair_score
            best_labels.extend(best_pair)
            labels = [label for label in labels if label not in best_pair]
            init_text = f"{init_text}, {', '.join(best_pair)}" if init_text else ', '.join(best_pair)
            i += 1

        return best_labels

    def find_best_concat(image_features, labels, long_labels, last_label= ''):
        best_label, best_score = '', float('-inf')
        labels_length = len(labels)

        #combined_text = ', '.join(long_labels)
        #long_text_tensor = longclip.tokenize([combined_text]).to(device)

        #with torch.no_grad():
        #    long_text_features = clip_model.encode_text(long_text_tensor)
        #    long_text_features = F.normalize(long_text_features, dim=-1)
        
        while True:
            for label in labels:
                combined_text = f'{last_label}, {label}'
                text_tensor = longclip.tokenize([combined_text]).to(device)

                with torch.no_grad():
                    text_features = clip_model.encode_text(text_tensor)
                    text_features = F.normalize(text_features, dim=-1)
                    logits_per_image = (image_features @ text_features.T).item()
                    #logits_per_text = (long_text_features.squeeze() @ text_features.squeeze()).item()
                    #score = (logits_per_image + logits_per_text) / 2

                if logits_per_image > best_score:
                    best_score = logits_per_image
                    best_label = label
                    last_label = combined_text
                    labels.remove(label)
                    if labels_length > 10:
                        break
            if best_label is None:
                break
            best_label = None

        return last_label, best_score

    image_tensor = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_tensor)
        image_features = F.normalize(image_features, dim=-1) 
        
    labels, long_labels, clothes_labels, people_labels = [], [], [], []
    clothtag, persontag, peopletag, custom_keeptag = '', '', '', ''
    long_labels = [label.lower().strip() for label in long_caption.split(", ") if label.strip() and label not in long_labels and '"' not in label and not any(char.isupper() for char in label[1:])]
    parent_folder = Path(image_path).parent.name
    tag_from_folder = ""
    if args.not_char and "_" in parent_folder and parent_folder.split("_")[0].isdigit():
        tag_from_folder = parent_folder.split('_')[1].replace('_', ' ').strip().lower()    
    labels = [label.strip() for label in short_caption.split(", ") if label.strip() and label not in labels and label not in char_tags and label != tag_from_folder]
    is_solo = False 
    is_solo = "solo" in short_caption

    clip_scores = []
    for label in clip_labels:
        with torch.no_grad():
            logits_per_image = (image_features @ text_features_dict[label].T).item()
            clip_scores.append((label, logits_per_image))
    
    clip_scores.sort(key=lambda x: x[1], reverse=True)
    top_clip_labels = [clip_scores[0][0], clip_scores[1][0], clip_scores[3][0]] 

    if not is_solo:
        preson_scores = []
        for label in preson_labels:
            with torch.no_grad():
                logits_per_image = (image_features @ text_features_dict[label].T).item()
                preson_scores.append((label, logits_per_image))
        
        preson_scores.sort(key=lambda x: x[1], reverse=True)
        persontag = preson_scores[0][0]
        if persontag == 'focus on one person':
            is_solo = True
            
    if args.clothtag and is_solo:
        clothes_labels = [label.strip() for label in short_caption.split(", ") if label.strip() and label not in clothes_labels and label in clothing_tags]
        clothtags = []
        clothtags = find_best_combined_text(image_features, clothes_labels, 'the person is wearing', 3)
        clothtag = ' '.join(clothtags)
        labels = [label for label in labels if label not in clothtags]

    if args.peopletag and (is_solo or persontag == 'two persons'):
        people_labels = [label.strip() for label in short_caption.split(", ") if label.strip() and label not in people_labels and label in people_tags]
        peopletags = []
        peopletags = find_best_combined_text(image_features, people_labels, 'they are doing ', 1)
        peopletag = ' '.join(peopletags)
        labels = [label for label in labels if label not in peopletags]

    if args.custom_keeptag:
        custom_keeptags = []
        custom_keeptags = find_best_combined_text(image_features, labels, f'{args.custom_keeptag} ', 2)
        print(f"{args.custom_keeptag} {custom_keeptags}")
        custom_keeptag = ', '.join(custom_keeptags[:4])
        labels = [label for label in labels if label not in custom_keeptags]

    best_concat, best_score = find_best_concat(image_features, labels + top_clip_labels, long_labels)
    all_labels = set(best_concat.split(', '))    
    #labels = [label for label in short_caption.split(", ") if label not in all_labels and label not in char_tags and label != tag_from_folder] + long_labels + top_clip_labels
    #best_concat, best_score = find_best_concat(image_features, labels, best_concat)
    #all_labels = set(best_concat.split(', '))    
    labels = [label for label in short_caption.split(", ") + long_labels if label not in all_labels and label != tag_from_folder]

    for label in labels:
        if label not in text_features_dict:
            text_tensor = longclip.tokenize([label]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tensor)
                text_features = F.normalize(text_features, dim=-1)
            text_features_dict[label] = text_features
    
    label_scores = []
    image_info=[]
    image_info = [image_path, image_features, labels]

    for label in labels:
        with torch.no_grad():
            logits_per_image = (image_features @ text_features_dict[label].T).item()
        label_scores.append((label, logits_per_image))

    if args.debiased:
        filtered_label_scores = []
        for item in label_scores:
            if item[1] < average_score * 2:
                filtered_label_scores.append(item)
            else:
                print(f"Discarding label: {item[0]}, score: {item[1] / average_score}")
        label_scores = filtered_label_scores    
        
    if args.clustertag:
        selected_labels = cluster_labels(label_scores, is_solo, persontag, image)
    else:    
        thresholds = np.linspace(0, 1, num=11)
        selected_labels = [""] * 11
        total_labels = len(label_scores)
        for i, threshold in enumerate(thresholds):
            index = int(threshold * total_labels)
            if index <= total_labels:
                selected_labels[i] = ", ".join([label.replace(clip_word, "").replace(lebel_word, "") for label, _ in label_scores[:index]])
    final_score = best_score 
    return selected_labels, final_score, clothtag, persontag, peopletag, custom_keeptag, image_info, best_concat

def process_image(image_path, folder_chartag, args):
    """
    處理單個圖片，獲取標籤並存儲。修改以支持多進程數據傳遞。
    """

    def resize_image(image_path, max_size=448):
        """
        縮小圖像使其最大邊不超過 max_size，返回縮小後的圖像數據
        """
        image = Image.open(image_path)
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
        """
        處理features字典，移除指定模式的鍵值對並生成keep_tags字串。
        
        參數:
        features (dict): 包含特徵的字典。

        返回:
        (dict, str): 返回處理後的features字典和keep_tags字串。
        """        
        patterns_to_keep = [
            r'^anime.*$', r'^monochrome$', r'^.*background$', r'^comic$', r'^greyscale$', r'^sketch$', 
            r'^.*censor.*$', r'^.*_name$', r'^signature$', r'^.*_username$', r'^.*text.*$', 
            r'^.*_bubble$', r'^multiple_views$', r'^.*blurry.*$', r'^.*koma$', r'^watermark$', 
            r'^traditional_media$', r'^parody$', r'^.*cover$', r'^.*_theme$', r'^.*realistic$', 
            r'^oekaki$', r'^3d$', r'^.*chart$', r'^letterboxed$', r'^variations$', r'^.*mosaic.*$', 
            r'^omake$', r'^column.*$', r'^.*_(medium)$', r'^manga$', r'^lineart$', r'^.*logo$'            
            #r'^(from_side|from_behind|from_above|from_below)$', r'^(close_up|dutch_angle|downblouse|downpants|pantyshot|upskirt|atmospheric_perspective|fisheye|panorama|perspective|pov|rotated|sideways|upside_down|vanishing_point|straight-on)$', r'^(face|cowboy_shot|portrait|upper_body|lower_body|feet_out_of_frame|full_body|wide_shot|very_wide_shot|cut_in|cropped_legs|head_out_of_frame|cropped_torso|cropped_arms|cropped_shoulders|profile|group_profile)$', r'^(armpit_focus|ass_focus|back_focus|breast_focus|eye_focus|foot_focus|hand_focus|hip_focus|navel_focus|pectoral_focus|thigh_focus|soft_focus|solo_focus)$'
        ]
        keep_tags_set = set()
        if 'solo' in features or 'solo_focus' in features:
            patterns_to_keep.extend([r'^holding_.*$'])
            #, r'^.*grab.*$', r'^.*lift.*$', r'^.*pull$', r'^.*_own_.*$', r'^.*covered.*$', r'^.*_masturbation.*$', r'^.*out.*$', r'^.*_between_.*$'
        keys = list(features.keys())
        keys_to_delete = []
        
        for key in keys:
            for pattern in patterns_to_keep:
                regex = re.compile(pattern)
                if regex.match(key):
                    keep_tags_set.add(key.replace('_', ' '))
                    keys_to_delete.append(key)

        for key in keys_to_delete:
            if key in features:
                del features[key]
        
        keep_tags = ', '.join(keep_tags_set).rstrip(', ')
        
        return features, keep_tags

    def build_folder_chartag(text, folder_chartag):
        """
        构建folder_chartag字典
        输入: 字符串text, 集合chartags
        输出: folder_chartag字典
        """
        tags = [tag.strip() for tag in text.split(',')]
        folder_chartag = {} if folder_chartag is None else folder_chartag
        
        for tag in tags:
            if tag in char_tags:
                if tag in folder_chartag:
                    folder_chartag[tag] += 1
                else:
                    folder_chartag[tag] = 1
                    
        return folder_chartag

    def format_wd14_caption(wd14_caption):       
        tags = wd14_caption.split(", ")
        tags_to_delete = []
        lying_conditions = ['on stomach', 'on back', 'on side']
        if 'lying' in tags and any(cond in tags for cond in lying_conditions):
            for cond in lying_conditions:
                if cond in tags:
                    tags.append(f'lying {cond}')
                    tags_to_delete.append(cond)
            tags_to_delete.append('lying')
            
        boygirl_tags = [tag for tag in tags if tag in {'multiple girls', '1girl', 'multiple boys', '1boy'}]
        if boygirl_tags:
            boygirl_tag = ' '.join(sorted(boygirl_tags))
            tags.append(boygirl_tag)
            for tag in boygirl_tags:
                tags_to_delete.append(tag)         
        tags = [tag for tag in tags if tag not in tags_to_delete]
        wd14_caption = ', '.join(tags)
        return wd14_caption

    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')

    # 檢查文件最後修改時間，如果在一周內則略過
    if tag_file_path.exists():
        last_modified_time = datetime.fromtimestamp(tag_file_path.stat().st_mtime)
        if datetime.now() - last_modified_time < timedelta(days=args.continue_caption):
            print(f"Skipping {tag_file_path} as it was modified within the last week.")
            return None, None, 'skipped'   
    try:
        image = resize_image(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 使用 imgutils 獲取圖片等級
        rating, features, chars = get_wd14_tags(image, character_threshold=0.6, general_threshold=0.2682, drop_overlap=True)
        #features, keeptag = process_features(features)
        #features = drop_basic_character_tags(features)
        wd14_caption = tags_to_text(features, use_escape=False, use_spaces=True)
        special_text, chartags, boorutag, artisttag = generate_special_text(image_path, args, features, chars)
        ratingtag = max(rating, key=rating.get)
        #wd14_caption = wd14_caption + ', ' + boorutag
        #wd14_caption = format_wd14_caption(wd14_caption)
        wd14_caption = wd14_caption + ', ' + boorutag
        wd14_caption = transform_caption(wd14_caption)
        more_detailed_caption, _ = run_example('<MORE_DETAILED_CAPTION>', image) 
        #clip_caption = []
        #clip_caption, final_score, clothtag, persontag, peopletag, custom_keeptag, image_info, best_concat = calculate_best_labels(image, wd14_caption, more_detailed_caption, image_path)
        aestag = get_aesthetic_tag(image)
        #folder_chartag = build_folder_chartag(clip_caption[10], folder_chartag) 
        #others_info = ""
#        if persontag:
#            special_text = f"{persontag} " + special_text
        if args.not_char:
            parent_folder = Path(image_path).parent.name
            concept_tag = f"{parent_folder.split('_')[1].replace('_', ' ').strip()} is/looks as follows: "
            special_text = f"{concept_tag}, " + special_text 
#        if keeptag:
#            special_text = f"{keeptag}, " + special_text
        if aestag:
            special_text += f", looks {aestag}"
#        if clothtag:
#            special_text += f", with {clothtag}"
#        if peopletag:
#            special_text += f", are {peopletag}"
#        if custom_keeptag:
#            special_text += f", {args.custom_keeptag} {custom_keeptag}"
#        if best_concat:
#            special_text += f", {best_concat}"
        if ratingtag:
            special_text += f", rating:{ratingtag}"
#        if artisttag:
#            special_text += f", {artisttag}"
        
        special_text = ' '.join([text.strip() for text in special_text.split(',') if text.strip()])
        
        if not args.rawdata and None:           
            tags_text = (
                f"{special_text}, {more_detailed_caption}"
            )
        else:
            tags_text =(
                f"{special_text}, {more_detailed_caption}, {wd14_caption}"
            )            
        with open(tag_file_path, 'w', encoding='utf-8') as f:
            f.write(tags_text.lower()) 
#        return folder_chartag, final_score, image_info, clip_caption
    except Exception as e:
        print(f"Failed to process image {image_path}: {e}")
        traceback.print_exc()

def drop_chartags_in_folder(folder_path, folder_chartag):
    """
    在指定目录中删除高频chartag
    输入: 目录路径folder_path, folder_chartag字典
    """
    max_count = max(folder_chartag.values())    
    threshold = max_count / 3
    tags_to_drop = {tag for tag, count in folder_chartag.items() if count > threshold}
    
    # 遍历目录中的每个txt文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            
            for i, line in enumerate(lines):
                new_content = []
                tags = [tag.strip() for tag in line.split(',')]
                for tag in tags:
                    if tag and tag not in tags_to_drop:
                        new_content.append(tag)
                lines[i] = ', '.join(new_content)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(lines))

def drop_features_in_folder(root, image_infos_list, drop_percent = 0.3):
    # 合并所有的 image_features
    combined_image_features = torch.cat([info[1] for info in image_infos_list]).mean(dim=0, keepdim=True)
    combined_image_features = F.normalize(combined_image_features, dim=-1)

    for image_info in image_infos_list:
        image_path, _, labels = image_info
        final_labels = {}

    all_labels_list = [label for info in image_infos_list for label in info[2]]
    all_labels = set(all_labels_list)
    final_labels = {}

    for label in all_labels:
        with torch.no_grad():
            logits_per_image = (combined_image_features @ text_features_dict[label].T).item()
            final_labels[label] = logits_per_image

    sorted_labels = sorted(final_labels.items(), key=lambda item: item[1], reverse=True)
    top_percent_index = max(1, int(len(sorted_labels) * drop_percent))
    
    tags_to_drop = {label.replace(lebel_word, '').replace(clip_word, '') for label, _ in sorted_labels[:top_percent_index] if all_labels_list.count(label) > len(image_path) * 0.1}
    print(tags_to_drop)
    for image_info in image_infos_list:
        image_path, _, labels = image_info

        file_path = Path(image_path).with_suffix('.txt')
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            for i, line in enumerate(lines):
                new_content = []
                tags = [tag.strip() for tag in line.split(',')]
                for tag in tags:
                    if tag and tag not in tags_to_drop:
                        new_content.append(tag)
                lines[i] = ', '.join(new_content)
            
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write('\n'.join(lines))

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
                folder_chartag, final_score, image_info, clip_caption = "","","",""
                process_image(image_path, folder_chartag, args)  
                all_final_scores.append((image_path, final_score))
                folder_final_scores.append((image_path, final_score, clip_caption))
                image_infos_list.append(image_info)
            except Exception as e:
                print(f"Failed to process image {image_path}: {e}")
                traceback.print_exc()
        if len(folder_final_scores) != 0 and False:    
            max_score = max(folder_final_scores, key=lambda x: x[1])[1]
            min_score = min(folder_final_scores, key=lambda x: x[1])[1]
            if min_score != max_score:
                for image_path, final_score, clip_caption in folder_final_scores:
                    relative_score = (final_score - min_score) / (max_score - min_score)
                    relative_score = np.log1p(relative_score * 9) / np.log(10)
                    relative_number = int(((1 - relative_score) * 8) + 2)
                    tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')
                    if tag_file_path.exists():
                        with open(tag_file_path, 'r', encoding='utf-8') as file:
                            content = file.read()
                            content = content.replace(f'{clip_caption[10]}', f'{clip_caption[relative_number]}') 
                            with open(tag_file_path, 'w', encoding='utf-8') as file:
                                file.write(content)
        if args.drop_chartag and folder_chartag:
            drop_chartags_in_folder(root, folder_chartag)

        if image_infos_list and args.autodroptag !=0:
            drop_features_in_folder(root, image_infos_list, args.autodroptag)
            
    if all_final_scores:
        max_score = max(all_final_scores, key=lambda x: x[1])[1]
        min_score = min(all_final_scores, key=lambda x: x[1])[1]
        if min_score != max_score:
            for image_path, final_score in all_final_scores:
                relative_score = (final_score - min_score) / (max_score - min_score)
                relative_score = np.log1p(relative_score * 9) / np.log(10)
                if relative_score >= 0.3:
                    accuracy_tag = ""
                elif relative_score >= 0.1:
                    accuracy_tag = "low accuracy."
                else:
                    accuracy_tag = "mess."
                
                tag_file_path = Path(image_path).with_suffix('').with_suffix('.txt')
                if tag_file_path.exists():
                    with open(tag_file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    if accuracy_tag:
                        content = content.replace('___', f'{accuracy_tag}, ___') 
                        # 将修改后的内容写回文件
                        with open(tag_file_path, 'w', encoding='utf-8') as file:
                            file.write(content)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="圖片標籤處理腳本")
    parser.add_argument("--folder_name", action="store_true", help="使用目錄名當作角色名")
    parser.add_argument("--drop_chartag", action="store_true", help="自動刪除角色特徵標籤")
    parser.add_argument("--drop_colortag", action="store_true", help="自動刪除顏色特徵標籤")
    parser.add_argument("--clothtag", action="store_true", help="自動處理服裝標籤")
    parser.add_argument("--not_char", action="store_true", help="目錄名不是角色")
    parser.add_argument("--debiased", action="store_true", help="設置clip score上限減少florence偏差")
    parser.add_argument("--peopletag", action="store_true", help="前置多人標籤(大多nsfw)")
    parser.add_argument("--custom_keeptag", type=str, default=None, help="自定義自動留標")
    parser.add_argument("--rawdata", action="store_true", help="大資料集")
    parser.add_argument("--continue_caption", type=int, default=0, help="忽略n天內打的標")
    parser.add_argument("--clustertag", action="store_true", help="對標籤聚類")
    parser.add_argument("--autodroptag", type=float, default=0, help="自動刪標，刪除跟資料集太接近的標，小數點是比例")
    parser.add_argument("directory", type=str, help="處理目錄地址")
    args = parser.parse_args()
    if args.not_char:
        args.folder_name = True
        
    clip_labels = [f"{clip_word}{label}" for label in clip_labels]
    for label in clip_labels + view_labels + preson_labels:
        if label not in text_features_dict:
            text_tensor = longclip.tokenize([label]).to(device)
            with torch.no_grad():
                text_features = clip_model.encode_text(text_tensor)
                text_features = F.normalize(text_features, dim=-1)
            text_features_dict[label] = text_features
    find_and_process_images(args.directory, args)