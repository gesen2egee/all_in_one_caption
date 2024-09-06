安裝方式：


git clone https://github.com/gesen2egee/all_in_one_caption

cd all_in_one_caption

python -m venv venv

.\venv\Scripts\activate (每次載入虛擬環境要用這個指令)

pip install -r requirements.txt




(更新)

自動分類圖片

python class.py "資料集目錄" --class 20

用siglip將圖片依照提取特徵聚類，分成20類移動到子資料夾


(再更)

WD + 新的florence-2微調自然語言

使用 main_script.py

python main_script.py "資料集目錄" --folder_name  --del_tag

--del_tag 刪除出現比例>50%的標
--folder_name 用子資料夾數字後面的詞當作的第一個詞


對於FLUX原有不懂概念及姿勢服從應該更好

如果已經安裝 需要重新安裝pip install -r requirements.txt



(更新)

根據

https://civitai.com/articles/6792/flux-captioning-differences-training-diary

https://civitai.com/articles/6982


FLUX訓練更適合用簡單標

增加wdcaption.py

python wdcaption.py "資料集目錄" --folder_name --del_tag

用WD自動刪除出現頻率太高或太相關的標


--folder_name 用子資料夾數字後面的詞當作的第一個詞


===========================================================================


集合wd tagger v3、aesthetic-predictor-v2-5和joycaption

謝謝imgutils、Initial_Elk5162的腳本




python caption.py "資料集目錄" --folder_name --del_tag


--del_tag 刪除在子資料夾>70%的WD標

--folder_name 用子資料夾數字後面的詞當作的第一個詞



需求VRAM 12G

12 s/it on 3080

打標參考

![image](https://github.com/user-attachments/assets/4f4f4488-e036-4aaa-9324-829f18cb7491)

![image](https://github.com/user-attachments/assets/2063eff4-0eed-402d-b91a-bc1c36476432)

