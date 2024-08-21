集合wd tagger v3、aesthetic-predictor-v2-5和joycaption

謝謝imgutils、Initial_Elk5162的腳本


使用方式：


git clone https://github.com/gesen2egee/all_in_one_caption

cd all_in_one_caption

python -m venv venv

.\venv\Scripts\activate (每次載入虛擬環境要用這個指令)

pip install -r requirements.txt

python caption.py "資料集目錄" --folder_name --del_tag


--del_tag 刪除在子資料夾>70%的WD標

--folder_name 用子資料夾數字後面的詞當作的第一個詞

--folder_name = 前置子目錄名


需求VRAM 12G

12 s/it on 3080

打標參考

![image](https://github.com/user-attachments/assets/4f4f4488-e036-4aaa-9324-829f18cb7491)

![image](https://github.com/user-attachments/assets/2063eff4-0eed-402d-b91a-bc1c36476432)

