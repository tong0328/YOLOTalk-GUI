# YoloTalk install tutorial


## Step 1. 下載 YoloTalk 檔案 (此步驟請參考健菖YOLOTalk 流程)
```bash=
git clone https://github.com/jim93073/YoloTalk.git

cd YoloTalk
git clone https://github.com/adipandas/multi-object-tracker.git
git clone https://github.com/AlexeyAB/darknet.git
```

## Step 2. 於 YOLOTalk 中 新增 Flask_web 資料夾  此為 Flask GUI 檔案夾

專案架構為

----- YOLOTalk-GUI
  |
  |-- libs (健菖的YOLO檔案)
  |        |
  |        |--YOLO_SSIM.py
  |        |--YOLO.py
  |        |--utils.py
  |
  |--darknet  
  |--multi-object-tracker
  |--weights
  |--cfg_person
  |--Flask_web ★★★
           |
           |--YOLOTalk.py  (Flask)
                   |--templates
                   |--static
                        |--Json_Info
                        |--alias_pict
                        |--record
                        
                        
## Step 3. 安裝套件並修改參數 

修改 YOLOTalk.py 內的 
1.app.run 內的 host、port 參數
2.import YoloDevice 所需的 YOLO_SSIM.py、libs 資料夾
3.修改各 html 內的 POST 參數


## 其餘資料夾說明 

1. Flask_web/static/Json_Info  : 放置各個圍籬紀錄參數檔案
2. Flask_web/static/alias_pict : 放置各個圍籬所擷取圖片，用於 plotarea 功能繪製圍籬的底圖
3. Flask_web/static/record     : 放置 YOLO_SSIM.py 所記錄之照片、圖片
