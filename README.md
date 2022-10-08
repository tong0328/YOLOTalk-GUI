# YoloTalk install tutorial


## Step 1. 下載 YoloTalk 檔案 (此步驟請參考健菖YOLOTalk 流程)
```bash=
git clone https://github.com/jim93073/YoloTalk.git

cd YoloTalk
git clone https://github.com/adipandas/multi-object-tracker.git
git clone https://github.com/AlexeyAB/darknet.git
```

## Step 2. 於 YoloTalk 中 新增 Flask_web 資料夾  此為 Flask GUI 檔案夾



## Step 3. 安裝套件並修改 YOLOTalk.py 內的 port、()



## Step 4. (Option) 設置 YoloDevice 物件參數


## Step 5: (Option) 設置IoTtalk與LineBot
> 設置IoTTalk與LineBot的連結


make: *** [obj/network_kernels.o] Error 1
```

編輯```src/network_kernels.cu```，註解```CHECK_CUDA(cudaStreamBeginCapture(stream0, cudaStreamCaptureModeGlobal));```(約在721行)，並執行以下指令，重新編譯 darknet

```bash=
make clean
make
```
