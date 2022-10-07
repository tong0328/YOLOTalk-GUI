from libs.YOLO import YoloDevice
from darknet import darknet
import libs.DAN as DAN
import LineNotify


if __name__ == '__main__': 
    ServerURL = 'rtsp://lab610:lab610@140.113.131.8:554/stream1'    
    Reg_addr = '555642434' #if None, Reg_addr = MAC address
    DAN.profile['dm_name']='Yolo_Device'
    DAN.profile['df_list']=['yPerson-I',]
    DAN.profile['d_name']= 'YOLOjim'

    # DAN.device_registration_with_retry(ServerURL, Reg_addr)
    # DAN.deregister()  #if you want to deregister this device, uncomment this line
    # exit()            #if you want to deregister this device, uncomment this line   


    # results:[(class, confidence, (center_x, center_y, width, height), id), (...)]
    def on_data(img_path, group, alias, results): 

        for det in results:
            class_name = det[0]
            confidence = det[1]
            center_x, center_y, width, height = det[2]
            left, top, right, bottom = darknet.bbox2points(det[2])
            print(class_name, confidence, center_x, center_y)
        if len(results) > 0:
            LineNotify.line_notify(class_name)   # 要在LINENOTIFY.py 改 token
#             DAN.push('yPerson-I', str(class_name), center_x, center_y, img_path)

# 以下新增 
    yolo1 = YoloDevice(
        config_file = './darknet/cfg/yolov4-tiny.cfg',
        data_file = './darknet/cfg/coco.data',
        weights_file = './weights/yolov4-tiny.weights',
        thresh = 0.6,                 # 要更改
        output_dir = '',              # 
        video_url = 'rtsp://lab610:lab610@140.113.131.8:554/stream1',   # 要更改
        is_threading = True,         # rtsp  ->true  video->false
        vertex = None,                # 要更改    
        alias="demo",                 # 要更改
        display_message = True,
        obj_trace = True,        
        save_img = True,
        save_video = True,            # 再改成 false
        target_classes=["person"],
        auto_restart = True,
     )    

    yolo1.set_listener(on_data)
    yolo1.start()  

    
    # 1.新增URL
    # 2.傳到 YOLODEVICE 內 改參數
    # 3.啟動YOLO
    # 4.點 NAV 上的 video 可以看見新增的內容 video 是可以及時更新畫面的，點擊後可以放大