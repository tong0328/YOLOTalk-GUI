# -*- coding: utf-8 -*
# import below is jim's YOLOtalk code
import sys
sys.path.append("..")
from libs.YOLO_SSIM import YoloDevice
from libs.utils import * 
from darknet import darknet
from config import Config
from interval import Interval

import os
import shutil
import sys
import json
import cv2
import numpy as np
import time
import datetime

# import LineNotify
import LineNotify_TONG


def GET_POST_URL(url):
    if Config["host"] == "0.0.0.0":
        postURL = os.path.join("http://panettone.iottalk.tw:", Config["port"], f"/{url}")
    else:
        postURL = os.path.join("http://", Config["host"], ":", Config["port"], f"/{url}")
    return postURL

def connectOPENCV(URL : str):
    fig = cv2.VideoCapture(URL)
    stat, Img = fig.read()

    return stat, Img
    
def Restart_YoloDevice():
    oldJson = os.listdir(r"static/Json_Info")
    actived_yolo = {}
    for j in oldJson :
        if ".json" not in j:
            continue

        path = os.path.join("static/Json_Info", j)
        with open(path, 'r', encoding='utf-8') as f:             
            Jdata = json.load(f)

        alias = Jdata["alias"]
        URL   = Jdata["viedo_url"]

        if Jdata["fence"]  == {}:
            vertex = None

        else:
            key_list = list(Jdata["fence"].keys())
            vertex = {}

            for key in key_list :
                old_vertex = Jdata["fence"][key]["vertex"][1:-1]
                # vertex
                new_vertex = transform_vertex(old_vertex)
                vertex[key] = new_vertex 
                # sensitivity
                old_sensitivity = float(Jdata["fence"][key]["Sensitivity"])

            yolo1 = YoloDevice(
                            config_file = '../darknet/cfg/yolov4-tiny.cfg',
                            data_file = '../darknet/cfg/coco.data',
                            weights_file = '../weights/yolov4-tiny.weights',
                            thresh = old_sensitivity,                 
                            output_dir = './static/record/',              
                            video_url = URL,              
                            is_threading = True,          
                            vertex = vertex,                 
                            alias = alias,                
                            display_message = False,
                            obj_trace = True,        
                            save_img = True,
                            img_expire_day = 1,
                            save_video = False,
                            video_expire_day = 1,           
                            target_classes = ["person"],
                            auto_restart = False,
                            )    

            print(f"\n======== Activing YOLO , alias:{alias}========\n")
            yolo1.set_listener(on_data)
            yolo1.start()
            actived_yolo[alias] = yolo1
        # ======== FOR YOLO ========
    return actived_yolo


def read_all_fences():
    oldFences = os.listdir(r'static/alias_pict')
    oldFences.sort()
    all_fences_names = []
    for name in oldFences :
        if "ipynb" in name:
            continue
        name = name[:-4]
        all_fences_names.append(name)
    
    return all_fences_names

tmp = "19970101000000"
tmp = datetime.datetime.strptime(tmp, "%Y%m%d%H%M%S")
def on_data(img_path, group, alias, results): 
    now = datetime.datetime.now()
    global tmp
    for det in results:
        class_name = det[0]
        confidence = det[1]
        center_x, center_y, width, height = det[2]
        left, top, right, bottom = darknet.bbox2points(det[2])
        ids = det[3]
        img_path = f"http://panettone.iottalk.tw:10328/{img_path}"
        msg = f"\n場域:{alias},\n網址:\n{img_path}"
        if int(now.strftime("%Y%m%d%H%M%S"))>int((tmp+datetime.timedelta(seconds=3)).strftime("%Y%m%d%H%M%S")):
            if len(results) > 0:   
                tmp = now    
                LineNotify_TONG.line_notify(msg)   # LINENOTIFY.py  token
                # print(alias, img_path, ids)
                # print(now.strftime("%Y%m%d%H%M%S"))     
                # print(int((tmp+datetime.timedelta(seconds=10)).strftime("%Y%m%d%H%M%S")))
                # DAN.push('yPerson-I', str(class_name), center_x, center_y, img_path)


def transform_vertex(old_vertex):
    """
        transform vertex from [x1,y1,x2,y2,x3,....]  to [(x1,y1),(x2,y2),...]
    """
    new_vertex = []
    vertexs = old_vertex.split(",") 

    for i in range(0 ,len(vertexs), 2):
        new = (int(vertexs[i]), int(vertexs[i+1]))
        new_vertex.append(new)    

    return new_vertex


def gen_frames(yolo):

    def fill_mask(active:bool, frame, vertex:dict, mask:np.array, pts:list, color):
        if active != False :
            # for singal_vertex in vertex.values():
            temp =[]
                # for point in singal_vertex :
                    # temp.append(point)
                # pts.append(np.array(temp, dtype=np.int32))
            pts.append(np.array(vertex, dtype=np.int32))
            mask = cv2.fillPoly(mask, pts, color)     # Filling the mask of polygon 
            frame =  0.5 * mask + frame

            return frame
    print(f"========{yolo.alias}  YOLO 影像讀取中========")

    time.sleep(0.5)
    filepath = f"static/Json_Info/camera_info_{str(yolo.alias)}.json"
    with open(filepath, 'r', encoding='utf-8') as f:                    
        Jdata = json.load(f)
    fence_list = list(Jdata["fence"].keys())
    vertex = {}
    for fence in fence_list :
        old_vertex = Jdata["fence"][fence]["vertex"][1:-1]
        new_vertex = transform_vertex(old_vertex)
        vertex[fence] = new_vertex

    # use this to compute mask showing time    
    detect_target = 0

    # judge time every minute 
    oldtime_min =  -1
    newtime_min =  time.localtime(time.time()).tm_min

    while True:
        frame = yolo.get_current_frame()
        # judge time schedule
        newtime_min =  time.localtime(time.time()).tm_min
        if oldtime_min != newtime_min :
            schedule_on_dict = time_interval(yolo)     
            oldtime_min = newtime_min

        # draw the polygon lines
        frame = draw_polylines(frame, vertex)  
        mask = np.zeros((frame.shape), dtype = np.uint8)
        pts = []

        # plot mask in vertex if suchedule == True
        for fence in schedule_on_dict.keys():
            for order in schedule_on_dict[fence].keys():
                if schedule_on_dict[fence][order] == True:
                    # Filling mask
                    if  len(yolo.detect_target) != 0 :
                        print(f"[Detect] {yolo.detect_target[:][:2]}")
                        frame = fill_mask(True, frame, vertex[fence], mask, pts, (180, 0, 255))
                        detect_target = 0              # count mask time
                    else:
                        if detect_target < 3 :
                            frame = fill_mask(True, frame, vertex[fence], mask, pts, (180, 0, 255))
                            detect_target +=1
                else:
                    frame = fill_mask(True, frame, vertex[fence], mask, pts, (255, 255, 255))
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def time_interval(yolo):
    filepath = f"static/Json_Info/camera_info_{str(yolo.alias)}.json"
    with open(filepath, 'r', encoding='utf-8') as f:                     
        Jdata = json.load(f)
    Fence_list = list(Jdata["fence"].keys())
    schedule_on_dict = {}

    for fence in Fence_list:
        
        Schedule_list = list(Jdata['fence'][fence]['Schedule'].keys())
        data = {}   # used to save every order in fence 

        for Order in Schedule_list:
            # Jdata['fence'][FenceName]['Group'] = Group
            Start_time = Jdata['fence'][fence]['Schedule'][Order]['Start_time']  
            End_time   = Jdata['fence'][fence]['Schedule'][Order]['End_time']    
    
            nowtime = time.strftime("%H:%M:%S", time.localtime())

            if Start_time != "--:--" and End_time != "--:--":

                now_time = Interval(nowtime, nowtime)
                time_interval_one = Interval(Start_time, End_time)

                if now_time in time_interval_one :
                    if data == {}:
                        data = {Order : True}
                    else :
                        data[Order] = True
                else :
                    if data == {}:
                        data = {Order : False}
                    else :
                        data[Order] = False
                schedule_on_dict[fence] = data

            if Start_time == "--:--" and End_time == "--:--":
                if data == {}:
                    data = {Order : True}
                else :
                    data[Order] = True
                schedule_on_dict[fence] = data
    return schedule_on_dict


def replot(alias, URL, Addtime):
    data  = {"alias":"",
        "viedo_url":"",
        "add_time":"",
        "fence": {}
        }
    # print("REPLOT")
    IMGpath = "static/alias_pict/"+str(alias)+".jpg"
    data["alias"]=alias
    data["viedo_url"]=URL
    data["add_time"]=Addtime

    filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
    with open(filepath, 'r', encoding='utf-8') as f:                    
        Jdata = json.load(f)
    old_fence_data = Jdata['fence']

    fences = list(old_fence_data.keys())

    vertexs = []

    fig = cv2.imread(IMGpath)
    shape = fig.shape

    for fence in fences:
        old_vertex = old_fence_data[fence]['vertex']
        vertexs.append(old_vertex)

    return IMGpath, shape