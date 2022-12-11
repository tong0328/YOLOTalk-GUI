import os
import shutil
import sys
import json
import cv2
import numpy as np
import time
from interval import Interval

# import below is jim's YOLOtalk code
import sys
sys.path.append("..")
sys.path.insert(1, '../multi-object-tracker') 
from libs.YOLO_SSIM import YoloDevice 
from darknet import darknet
from libs.utils import *



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
                    config_file = '../cfg_person/yolov4-tiny.cfg',
                    data_file = '../cfg_person/coco.data',
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
                    save_video = False,           
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
    # print(f"oldFences : {oldFences}")
    all_fences_names = []
    for name in oldFences :
        if "ipynb" in name:
            continue
        name = name[:-4]
        all_fences_names.append(name)
    
    return all_fences_names


def on_data(img_path, group, alias, results): 

    for det in results:
        class_name = det[0]
        confidence = det[1]
        center_x, center_y, width, height = det[2]
        left, top, right, bottom = darknet.bbox2points(det[2])
#             print(class_name, confidence, center_x, center_y)
#         if len(results) > 0:            
#             LineNotify.line_notify(class_name)   # LINENOTIFY.py  token
#             DAN.push('yPerson-I', str(class_name), center_x, center_y, img_path)


def transform_vertex(old_vertex):
    # transform vertex from [x1,y1,x2,y2,x3,....]  to [(x1,y1),(x2,y2),...]
    new_vertex = []
    vertexs = old_vertex.split(",") 

    for i in range(0 ,len(vertexs), 2):
        new = (int(vertexs[i]), int(vertexs[i+1]))
        new_vertex.append(new)    

    return new_vertex



def gen_frames(yolo):

    def fill_mask(active:bool, frame, vertex:dict, mask:np.array, pts:list):
        if active != False :
            for singal_vertex in vertex.values():
                temp =[]
                for point in singal_vertex :
                    temp.append(point)
                pts.append(np.array(temp, dtype=np.int32))
            mask = cv2.fillPoly(mask, pts, (180,0,255))     # Filling the mask of polygon 
            frame =  0.5 * mask + frame

            return frame
    time.sleep(0.5)

    print("========YOLO 影像讀取中========")

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
    print(oldtime_min)
    print(newtime_min)

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

        # plot mask in vertex
        for fence in schedule_on_dict.keys():    

            if schedule_on_dict[fence] == True :

                # Filling mask
                if  len(yolo.detect_target) != 0 :
                    print(f"[Detect] {yolo.detect_target[:][:2]}")
                    frame = fill_mask(True, frame, vertex[fence], mask, pts)
                    detect_target = 0              # count mask time
                else:
                    if detect_target < 3 :
                        frame = fill_mask(True, frame, vertex[fence], mask, pts)
                        detect_target +=1

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def time_interval(yolo):

    print("Judge time schedule")
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
    # print(f"now time : {now_time}")
    # print(schedule_on_dict)
    return schedule_on_dict


def replot(alias, URL, Addtime):
    data  = {"alias":"",
        "viedo_url":"",
        "add_time":"",
        "fence": {}
        }
    print("REPLOT")
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