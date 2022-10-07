from flask import Flask, render_template,Response ,request ,jsonify,redirect
import time
import numpy as np
import threading
import cv2
import requests
import json
import os
import jinja2

# import below is jim's YOLOtalk code
import sys
sys.path.append("..") 
from libs.YOLO_SSIM import YoloDevice
from darknet import darknet
from libs.utils import *


app = Flask(__name__)

oldFences = os.listdir(r'static/alias_pict')
FencesNames = []
for name in oldFences :
    if "ipynb" in name:
        continue
    name = name[:-4]
    FencesNames.append(name)


# Temporary information
data  = {"alias":"",
         "viedo_url":"",
         "add_time":"",
         "fence": {}
        }

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
    a = old_vertex.split(",") 

    for i in range(0,len(a),2):
        x = (int(a[i]), int(a[i+1]))
        new_vertex.append(x)    

    return new_vertex


# active YOLO already in Json file
oldJson = os.listdir(r"static/Json_Info")
actived_yolo = {}
for j in oldJson :
    if "ipynb" in j:
        continue

    path = os.path.join("static/Json_Info", j)
    with open(path, 'r', encoding='utf-8') as f:             
        Jdata = json.load(f)

 
    alias = Jdata["alias"]
    URL   = Jdata["viedo_url"]

    if Jdata["fence"]  == {}:
        print("fence is none")
        vertex = None
    else:
        key_list = list(Jdata["fence"].keys())
        vertex = {}
        for key in key_list :
            old_vertex = Jdata["fence"][key]["vertex"][1:-1]

            new_vertex = transform_vertex(old_vertex)
            vertex[key] = new_vertex 

        print("===========================")
        print(f" Alias name is :  {alias}")
        print(f" Fence name is : {key_list}")
        print(f" Json is : {j}")
        print(f" Vertex is :{vertex}")

# ======== FOR YOLO ========
        yolo1 = YoloDevice(
                        config_file = '../darknet/cfg/yolov4-tiny.cfg',
                        data_file = '../darknet/cfg/coco.data',
                        weights_file = '../weights/yolov4-tiny.weights',
                        thresh = 0.3,                 # need modify (ok)
                        output_dir = './static/record/',              
                        video_url = URL,              # need modify (ok) 
                        is_threading = True,          # rtsp  ->true  video->false (ok)
                        vertex = vertex,              # need modify (ok)    
                        alias = alias,                # need modify (ok)
                        display_message = True,
                        obj_trace = True,        
                        save_img = False,
                        save_video = False,           # modify to False
                        target_classes = ["person"],
                        auto_restart = False,
                        )    
        yolo1.set_listener(on_data)
        yolo1.start()
        actived_yolo[alias] = yolo1
# ======== FOR YOLO ========

def gen_frames(yolo):

    filepath = "static/Json_Info/camera_info_" + str(yolo.alias) + ".json"
    with open(filepath, 'r', encoding='utf-8') as f:                    
        Jdata = json.load(f)

    key_list = list(Jdata["fence"].keys())
    vertex = {}

    while True:
        frame = yolo.get_current_frame()
        # copy_frame = frame

        for key in key_list :

            old_vertex = Jdata["fence"][key]["vertex"][1:-1]
            new_vertex = transform_vertex(old_vertex)
            vertex[key] = new_vertex

        frame = draw_polylines(frame, vertex)  # draw the polygon
            
        if len(yolo.detect_target) != 0:
            mask = np.zeros((frame.shape), dtype = np.uint8)
            pts = []

            
            for singal_vertex in vertex.values():
                temp =[]
                for point in singal_vertex :
                    temp.append(point)
                pts.append(np.array(temp, dtype=np.int32))

            mask = cv2.fillPoly(mask, pts, (180,0,255))  # Filling the mask of polygon 
            frame =  0.5 * mask + frame  

            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/',methods=[ 'GET','POST'])
def home():
    print(f"[actived_yolo] :{actived_yolo} ")
    if request.method == 'POST':

        alias  = request.form.get('area')
        URL        = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]

        
        if URL =="REPLOT":
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

            for fence in fences:
                old_vertex = old_fence_data[fence]['vertex']
                vertexs.append(old_vertex)  

            return render_template('plotarea.html', data = IMGpath, name=str(alias))
        
        else:    

            fig = cv2.VideoCapture(str(URL))
            stat, I = fig.read()

            data["alias"]     = alias
            data["viedo_url"] = URL
            data["add_time"]  = Addtime
            

            filepath = "static/Json_Info/camera_info_" + data["alias"] + ".json"
            with open(filepath, 'w', encoding='utf-8') as file:             # 
                json.dump(data, file,separators=(',\n', ':'),indent = 4)
        
        
            IMGpath = "static/alias_pict/"+str(alias)+".jpg"
            FencesNames.append(str(alias))
            cv2.imwrite(IMGpath,I)

# ======== FOR YOLO ========
            yolo1 = YoloDevice(
                    config_file = '../darknet/cfg/yolov4-tiny.cfg',
                    data_file = '../darknet/cfg/coco.data',
                    weights_file = '../weights/yolov4-tiny.weights',
                    thresh = 0.3,                 # need modify (ok)
                    output_dir = './static/record/',              
                    video_url = URL,              # need modify (ok) 
                    is_threading = True,          # rtsp  ->true  video->false (ok)
                    vertex = None,                # need modify (ok)    
                    alias = alias,                  # need modify (ok)
                    display_message = True,
                    obj_trace = True,        
                    save_img = False,
                    save_video = False,           # modify to False
                    target_classes = ["person"],
                    auto_restart = False,
                    )    
            yolo1.set_listener(on_data)
            yolo1.start()
            actived_yolo[alias] = yolo1
# ======== FOR YOLO ========
        
            return render_template('plotarea.html', data = IMGpath, name=str(alias))
     
    return render_template('home.html', navs = FencesNames)
    


@app.route('/plotarea',methods=[ 'GET','POST'])
def plotarea():
    if request.method == 'POST':
        
        print("plotarea enter & POST")
        
        alias = request.form['alias']
        FenceName = request.form['FenceName']   # plot name
        vertex = request.form['vertex']         # plot point

        Fence_info={'vertex':vertex,
                    'Group':'-',                
                    'Alarm_Level':'General',    # General, High
                    'Note':' - ',           
                    'Sensitivity':'0.5',
                    'Schedule':{
                                    '1':{'Start_time':'--:--','End_time':'--:--'},
                               }
                   }
     
        filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
        print("Json File Path : " , filepath)
        if os.path.isfile(filepath):                                        # If file is exist
            with open(filepath, 'r', encoding='utf-8') as f:                
                Jdata = json.load(f)

            if vertex == "DELETE" :
                Jdata["fence"].pop(FenceName)                              
            else:
                Jdata["fence"][FenceName]=Fence_info                        
                # ======== FOR YOLO ========
                old_vertex = vertex[1:-1]
                new_vertex = transform_vertex(old_vertex)
                data = { FenceName : new_vertex }
                actived_yolo[alias].vertex = data
                print(f"alias:{alias}")
                print(actived_yolo[alias].vertex)
                # ======== FOR YOLO ======== 
            with open(filepath, 'w', encoding='utf-8') as file:             
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)
            
        else:                                                               
            print("\nFile doesn't exist !! \n")
                                  # 
            data["fence"][FenceName]=Fence_info                             
            with open(filepath, 'w', encoding='utf-8') as f:                
                json.dump(data, f,separators=(',\n', ':'),indent = 4)
            data['fence']={}                                                


    return render_template('plotarea.html')



@app.route('/management',methods=[ 'GET','POST'])
def management():
    # nav Replot 
    if request.method == 'POST' :


        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
        
        if (URL == "REPLOT"):

            IMGpath = "static/alias_pict/"+str(alias)+".jpg"
            data["alias"]     = alias
            data["viedo_url"] = URL
            data["add_time"]  = Addtime
                
            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    
                Jdata = json.load(f)
            old_fence_data = Jdata['fence']

            fences = list(old_fence_data.keys())

            vertexs = []

            for fence in fences:
                old_vertex = old_fence_data[fence]['vertex']
                vertexs.append(old_vertex)  

            return render_template('plotarea.html', data = IMGpath, name=str(alias))

        
        if (URL == "Edit"):
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            Group       = request.form.get('Group')
            Alarm_Level = request.form.get('Alarm_Level')
            Note        = request.form.get('Note')
            Sensitivity = request.form.get('Sensitivity') 

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    
                Jdata = json.load(f)

            Jdata['fence'][FenceName]['Group']       = Group
            Jdata['fence'][FenceName]['Alarm_Level'] = Alarm_Level
            Jdata['fence'][FenceName]['Note']        = Note
            Jdata['fence'][FenceName]['Sensitivity'] = Sensitivity

            with open(filepath, 'w', encoding='utf-8') as file:                 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)

            return render_template('management.html', navs=FencesNames)

        if (URL == "Delete"):    
            print("Enter Delete")
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            print("alias : ", alias)

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    
                Jdata = json.load(f)

            Jdata['fence'].pop(FenceName)
            print("\n Jdata :", Jdata, "\n")

            with open(filepath, 'w', encoding='utf-8') as file:                 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)

        
    # management items 
    items=[]
    filepath_list = os.listdir("./static/Json_Info/")
    
    for path in filepath_list:
        if ".ipynb" in path :
            continue
        else:
            path = "./static/Json_Info/" + path
            with open(path, 'r', encoding='utf-8') as f:
                file = json.load(f)
                items.append(file)
    return render_template('management.html', navs=FencesNames, items=items)



@app.route('/streaming',methods=[ 'GET','POST'])
def streaming():
    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
        
        if (URL == "REPLOT"):

            IMGpath = "static/alias_pict/"+str(alias)+".jpg"
            data["alias"]     = alias
            data["viedo_url"] = URL
            data["add_time"]  = Addtime
                
            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                     
                Jdata = json.load(f)
            old_fence_data = Jdata['fence']

            fences = list(old_fence_data.keys())

            vertexs = []

            for fence in fences:
                old_vertex = old_fence_data[fence]['vertex']
                vertexs.append(old_vertex)  

            return render_template('plotarea.html', data = IMGpath, name=str(alias))

    alias_list = os.listdir(r'static/alias_pict')
    alias_list.remove('.ipynb_checkpoints')

    return render_template('streaming.html', navs=FencesNames, alias_list=alias_list, length=len(alias_list))



@app.route('/schedule',methods=[ 'GET','POST'])
def schedule():
    
    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
            
        if (URL == "REPLOT"):

            IMGpath = "static/alias_pict/"+str(alias)+".jpg"
            data["alias"]     = alias
            data["viedo_url"] = URL
            data["add_time"]  = Addtime
  
            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    
                Jdata = json.load(f)
            old_fence_data = Jdata['fence']

            fences = list(old_fence_data.keys())

            vertexs = []

            for fence in fences:
                old_vertex = old_fence_data[fence]['vertex']
                vertexs.append(old_vertex)  

            return render_template('plotarea.html', data = IMGpath, name=str(alias))

        
        if (URL == "Edit_time"):
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            # Group       = request.form.get('Group')
            Order       = str(request.form.get('Order'))
            Start_time  = request.form.get('start_time')
            End_time    = request.form.get('end_time') 

            new_schedule = {'Start_time':Start_time,'End_time':End_time}

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                     
                Jdata = json.load(f)

            Schedule_keys = list(Jdata['fence'][FenceName]['Schedule'].keys())

            if Order in Schedule_keys:
                # Jdata['fence'][FenceName]['Group'] = Group
                Jdata['fence'][FenceName]['Schedule'][Order]['Start_time']  = Start_time
                Jdata['fence'][FenceName]['Schedule'][Order]['End_time']    = End_time
                print("in")
            else:
                # data['fence'][FenceName]['Group'] = Group

                Jdata['fence'][FenceName]['Schedule'][Order] = new_schedule
                print("not in")
                print(Jdata['fence'][FenceName]['Schedule'][Order])

            with open(filepath, 'w', encoding='utf-8') as file:                  
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)


        if (URL == "Delete_Schedule"):
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            Order       = request.form.get('Order')
            
            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    
                Jdata = json.load(f)

            Jdata['fence'][FenceName]['Schedule'][Order]['Start_time']  = "--:--"
            Jdata['fence'][FenceName]['Schedule'][Order]['End_time']    = "--:--"

            with open(filepath, 'w', encoding='utf-8') as file:                 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)
    items=[]
    filepath_list = os.listdir("./static/Json_Info/")
    
    for path in filepath_list:
        if ".ipynb" in path :
            continue
        else:
            path = "./static/Json_Info/" + path
            with open(path, 'r', encoding='utf-8') as f:
                file = json.load(f)
                items.append(file)
  
    return render_template('schedule.html', navs=FencesNames, items=items)

@app.route('/video/<order>',methods=[ 'GET','POST'])
def video_feed(order):
    alias_list = os.listdir(r'static/alias_pict')
    alias_list.remove('.ipynb_checkpoints')

    if len(actived_yolo) > int(order) : 
        print(f"actived_yolo len = {len(actived_yolo)} , order = {order}")
        alias = alias_list[int(order)].replace('.jpg','')
        return Response(gen_frames(actived_yolo[alias]),mimetype='multipart/x-mixed-replace; boundary=frame' )
    else:
        return 'Error'


# @app.route('/test',methods=[ 'GET','POST'])
# def test():
    
#     return render_template('test.html', )

  
if __name__ == '__main__':
    app.run(debug = True, host="0.0.0.0",port="10328")
