from flask import Flask, render_template,Response ,request ,jsonify,redirect
import time
import numpy as np
import threading
import cv2
import requests
import json
import os
import jinja2


import sys
sys.path.append("..") 
from libs.YOLO import YoloDevice
from darknet import darknet


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


oldJson = os.listdir(r"static/Json_Info")

actived_yolo = {}
for j in oldJson :
    if "ipynb" in j:
        continue
    print("===========================")
    print(" Alias name is :",j)
    print("===========================")      
    path = os.path.join("static/Json_Info", j)
    with open(path, 'r', encoding='utf-8') as f:             
        Jdata = json.load(f)
        
    alias = Jdata["alias"]
    URL   = Jdata["viedo_url"]
    
# ======== FOR YOLO ========

    yolo1 = YoloDevice(
                    config_file = '../darknet/cfg/yolov4-tiny.cfg',
                    data_file = '../darknet/cfg/coco.data',
                    weights_file = '../weights/yolov4-tiny.weights',
                    thresh = 0.5,                 # need modify (ok)
                    output_dir = './static/record/',              
                    video_url = URL,              # need modify (ok) 
                    is_threading = True,          # rtsp  ->true  video->false (ok)
                    vertex = None,                # need modify (ok)    
                    alias = alias,                  # need modify (ok)
                    display_message = True,
                    obj_trace = True,        
                    save_img = False,
                    save_video = False,           # modify to False
                    save_original_img = False,    # save ori img & txt
                    target_classes = ["person"],
                    auto_restart = True,
                    )    
    yolo1.set_listener(on_data)
    yolo1.start()
    actived_yolo[alias] = yolo1

# ======== FOR YOLO ========
#URL = "rtsp://user:VIDEO2021@140.113.169.201:554/profile1"

def gen_frames(yolo):
    while True:
        frame = yolo.get_current_frame()
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/',methods=[ 'GET','POST'])
def home():

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
                    thresh = 0.5,                 # need modify (ok)
                    output_dir = './static/record/',              
                    video_url = URL,              # need modify (ok) 
                    is_threading = True,          # rtsp  ->true  video->false (ok)
                    vertex = None,                # need modify (ok)    
                    alias = alias,                  # need modify (ok)
                    display_message = True,
                    obj_trace = True,        
                    save_img = False,
                    save_video = False,           # modify to False
                    save_original_img = False,
                    target_classes = ["person"],
                    auto_restart = True,
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
        
        FenceName = request.form['FenceName']   # plot name
        vertex = request.form['vertex']         # plot point
            
        Fence_info={'vertex':vertex,
                    'Group':'-',                
                    'Alarm_Level':'General',    # General, High
                    'Note':' - ',           
                    'Sensitivity':'0.5',
                    'Schedule':{
                                    'First':{'Start_time':'－－：－－','End_time':'－－：－－'},
                                    'Second':{'Start_time':'－－：－－','End_time':'－－：－－'}
                               }
                   }
        # 
        # 
        filepath = "static/Json_Info/camera_info_" + data["alias"] + ".json"
        print("Json File Path : " , filepath)
        if os.path.isfile(filepath):                                        # If file is exist
            with open(filepath, 'r', encoding='utf-8') as f:                # 
                Jdata = json.load(f)

            if vertex == "DELETE" :
                Jdata["fence"].pop(FenceName)                               #
            else:
                Jdata["fence"][FenceName]=Fence_info                        # 

            with open(filepath, 'w', encoding='utf-8') as file:             # 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)
            
        else:                                                               #
            print("\nFile doesn't exist !! \n")
                                  # 
            data["fence"][FenceName]=Fence_info                             # 
            with open(filepath, 'w', encoding='utf-8') as f:                # 
                json.dump(data, f,separators=(',\n', ':'),indent = 4)
            data['fence']={}                                                # 


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
            with open(filepath, 'r', encoding='utf-8') as f:                    # 
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
            with open(filepath, 'r', encoding='utf-8') as f:                    # 
                Jdata = json.load(f)

            Jdata['fence'][FenceName]['Group']       = Group
            Jdata['fence'][FenceName]['Alarm_Level'] = Alarm_Level
            Jdata['fence'][FenceName]['Note']        = Note
            Jdata['fence'][FenceName]['Sensitivity'] = Sensitivity

            with open(filepath, 'w', encoding='utf-8') as file:                 # 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)

            return render_template('management.html', navs=FencesNames)

        if (URL == "Delete"):    
            print("Enter Delete")
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            print("alias : ", alias)

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    # 
                Jdata = json.load(f)

            Jdata['fence'].pop(FenceName)
            print("\n Jdata :", Jdata, "\n")

            with open(filepath, 'w', encoding='utf-8') as file:                 # 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)

#             return render_template('management.html', navs=FencesNames)
        
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
    old_name = "./static/inner.png"
    name = os.listdir(r'static/alias_pict')
    name.remove('.ipynb_checkpoints')
    length = len(name)
    return render_template('streaming.html', navs=FencesNames, length=length)



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
            with open(filepath, 'r', encoding='utf-8') as f:                    # 
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
            Group       = request.form.get('Group')
            Order       = request.form.get('Order')
            Start_time  = request.form.get('start_time')
            End_time    = request.form.get('end_time') 

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    # 
                Jdata = json.load(f)

            Jdata['fence'][FenceName]['Group']       = Group
            Jdata['fence'][FenceName]['Schedule'][Order]['Start_time']  = Start_time
            Jdata['fence'][FenceName]['Schedule'][Order]['End_time']    = End_time

            with open(filepath, 'w', encoding='utf-8') as file:                 # 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)


        if (URL == "Delete_Schedule"):
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            Order       = request.form.get('Order')
            
            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                    # 
                Jdata = json.load(f)

            Jdata['fence'][FenceName]['Schedule'][Order]['Start_time']  = "－－：－－"
            Jdata['fence'][FenceName]['Schedule'][Order]['End_time']    = "－－：－－"

            with open(filepath, 'w', encoding='utf-8') as file:                 # 
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




@app.route('/test',methods=[ 'GET','POST'])
def test():
    
    return render_template('test.html', )


@app.route('/video_feed_0')
def video_feed_0():
    name = os.listdir(r'static/alias_pict')
    name.remove('.ipynb_checkpoints')
    print("            name :",name)
    if len(name) >= 1 : 
        alias = name[0].replace('.jpg','')
        return Response(gen_frames(actived_yolo[alias]),mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_1')
def video_feed_1():
    name = os.listdir(r'static/alias_pict')
    name.remove('.ipynb_checkpoints')
    print("            name :",name)
    if len(name) >= 2 : 
        alias = name[1].replace('.jpg','')
        return Response(gen_frames(actived_yolo[alias]),mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/video_feed_2')
def video_feed_2():
    name = os.listdir(r'static/alias_pict')
    name.remove('.ipynb_checkpoints')
    print("            name :",name)
    if len(name) >= 3 : 
        alias = name[2].replace('.jpg','')
        return Response(gen_frames(actived_yolo[alias]),mimetype='multipart/x-mixed-replace; boundary=frame')
    

@app.route('/video_feed_3')
def video_feed_3():
    name = os.listdir(r'static/alias_pict')
    name.remove('.ipynb_checkpoints')
    print("            name :",name)
    if len(name) >= 4 : 
        alias = name[3].replace('.jpg','')
        return Response(gen_frames(actived_yolo[alias]),mimetype='multipart/x-mixed-replace; boundary=frame')
    

    
if __name__ == '__main__':
    app.run(debug = True, host="0.0.0.0",port="10328")
