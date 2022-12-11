from flask import Flask, render_template,Response ,request ,jsonify,redirect, send_file
import time
import numpy as np
import threading
import cv2
import requests
import json
import os
import jinja2

from web_ultis import *
from config import Config
# import below is jim's YOLOtalk code
import sys
sys.path.append("..")
sys.path.insert(1, '../multi-object-tracker') 
from libs.YOLO_SSIM import YoloDevice
from darknet import darknet
from libs.utils import *





app = Flask(__name__)


# Restart YoloDevice
actived_yolo = Restart_YoloDevice()

# Set host
if Config["host"] == "192.168.2.18":
    host = "140.113.131.8"
elif Config["host"] == "0.0.0.0":
    host = "panettone.iottalk.tw"

@app.route('/',methods=[ 'GET','POST'])
def home():
    
    all_fences_names = read_all_fences()

    if request.method == 'POST':

        alias  = request.form.get('area')
        URL     = str(request.form.get('URL'))
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
        postURL = os.path.join("http://" + host  + ":" + Config["port"] + "/plotarea")

        print(f"Home Page")
        print(f"alias :\t{alias}")
        print(f"URL :\t{URL}")
        print(f"Addtime :\t{Addtime}")

        if URL =="REPLOT":

            IMGpath, shape =  replot(alias, URL, Addtime)
            print(f"Plotarea post URL : {postURL}")

            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape, postURL=postURL)

        else:    
            time.sleep(1)   # 避免opencv反應慢
            print("Connecting URL...")
            fig = cv2.VideoCapture(URL)
            stat, I = fig.read()
            fail_number = 0

            if stat == True :
                # Temporary information
                data  = {"alias":"",
                        "viedo_url":"",
                        "add_time":"",
                        "fence": {}
                        }

                data["alias"]     = alias
                data["viedo_url"] = URL
                data["add_time"]  = Addtime
                
                filepath = "static/Json_Info/camera_info_" + data["alias"] + ".json"
                with open(filepath, 'w', encoding='utf-8') as file:             
                    json.dump(data, file,separators=(',\n', ':'),indent = 4)
            
                IMGpath = "static/alias_pict/"+str(alias)+".jpg"
                all_fences_names.append(str(alias))
                cv2.imwrite(IMGpath,I)
                # ======== FOR YOLO ========
                yolo1 = YoloDevice(
                        config_file = '../cfg_person/yolov4-tiny.cfg',
                        data_file = '../cfg_person/coco.data',
                        weights_file = '../weights/yolov4-tiny.weights',
                        thresh = 0.5,                 # need modify (ok)
                        output_dir = './static/record/',              
                        video_url = URL,              # need modify (ok) 
                        is_threading = True,          # rtsp  ->true  video->false (ok)
                        vertex = None,                # need modify (ok)    
                        alias = alias,                  # need modify (ok)
                        display_message = False,
                        obj_trace = True,        
                        save_img = True,
                        save_video = False,           # modify to False
                        target_classes = ["person"],
                        auto_restart = False,
                        )
                print(f"\n======== Activing YOLO , alias:{alias}========\n")    
                yolo1.set_listener(on_data)
                yolo1.start()
                actived_yolo[alias] = yolo1
                # ======== FOR YOLO ========
            
                return render_template('plotarea.html', data = IMGpath, name=str(alias), shape = I.shape, postURL=postURL)

    return render_template('home.html', navs = all_fences_names, alert = False)
    

@app.route('/plotarea',methods=[ 'GET','POST'])
def plotarea():

    all_fences_names = read_all_fences()
    postURL = os.path.join("http://"+ host  + ":" + Config["port"] + "/plotarea")
    print(f"Plotarea post URL : {postURL}")

    if request.method == 'POST':
        
        print("plotarea enter & POST")
        
        alias = request.form['alias']
        FenceName = request.form['FenceName']   # plot name
        vertex = request.form['vertex']         # plot point
        oldName = request.form['oldName']       # oldName
        
        IMGpath = "static/alias_pict/"+str(alias)+".jpg"
        filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
        fig = cv2.imread(IMGpath)
        shape = fig.shape

        Fence_info={'vertex':vertex,
                    'Group':'-',                
                    'Alarm_Level':'General',    # General, High
                    'Note':' - ',           
                    'Sensitivity':'0.5',
                    'Schedule':{
                                '1':{'Start_time':'--:--','End_time':'--:--'},
                               }
                   }
     
        if os.path.isfile(filepath):                                        # If file is exist
            with open(filepath, 'r', encoding='utf-8') as f:                
                Jdata = json.load(f)
            if vertex == "DELETE" :
                Jdata["fence"].pop(FenceName)     
            elif vertex == "Rename" :
                Jdata["fence"][FenceName] = Jdata["fence"][oldName]         # copy
                Jdata["fence"].pop(oldName)                                 # del
            else:
                Jdata["fence"][FenceName]=Fence_info                        
                # ======== FOR YOLO ========
                print(f"\n======== Editing YOLO vertexs, alias:{alias} ,vertex:{actived_yolo[alias].vertex}========\n")
                old_vertex = vertex[1:-1]
                new_vertex = transform_vertex(old_vertex)
                data = { FenceName : new_vertex }
                actived_yolo[alias].vertex = data
                # ======== FOR YOLO ========

            with open(filepath, 'w', encoding='utf-8') as file:             
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)
            
        else:                                                               
            print("\nFile doesn't exist !! \n")
            data["fence"][FenceName]=Fence_info                             
            with open(filepath, 'w', encoding='utf-8') as f:                
                json.dump(data, f,separators=(',\n', ':'),indent = 4)
            data['fence']={}                                                


    return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape, postURL=postURL)


@app.route('/management',methods=[ 'GET','POST'])
def management():

    all_fences_names = read_all_fences()
    postURL = os.path.join("http://"+ host  + ":" + Config["port"] + "/management")
    # nav Replot 
    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]

        if URL =="REPLOT":

            IMGpath, shape =  replot(alias, URL, Addtime)
            postURL = os.path.join("http://" + host  + ":" + Config["port"] + "/plotarea")

            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape, postURL=postURL)

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

            old_Sensitivity = Jdata['fence'][FenceName]['Sensitivity']
            Jdata['fence'][FenceName]['Sensitivity'] = Sensitivity
            print(f"Sensitivity : {Sensitivity}")

            if old_Sensitivity != Sensitivity :
                print(f"actived_yolo[alias].thresh : {actived_yolo[alias].thresh}")
                actived_yolo[alias].thresh = float(Sensitivity)
                print(f"actived_yolo[alias].thresh : {actived_yolo[alias].thresh}")
            with open(filepath, 'w', encoding='utf-8') as file:                 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)

            return render_template('management.html', navs=all_fences_names ,postURL=postURL)

        if (URL == "Delete"):    
            
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
    return render_template('management.html', navs=all_fences_names, items=items, postURL=postURL)


@app.route('/streaming',methods=[ 'GET','POST'])
def streaming():

    all_fences_names = read_all_fences()

    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
        
        if URL =="REPLOT":

            IMGpath, shape =  replot(alias, URL, Addtime)
            postURL = os.path.join("http://"+ host  + ":" + Config["port"] + "/plotarea")

            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape, postURL=postURL)

    alias_list = os.listdir(r'static/alias_pict')

    if ".ipynb_checkpoints" in alias_list :
        alias_list.remove('.ipynb_checkpoints')

    return render_template('streaming.html', navs=all_fences_names, alias_list=alias_list, length=len(alias_list))


@app.route('/schedule',methods=[ 'GET','POST'])
def schedule():
    
    all_fences_names = read_all_fences()
    postURL = os.path.join("http://"+ host  + ":" + Config["port"] + "/schedule")

    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]

        if URL =="REPLOT":

            IMGpath, shape =  replot(alias, URL, Addtime)
            postURL = os.path.join("http://"+ host  + ":" + Config["port"] + "/plotarea")

            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape, postURL=postURL)

        if (URL == "Edit_time"):
            
            alias       = request.form.get('alias')
            FenceName   = request.form.get('FenceName')
            # Group       = request.form.get('Group')
            Order       = str(request.form.get('Order'))
            Start_time  = request.form.get('start_time')
            End_time    = request.form.get('end_time') 
            print(alias, Start_time, End_time)
            new_schedule = {'Start_time':Start_time,'End_time':End_time}

            filepath = "static/Json_Info/camera_info_" + str(alias) + ".json"
            with open(filepath, 'r', encoding='utf-8') as f:                     
                Jdata = json.load(f)

            Schedule_keys = list(Jdata['fence'][FenceName]['Schedule'].keys())

            if Order in Schedule_keys:
                # Jdata['fence'][FenceName]['Group'] = Group
                Jdata['fence'][FenceName]['Schedule'][Order]['Start_time']  = Start_time
                Jdata['fence'][FenceName]['Schedule'][Order]['End_time']    = End_time

            else:
                # data['fence'][FenceName]['Group'] = Group
                Jdata['fence'][FenceName]['Schedule'][Order] = new_schedule

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
  
    return render_template('schedule.html', navs=all_fences_names, items=items, postURL=postURL)


@app.route('/video/<order>',methods=[ 'GET','POST'])
def video_feed(order):
    alias_list = os.listdir(r'static/alias_pict')
    if ".ipynb_checkpoints" in alias_list :
        alias_list.remove('.ipynb_checkpoints')

    if len(actived_yolo) > int(order) : 
        print(f"actived_yolo len = {len(actived_yolo)} , order = {order}")
        alias = alias_list[int(order)].replace('.jpg','')
        return Response(gen_frames(actived_yolo[alias]),mimetype='multipart/x-mixed-replace; boundary=frame' )
    else:
        return 'Error'


@app.route('/files/', defaults={'req_path': ''})
@app.route('/files/<path:req_path>')
def dir_listing(req_path):

    BASE_DIR = './static/record'
    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        print("Error")
    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)
    # Show directory contents
    try:
        files = os.listdir(abs_path)
    except:
        files = os.listdir(BASE_DIR)

    files.sort()
    return render_template('files.html', files=files)

@app.route('/image',methods=[ 'GET','POST'])
def image():
    return render_template('image.html')
if __name__ == '__main__':
    app.debug = True
    app.run(debug=Config["DEBUG"], use_reloader=Config["use_reloader"] , host=Config["host"], port=Config["port"])
    