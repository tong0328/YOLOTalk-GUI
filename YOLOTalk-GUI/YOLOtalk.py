from flask import Flask, render_template, Response, request, send_file
from web_ultis import Restart_YoloDevice
from config import Config
from web_ultis import *
# import below is jim's YOLOtalk code
import sys
sys.path.append("..")
from libs.YOLO_SSIM import YoloDevice
import time
import cv2
import json
import os

app = Flask(__name__)

# Restart YoloDevice
actived_yolo = Restart_YoloDevice()



@app.route('/',methods=['GET', 'POST'])
def home():
    
    all_fences_names = read_all_fences()

    if request.method == 'POST':
        alias = request.form.get('area')
        URL = str(request.form.get('URL'))
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
        print(f"=== Home Page ===")
        print(f"alias   :\t{alias}")
        print(f"URL     :\t{URL}")
        print(f"Addtime :\t{Addtime}")

        postURL = GET_POST_URL('plotarea')

        if URL =="REPLOT":
            IMGpath, shape =  replot(alias, URL, Addtime)
            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape, postURL=postURL)

        else:    
            # time.sleep(1)   # 避免opencv反應慢
            print("Connecting URL...")
            fig = cv2.VideoCapture(URL)
            stat, I = fig.read()
            print(f"stat : {stat}")
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
                        config_file = '../darknet/cfg/yolov4-tiny.cfg',
                        data_file = '../darknet/cfg/coco.data',
                        weights_file = '../weights/yolov4-tiny.weights',
                        thresh = 0.5,
                        output_dir = './static/record/',
                        video_url = URL,
                        is_threading = True,
                        vertex = None,
                        alias = alias,
                        display_message = False,
                        obj_trace = True,
                        save_img = True,
                        img_expire_day = 3,
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
                return render_template('plotarea.html', data = IMGpath, name=str(alias), shape = I.shape, postURL=postURL)
            
            else:
                return render_template('home.html', navs = all_fences_names, alert = True)
    return render_template('home.html', navs = all_fences_names, alert = False)
    

@app.route('/plotarea',methods=[ 'GET','POST'])
def plotarea():

    all_fences_names = read_all_fences()
    postURL = GET_POST_URL('plotarea')

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
                print(f"Jdata:{Jdata}")
                print(f"actived_yolo:{actived_yolo}")
            if vertex == "DELETE" :
                Jdata["fence"].pop(FenceName)     
            elif vertex == "Rename" :
                Jdata["fence"][FenceName] = Jdata["fence"][oldName]         # copy
                Jdata["fence"].pop(oldName)                                 # del
            else:
                Jdata["fence"][FenceName]=Fence_info                        
                # ======== FOR YOLO ========
                old_vertex = vertex[1:-1]
                new_vertex = transform_vertex(old_vertex)
                data = { FenceName : new_vertex }
                actived_yolo[alias].vertex = data
                print(f"\n======== Editing YOLO vertexs, alias:{alias} ,vertex:{actived_yolo[alias].vertex}========\n")
                # ======== FOR YOLO ========
            print(f"NEW Jdata:{Jdata}")
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
    postURL = GET_POST_URL('management')

    # nav Replot 
    if request.method == 'POST' :

        alias   = request.form.get('area')
        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
        
        if URL =="REPLOT":
            IMGpath, shape =  replot(alias, URL, Addtime)
            postURL = GET_POST_URL('plotarea')
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

            if old_Sensitivity != Sensitivity :
                print(f"actived_yolo[alias].thresh : {actived_yolo[alias].thresh}")
                actived_yolo[alias].thresh = float(Sensitivity)

            with open(filepath, 'w', encoding='utf-8') as file:                 
                json.dump(Jdata, file,separators=(',\n', ':'),indent = 4)

            return render_template('management.html', navs=all_fences_names, postURL=postURL)

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
            postURL = GET_POST_URL('plotarea')
            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape, postURL=postURL)

    alias_list = os.listdir(r'static/alias_pict')
    alias_list.sort()
    if ".ipynb_checkpoints" in alias_list :
        alias_list.remove('.ipynb_checkpoints')

    return render_template('streaming.html', navs=all_fences_names, alias_list=alias_list, length=len(alias_list))


@app.route('/schedule',methods=[ 'GET','POST'])
def schedule():
    
    all_fences_names = read_all_fences()
    postURL = GET_POST_URL('schedule')

    if request.method == 'POST' :

        URL     = request.form.get('URL')        
        Addtime = str(time.asctime( time.localtime(time.time()) ))[4:-5]
            
        if URL =="REPLOT":
            IMGpath, shape =  replot(alias, URL, Addtime)
            postURL = GET_POST_URL('plotarea')
            return render_template('plotarea.html', data=IMGpath, name=str(alias), shape=shape, postURL=postURL)

        if (URL == "Edit_time"):
            alias       = str(request.form.get('alias'))
            FenceName   = request.form.get('FenceName')
            # Group       = request.form.get('Group')
            Order       = str(request.form.get('Order'))
            Start_time  = request.form.get('start_time')
            End_time    = request.form.get('end_time')
            new_schedule = {'Start_time': Start_time, 'End_time': End_time}

            filepath = f"static/Json_Info/camera_info_{alias}.json"
            with open(filepath, 'r', encoding='utf-8') as f:                     
                Jdata = json.load(f)
            Schedule_keys = list(Jdata['fence'][FenceName]['Schedule'].keys())

            if Order in Schedule_keys:
                # Jdata['fence'][FenceName]['Group'] = Group
                Jdata['fence'][FenceName]['Schedule'][Order]['Start_time'] = Start_time
                Jdata['fence'][FenceName]['Schedule'][Order]['End_time'] = End_time
            else:
                # data['fence'][FenceName]['Group'] = Group
                Jdata['fence'][FenceName]['Schedule'][Order] = new_schedule
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(Jdata, file, separators=(',\n', ':'), indent=4)

        if(URL == "Delete_Schedule"):
            alias = request.form.get('alias')
            FenceName = request.form.get('FenceName')
            Order = request.form.get('Order')
            filepath = f"static/Json_Info/camera_info_{alias}.json"
            with open(filepath, 'r', encoding='utf-8') as f:
                Jdata = json.load(f)

            Jdata['fence'][FenceName]['Schedule'][Order]['Start_time'] = "--:--"
            Jdata['fence'][FenceName]['Schedule'][Order]['End_time'] = "--:--"

            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(Jdata, file, separators=(',\n', ':'), indent=4)
    items=[]
    filepath_list = os.listdir("./static/Json_Info/")
    for path in filepath_list:
        if ".ipynb" in path:
            continue
        else:
            path = "./static/Json_Info/" + path
            with open(path, 'r', encoding='utf-8') as f:
                file = json.load(f)
                items.append(file)

    return render_template('schedule.html', navs=all_fences_names, items=items, postURL=postURL)


@app.route('/', defaults={'req_path': ''})
@app.route('/<path:req_path>')
def dir_listing(req_path):
    BASE_DIR = '/home/apple895263/YoloTalk/Flask_web/static'    # The static path under the Flask  
    abs_path = os.path.join(BASE_DIR, req_path)     # Joining the base and the requested path                 
    if not os.path.exists(abs_path):    # Return 404 if path doesn't exist
        print("Error")
    if os.path.isfile(abs_path):    # Check if path is a file and serve
        return send_file(abs_path)
    files = os.listdir(abs_path)    # Show directory contents
    files.sort()
    return render_template('files.html', files=files)

@app.route('/video/<order>', methods=['GET', 'POST'])
def video_feed(order):
    alias_list = os.listdir(r'static/alias_pict')
    alias_list.sort()
    if ".ipynb_checkpoints" in alias_list:
        alias_list.remove('.ipynb_checkpoints')

    if len(actived_yolo) > int(order):
        print(f"actived_yolo len = {len(actived_yolo)}, order = {order}")
        alias = alias_list[int(order)].replace('.jpg', '')
        return Response(gen_frames(actived_yolo[alias]), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return 'Error'


if __name__ == '__main__':
    app.run(debug=Config["DEBUG"], use_reloader=Config["use_reloader"], host=Config["host"], port=Config["port"])
