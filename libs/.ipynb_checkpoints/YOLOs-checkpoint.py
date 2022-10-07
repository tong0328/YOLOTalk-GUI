import cv2
import numpy as np
import datetime
import time
import os
import shutil
import threading
import sys
import ctypes
import  random

# motpy
from motpy.testing_viz import draw_detection, draw_track
from motpy import Detection, MultiObjectTracker

# multiobjecttracker
sys.path.insert(1, 'multi-object-tracker')
from motrackers import CentroidTracker, CentroidKF_Tracker, SORT, IOUTracker
from motrackers.utils import draw_tracks

# SmartFence  lib
from libs.utils import *

from darknet import darknet

class YoloDevice:
    def __init__(self, video_url="", video_url_list=[], output_dir="", run=True, auto_restart=False, obj_trace=False,
                 display_message=True, data_file="", config_file="", weights_file="", is_count_id=False,
                 names_file="", thresh=0.5, vertex=None, vertex_list=[], target_classes=None, draw_polygon=True,
                 alias="demo", group="", place="", cam_info="", warning_level=None, is_threading=True, skip_frame=None,
                 schedule=[], save_img=True, save_original_img=False, save_video=False, save_video_original=False, 
                 is_save_after_detect=False):
        
        self.video_url = video_url
        self.video_url_list = video_url_list # multi-camera url input, format: [[],[],...]
        self.vertex = vertex
        self.vertex_list = vertex_list # set for the multi-camera url input, format: [[],[],...]
        self.ourput_dir = output_dir # output results directory
        self.run = run 
        self.auto_restart = auto_restart # restart the program if error occure
        self.display_message = display_message
        self.data_file = data_file
        self.config_file = config_file # yolo config file
        self.weights_file = weights_file # yolo weights file
        self.names_file = names_file
        self.thresh = thresh # detect ooject thresold
        self.skip_frame = skip_frame # set for the developer testing        
        self.target_classes = target_classes # set None to detect all target
        self.draw_polygon = draw_polygon
        self.alias = alias
        self.group = group
        self.place = place
        self.obj_trace = obj_trace # set True to start object tracking
        self.cam_info = cam_info
        self.warning_level = warning_level
        self.is_threading = is_threading # set False if the input is video file
        self.schedule = schedule
        self.save_img = save_img # set True to restore the detection image
        self.save_img_original = save_original_img # set True to restore the original image
        self.save_video = save_video # set True to save the result to video
        self.save_video_original = save_video_original # set True to save the video stream        
        self.detection_listener = None # callback function        
        self.url_id_queue = [] # the queue store the camera for detection
        self.url_id = 0 # the id of the camera url        
        
        if self.video_url != "" and len(self.video_url_list) > 0:
            raise AssertionError("Only one of 'video_url' or 'video_url_list' arguments can be set.")  
            
        if self.video_url != "":
            self.video_url_list.append(self.video_url)
        
        if len(self.vertex_list) == 0:
            self.vertex_list.append(self.vertex)
            
        if len(self.video_url_list) > 1 and self.is_threading == False:
            raise AssertionError("'is_threading' argument shoud be set 'True' if the input arguments is 'video_url_list'.") 
            
        if self.video_url == "" and len(self.video_url_list) == 0:
            raise AssertionError("Please input the video url")
            
        if len(self.video_url_list) != len(self.vertex_list):
            raise AssertionError("The length of video url list should be same as the vertex list.")        
        
        # save the image after the object was detected (Sentry Mode)
        self.is_save_after_detect = is_save_after_detect # set True to save a series of images after object is detected
        self.count_after_frame = [] # count frame numbers when object was detected. For the save_after_image()
        
        # return the result only when the object id is reach the threshold
        self.is_count_id = is_count_id # set True will return the result only the object id is reach the threshold
        self.count_id = {} # count for the each tracking object detected numbers
        self.is_return_results = [] # return the detection results if count_id reach threshold
        
        # self.obj_trace should be set True if the is_count_id is True
        if is_count_id == True:
            self.obj_trace = True
            
        # object tracking
        '''The object id of object_track_motpy() is a garbled. 
        So the garbled will be saved to a 'self.id_storage' list,
        and return the index of the list as the object id'''
        self.bbox_colors = {} # set color for each object id
        self.tracker_motpy = []
        self.tracker = []  
        self.id_storage = [] # only used when motpy is called
               
        # saving results path initalize
        self.output_dir_img = []
        self.output_dir_video = []
        self.output_dir_img_draw = []
        self.output_dir_video_draw = []
        self.output_dir_img_after = []
        
        # Video initalize
        self.frame = []
        self.cap = [] 
        self.ret = []
        self.H = []
        self.W = []
        self.fourcc = []
        self.frame_original = [] # cv2.VideoWriter
        self.frame_draw = [] # cv2.VideoWriter        
        self.frame_id = [] # to name for each image
        self.retry = [] # count reconntecting camera numbers
        self.video_output_name = self.alias + '_output_original.avi'
        self.video_output_draw_name = self.alias + '_output_draw.avi' 

        for i in range(len(self.video_url_list)):
            self.output_dir_img.append(os.path.join(output_dir, alias, "img_" + str(i)))
            self.output_dir_video.append(os.path.join(output_dir, alias, "video_" + str(i)))
            self.output_dir_img_draw.append(os.path.join(output_dir, alias, "img_draw_" + str(i)))
            self.output_dir_video_draw.append(os.path.join(output_dir, alias, "video_draw_" + str(i)))
            self.output_dir_img_after.append(os.path.join(output_dir, alias, "img_draw_after_" + str(i)))                
            self.count_id[i] =  {} # count_id formate: {{},{},...}
            self.count_after_frame.append(0)
            self.retry.append(0)
            self.is_return_results.append(False)
            self.frame.append(None)
            self.frame_id.append(0)
            self.cap.append(cv2.VideoCapture(self.video_url_list[i]))
            self.ret.append(False)
            self.W.append(int(self.cap[i].get(3)))
            self.H.append(int(self.cap[i].get(4)))
            self.fourcc.append(cv2.VideoWriter_fourcc(*'XVID'))
            self.frame_original.append(cv2.VideoWriter(self.video_output_name, self.fourcc[i], 20.0, (self.W[i], self.H[i])))
            self.frame_draw.append(cv2.VideoWriter(self.video_output_draw_name, self.fourcc[i], 20.0, (self.W[i], self.H[i])))
            
            # object tracker initalize
            self.tracker_motpy.append(MultiObjectTracker(
            dt=1 / 30,
            tracker_kwargs={'max_staleness': 5},
            model_spec={'order_pos': 1, 'dim_pos': 2,
                        'order_size': 0, 'dim_size': 2,
                        'q_var_pos': 5000., 'r_var_pos': 0.1},
#                      matching_fn_kwargs={'min_iou': 0.25,
#                                      'multi_match_min_iou': 0.93}
            ))
            self.tracker.append(CentroidTracker(max_lost=3, tracker_output_format='mot_challenge'))
    #         self.tracker.append(CentroidKF_Tracker(max_lost=3, tracker_output_format='mot_challenge'))
    #         self.tracker.append(SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.1))
    #         self.tracker.append(IOUTracker(max_lost=3, iou_threshold=0.1, min_detection_confidence=0.4, max_detection_confidence=0.7,
    #                          tracker_output_format='mot_challenge'))
            
            
            
        '''
        When the user restart the program,
        the old exist video file should be removed.        
        '''
        if os.path.exists(self.video_output_name):
            os.remove(self.video_output_name)            

        if os.path.exists(self.video_output_draw_name):
            os.remove(self.video_output_draw_name)
            
        if self.is_threading:
            for i in range(len(self.output_dir_video)):
                video_path_original = os.path.join(self.output_dir_video[i], get_current_date_string(),get_current_hour_string(),
                                          get_current_hour_string() +"_"+ self.alias + ".avi")
                
                video_path_draw = os.path.join(self.output_dir_video_draw[i], get_current_date_string(), get_current_hour_string(),
                                      get_current_hour_string() +"_"+ self.alias + ".avi")    
                
                if os.path.exists(video_path_original):
                    os.remove(video_path_original) 

                if os.path.exists(video_path_draw):
                    os.remove(video_path_draw)
        
        
            
            # print camera connection status
        
#         for i in range(len(yolo1.video_url_list)):            
#             print("[Info] {url}. Camera status {alias}: {s}".format(
#                 url=self.video_url_list[i],alias=self.alias, s=self.cap[i].isOpened()))
            
            
    def start(self):
        self.th = []
        self.write_log("[Info] Start the program.")
        
        if self.is_threading:
            for i in range(len(self.video_url_list)):
                self.th.append(threading.Thread(target = self.video_capture, args=(i,)))                
            self.th.append(threading.Thread(target = self.prediction))            
        else:
            self.th.append(threading.Thread(target = self.prediction))
        
        for t in self.th:
            t.start()

    
    def video_capture_wo_threading(self): 
        self.ret[0], self.frame[0] = self.cap[0].read() 
            
        if not self.cap[0].isOpened() or not self.ret[0]: # if camera connecting is wrong
            print("[Info] Video detection is finished...")
            self.stop()            
        else:
            if self.save_video_original:
                self.save_video_frame(self.frame[0], 0)
    
    
    def video_capture(self, i):
        t = threading.currentThread()
        time.sleep(5) # waiting for loading yolo model
        
        while getattr(t, "do_run", True):
            self.ret[i], self.frame[i] = self.cap[i].read() 
            
            if not self.cap[i].isOpened() or not self.ret[i]: # if camera connecting is wrong 
                print("[Error] Reconnecting:{group}:{alias}:{url} ".format(
                    group=self.group, alias=self.alias, url=self.video_url_list[i]))
                self.retry[i] += 1
                self.cap[i].release()
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.video_url_list[i])
                time.sleep(5)
                
                if self.retry % 3 == 0:
                    self.write_log("[Error] Reconnecting to the camera.")
                    print("[Error] Restarting:{group}:{alias}:{url} "\
                          .format(group=self.group,alias=self.alias, url=self.video_url_list[i]))
                    if self.auto_restart:
                        try:                            
                            self.restart()
                        except Exception as e:
                            self.write_log("[Error] Restart the program failed: {e}".format(e=e))
                            print("[Error] Can not restart:{group}:{alias}:{url} "\
                                  .format(group=self.group, alias=self.alias, url=self.video_url_list[i]))
                            time.sleep(10)
            else:
                self.retry[i] = 0
                if self.save_video_original:
                    self.save_video_frame(self.frame[i], i)

                    
    def prediction(self):        
        network, class_names, class_colors = darknet.load_network(
            config_file = self.config_file,
            data_file = self.data_file,
            weights = self.weights_file)
        
        last_time = time.time() # to compute the fps
        cnt = 0  # to compute the fps
        connecting_flag = False # check all camera is connecting or not
        predict_time_sum = 0  # to compute the fps        
        t = threading.currentThread() # get this function threading status
        
        # waiting for all camera is connecting        
        while(connecting_flag == False):
            connecting_flag = True
            
            for i in range(len(self.video_url_list)):
                if self.ret[i] == False:
                    connecting_flag = False
                    print("[Info] Waiting:{group}:{alias}:{url} "\
                          .format(group=self.group,alias=self.alias, url=self.video_url_list[i]))
                    time.sleep(1)
                    break
        
        while getattr(t, "do_run", True):
            
            if not self.is_threading:
                self.video_capture_wo_threading()
                
                if self.skip_frame != None and self.frame_id[sel.url_id]%self.skip_frame != 0:
                    continue
                    
            self.url_id = random.randint(0,len(self.video_url_list)-1)   
#             print(self.url_id)
            if not self.cap[self.url_id].isOpened() or not self.ret[self.url_id]:                
                time.sleep(1)
                continue
            
            self.frame_id[self.url_id] += 1
            cnt+=1 
            
            frame_rgb = cv2.cvtColor(self.frame[self.url_id], cv2.COLOR_BGR2RGB) # original image
            darknet_image = darknet.make_image(self.H[self.url_id], self.W[self.url_id], 3)
            darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
            
            predict_time = time.time() # get start predict time
            detections = darknet.detect_image(network, class_names, darknet_image, thresh=self.thresh)
            predict_time_sum +=  (time.time() - predict_time) # add sum predict time
            
#             darknet.print_detections(detections, True) # print detection
            darknet.free_image(darknet_image)
    
            # filter the scope and target class   
            self.detect_target = detect_filter(detections, self.target_classes, self.vertex_list[self.url_id])  
            # convert to BGR image
            image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)            
            
            save_path_img = None
            save_path_img_orig = None
            save_video_draw_path = None
            
            # draw the image with object tracking or object detection
            if self.obj_trace: 
                image = self.object_tracker(image)    
            else:
                image = draw_boxes(detections, image, class_colors, self.target_classes, self.vertex_list[self.url_id])
            
            if self.draw_polygon: 
                image = draw_polylines(image, self.vertex)  # draw the polygon
            
            # save draw bbox image
            if self.save_img and len(self.detect_target) > 0:                 
                save_path_img = self.save_img_draw(image)            
            
            # save oiginal image
            if self.save_img_original and len(self.detect_target) > 0:
                save_path_img_orig = self.save_img_orig(self.frame[self.url_id])
            
            # set counter for save_after_img()
            if self.is_save_after_detect and len(self.detect_target) > 0:
                self.count_after_frame[self.url_id] = 30   
                
            # save image after object was detected
            for i in range(len(self.video_url_list)):
                if self.count_after_frame[i] > 0:
                    save_path_after_img = self.save_after_img(self.frame[i], i)
                    self.count_after_frame[i] -= 1  
            
            # save video with draw            
            if self.save_video:
                save_video_draw_path = self.save_video_draw(image)
            
            # compute the each object id detected numbers
            if self.is_count_id == True and len(self.count_id[self.url_id]) > 0:
                self.is_return_results[self.url_id] = False                
                max_id = max(self.count_id[self.url_id], key=self.count_id[self.url_id].get)
                
                if self.count_id[self.url_id][max_id] >= 3 : 
                    self.is_return_results = True                   
                    self.count_id[self.url_id][max_id] = 0
            
            # callback function for user       
            #  return the result only when the object id is reach the threshold
            if len(self.detect_target) > 0:                    
                if self.is_count_id == False or \
                    (self.is_count_id == True and self.is_return_results[url_id] == True):
                    self.__trigger_callback(save_path_img, self.group, self.alias, self.url_id, self.detect_target)   
                  
            
            # compute FPS
            if time.time() - last_time > 5:
                self.print_msg("[Info] FPS:{fps} , {alias}".format(alias=self.alias, fps=cnt / (time.time()-last_time)))
                self.print_msg("[Info] Predict FPS:{fps} , {alias}".format(alias=self.alias, fps=cnt / predict_time_sum))
                last_time = time.time()
                cnt = 0
                predict_time_sum = 0 
    
    
    # https://github.com/wmuron/motpy.git
    def object_tracker_motpy(self, image):           
        boxes = []
        scores = []
        class_ids = []
        
        # convert to the trace format
        for r in self.detect_target:
            boxes.append( darknet.bbox2points(r[2]) )
            scores.append( float(r[1]) )
            class_ids.append(r[0])        
            
        self.tracker_motpy[self.url_id].step(detections=[Detection(box=b, score=s, class_id=l) for b, s, l in zip(boxes, scores, class_ids)])
        tracks = self.tracker_motpy[self.url_id].active_tracks(min_steps_alive=3)
        
        self.detect_target = [] # re-assigned each bbox
        for track in tracks:
            # append the track.id to id_storage
            if track.id not in self.id_storage:
                self.id_storage.append(track.id)
                
            id_index = self.id_storage.index(track.id) #  the order of elements in the python list is persistent                
            self.detect_target.append((track.class_id, track.score, track.box, id_index)) # put the result to detect_target            
            draw_track(image, track, thickness=2, text_at_bottom=True, text_verbose=0) # draw the bbox
            
             # put the id to the image
            txt = track.class_id + " "+ str(track.score) +" ID=" + str(id_index)
            cv2.putText(image, txt, (int(track.box[0]), int(track.box[1])-7) ,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=(255,255,0))
            
        return image
    
    
    # https://github.com/adipandas/multi-object-tracker.git
    def object_tracker(self, image):
        boxes = []
        confidence = []
        class_ids = []
        
        # convert to the trace format
        for r in self.detect_target:
            center_x, center_y, width, height = r[2]
            left, top, right, bottom = darknet.bbox2points(r[2])
            boxes.append([left, top, width, height])
            confidence.append(int(float(r[1])))
            class_ids.append(int(self.target_classes.index(r[0])))
#             cv2.rectangle(image, (int(left), int(top)), (int(left+width), int(top+height)), (0,0,255), 2) # draw the bbox   
        output_tracks = self.tracker[self.url_id].update(np.array(boxes), np.array(confidence), np.array(class_ids))
        
        self.detect_target = [] # re-assigned each bbox
        for track in output_tracks:
            frame, idx, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
            assert len(track) == 10
#             print(track)
            bbox = (bb_left+bb_width/2, bb_top+bb_height/2,  bb_width, bb_height)
            self.detect_target.append((self.target_classes[0], confidence, bbox, idx)) # put the result to detect_target 
            
            # count id number and determine if post results            
            if self.is_count_id == True:
                if self.count_id[self.url_id].get(idx) == None:
                    self.count_id[self.url_id][idx] = 0
                else:
                    self.count_id[self.url_id][idx] += 1
                
            # assigen each id for a color
            if self.bbox_colors.get(idx) == None:
                self.bbox_colors[idx] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
                
            cv2.rectangle(image, (int(bb_left), int(bb_top)), 
                          (int(bb_left+bb_width), int(bb_top+bb_height)), self.bbox_colors[idx], 2) 
            image = draw_tracks(image, output_tracks) # draw the id
            
            # put the score and class to the image
            txt = str(r[0]) + " "+ str(confidence)
            cv2.putText(image, txt, (int(bb_left), int(bb_top-7)) ,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=self.bbox_colors[idx])        
        
        return image
        
        
    def set_listener(self, on_detection):
        self.detection_listener = on_detection
        
        
    def __trigger_callback(self, save_path_img, group, alias, video_url, detect):        
        if self.detection_listener is None:
            return
        
        self.detection_listener(save_path_img, group, alias, detect)
  

    def get_current_frame(self, i):        
        return self.frame[i]
        
        
    def stop(self):        
        for t in self.th:
            t.do_run = False
#             t.join()
            
        self.write_log("[Info] Stop the program.")
        
        for i in range(len(self.video_url_list)):
            self.cap[i].release()
        
            print('[Info] Stop the program: Group:{group}, alias:{alias}, URL:{url}'\
                  .format(group=self.group, alias=self.alias, url=self.video_url_list[i]))  
        
        
    def restart(self):
        self.stop()        
        self.write_log("[Info] Restart the program")
        restart()
        
        
    def write_log(self, msg):     
        f= open('log.txt', "a")    
        f.write("{msg}, Time:{time}, Group:{group}, alias:{alias}, URL:{url} \n".format(
            time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            group=self.group, alias=self.alias, url=self.video_url_list[0], msg=msg))
        f.close()
       
    
    def print_msg(self, msg):
        if self.display_message == True:            
            print(msg)
            
            
    '''
    Since the save_video_frame() fun using thread to save the video frame,
    so the input have 'i' parameter
    '''
    def save_video_frame(self, frame, i):
        
        # if video input, then output directory wihtout time name
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video[i], get_current_date_string(), 
                        get_current_hour_string())      
            video_path_name = os.path.join(video_path, get_current_hour_string() + ".avi")
        else:
            video_path_name = self.video_output_name
        
        if not os.path.exists(video_path_name):
            if self.is_threading: create_dir(video_path)
                
            self.frame_original[i] = cv2.VideoWriter(video_path_name, 
                self.fourcc[i], 30.0, (self.W[i], self.H[i]))
            print("[Info] {alias} Set video frame writer. Height={H}, Width={W}".format(alias=self.alias, 
                H=self.H[i], W=self.W[i]))
            
        self.frame_original[i].write(frame)
        
        return video_path_name
    
    
    def save_video_draw(self, frame):
        
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video_draw[self.url_id], get_current_date_string(), 
                                      get_current_hour_string())
            video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".avi")   
            
        else: # video input, so output wihtout time directory
            video_path_name = self.video_output_draw_name       
            
        if not os.path.exists(video_path_name):
            if self.is_threading: create_dir(video_path)
                
            self.frame_draw[self.url_id] = cv2.VideoWriter(video_path_name, self.fourcc[self.url_id], 20.0, 
                                              (self.W[self.url_id], self.H[self.url_id]))
            print("[Info] {alias} Set video draw writer. Height={H}, Width={W}".format(
                alias=self.alias, H=self.H[self.url_id], W=self.W[self.url_id]))  
            
        self.frame_draw[self.url_id].write(frame)
        
        return video_path_name
    
    
    def save_img_draw(self, image):      
        
        # if input is video, dir will not separate into the time directory
        if self.is_threading:
            img_path = os.path.join(self.output_dir_img_draw[self.url_id], get_current_date_string(), 
                                    get_current_hour_string())                 
        else:
            img_path = self.output_dir_img_draw[self.url_id]
        
        if not os.path.exists(img_path):
            create_dir(img_path)   
        
        img_path_name = os.path.join(img_path, str(self.frame_id[self.url_id]).zfill(5) +"_"+ self.alias + ".jpg")                  
        cv2.imwrite(img_path_name, image)
        
        return img_path_name
    
    
    def save_img_orig(self, image):
        
        # if input is video, dir will not separate into the time directory
        if self.is_threading:
            img_path = os.path.join(self.output_dir_img[self.url_id], get_current_date_string(), 
                                    get_current_hour_string())            
        else:
            img_path = self.output_dir_img[self.url_id]
        
        if not os.path.exists(img_path):
            create_dir(img_path) 
            
        img_path_name = os.path.join(img_path, str(self.frame_id[self.url_id]).zfill(5) +"_"+ self.alias + ".jpg")            
        txt_path_name = os.path.join(img_path, str(self.frame_id[self.url_id]).zfill(5) +"_"+ self.alias + ".txt")                
        cv2.imwrite(img_path_name, image)
        
        # save class bbox to txt file
        f= open(txt_path_name, "a")    
        f.write(str(self.detect_target))
        f.close()
        
        return img_path_name
    
    '''
    Since the save_after_img() fun may have to save over one camera img at the same time,
    so the input have 'i' parameter
    '''
    def save_after_img(self, image, i):
        
        # if input is video, dir will not separate into the time directory
        if self.is_threading:
            img_path = os.path.join(self.output_dir_img_after[i], get_current_date_string(), 
                                    get_current_hour_string())            
        else:
            img_path = self.output_dir_img_after[i]
        
        if not os.path.exists(img_path):
            create_dir(img_path) 
        
        img_path_name = os.path.join(img_path, str(self.frame_id[i]).zfill(5) +"_"+ self.alias + ".jpg")                
        cv2.imwrite(img_path_name, image)
        
        return img_path_name
