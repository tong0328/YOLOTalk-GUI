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
from skimage.metrics import structural_similarity

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
    def __init__(self, video_url="", output_dir="", run=True, auto_restart=False, obj_trace=False, catch_miss=False,
                 display_message=True, data_file="", config_file="", weights_file="", is_count_id=False,
                 names_file="", thresh=0.5, vertex=None, target_classes=None, draw_bbox=True, draw_polygon=True,
                 alias="", group="", place="", cam_info="", warning_level=None, is_threading=True, skip_frame=None,
                 schedule=[], save_img=True, save_original_img=False, save_video=False, save_video_original=False, 
                 is_save_after_img=False):
        
        self.video_url = video_url
        self.ourput_dir = output_dir
        self.run = run
        self.auto_restart = auto_restart
        self.display_message = display_message
        self.data_file = data_file
        self.config_file = config_file
        self.weights_file = weights_file
        self.names_file = names_file
        self.thresh = thresh
        self.skip_frame = skip_frame
        self.vertex = vertex # set None to detect the full image
        self.target_classes = target_classes # set None to detect all target
        self.draw_bbox = draw_bbox
        self.draw_polygon = draw_polygon
        self.alias = alias
        self.group = group
        self.place = place
        self.obj_trace = obj_trace
        self.catch_miss = catch_miss
        self.cam_info = cam_info
        self.warning_level = warning_level
        self.is_threading = is_threading # set False if the input is video file
        self.schedule = schedule
        self.save_img = save_img
        self.save_img_original = save_original_img # set True to restore the original image
        self.save_video = save_video # set True to save the result to video
        self.save_video_original = save_video_original # set True to save the video stream
        self.is_save_after_img = is_save_after_img # set True to save a series of images after object is detected
        self.count_after_frame = 0 # count frame for save_after_image()
        self.is_count_id = is_count_id # set True will return the result only the object id is exceed the threshold
        self.count_id = {}
        self.is_return_results = False         
        self.frame_name = None
        
        # diff function
        self.diff_pre_bbox = {"bbox":[], "nums":[]} # store prevuious diff bbox
        self.vote_bc = {"frame":[], "frame_id":[], "candidate_frame":[]} # vote for the current frame as background (beacuse person may in the frame)
        self.is_save_bc = True # save the current image as background           
        self.cnt_diff = 0 # count the different image
        self.cnt_person_frame = 0 # count the frame numbers which have person
        self.output_dir_diff_draw_img = os.path.join(output_dir, alias, "diff_draw_img")
        self.output_dir_diff_original_img = os.path.join(output_dir, alias, "diff_original_img")
#         re_make_dir(self.output_dir_diff_draw_img)
#         re_make_dir(self.output_dir_diff_original_img)
        
        if is_count_id == True:
            self.obj_trace = True
        
        self.detection_listener = None # callback function
        
        # saving path initilize
        self.output_dir_img = os.path.join(output_dir, alias, "img")
        self.output_dir_video = os.path.join(output_dir, alias, "video")
        self.output_dir_img_draw = os.path.join(output_dir, alias, "img_draw")
        self.output_dir_video_draw = os.path.join(output_dir, alias, "video_draw")
        self.output_dir_img_after = os.path.join(output_dir, alias, "img_draw_after")
        
        # Object Tracking
        self.id_storage = [] # save the trace id
        self.tracker_motpy = MultiObjectTracker(
                    dt=1 / 30,
                    tracker_kwargs={'max_staleness': 5},
                    model_spec={'order_pos': 1, 'dim_pos': 2,
                                'order_size': 0, 'dim_size': 2,
                                'q_var_pos': 5000., 'r_var_pos': 0.1},
#                     matching_fn_kwargs={'min_iou': 0.25,
#                                     'multi_match_min_iou': 0.93}
                    )     
        self.tracker = CentroidTracker(max_lost=3, tracker_output_format='mot_challenge')
#         self.tracker = CentroidKF_Tracker(max_lost=3, tracker_output_format='mot_challenge')
#         self.tracker = SORT(max_lost=3, tracker_output_format='mot_challenge', iou_threshold=0.1)
#         self.tracker = IOUTracker(max_lost=3, iou_threshold=0.1, min_detection_confidence=0.4, max_detection_confidence=0.7,
#                          tracker_output_format='mot_challenge')
        self.bbox_colors = {}
        
        # video initilize
        self.frame = None
        self.cap = cv2.VideoCapture(self.video_url)        
        self.ret = False
        self.W = int(self.cap.get(3))
        self.H = int(self.cap.get(4))      
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')              
        self.frame_id = 0
        self.obj_id = None
        self.retry = 0 # reconntecting counts of the video capture
        
        # remove the exist video file
        video_path_original = os.path.join(self.output_dir_video, get_current_date_string(), get_current_hour_string())
        video_path_draw = os.path.join(self.output_dir_video_draw, get_current_date_string(), get_current_hour_string())
        
        if self.alias == "":
            self.video_output_name = 'output_original.avi'
            self.video_output_draw_name = 'output_draw.avi'           
            video_path_original = os.path.join(video_path_original, get_current_hour_string() + ".avi")
            video_path_draw = os.path.join(video_path_draw, get_current_hour_string() + ".avi")
        else:
            self.video_output_name = self.alias + '_output_original.avi'
            self.video_output_draw_name = self.alias + '_output_draw.avi'            
            video_path_original = os.path.join(video_path_original, get_current_hour_string() +"_"+ self.alias + ".avi") 
            video_path_draw = os.path.join(video_path_draw, get_current_hour_string() +"_"+ self.alias + ".avi")
        
        self.frame_original = cv2.VideoWriter(self.video_output_name, self.fourcc, 20.0, (self.W, self.H))
        self.frame_draw = cv2.VideoWriter(self.video_output_draw_name, self.fourcc, 20.0, (self.W, self.H))  
        
        if os.path.exists(video_path_original):
            os.remove(video_path_original)     
            
        if os.path.exists(video_path_original):
            os.remove(video_path_original)     
            
        if os.path.exists(video_path_draw):
            os.remove(video_path_draw)
            
        if os.path.exists(self.video_output_name):
            os.remove(self.video_output_name)            
            
        if os.path.exists(self.video_output_draw_name):
            os.remove(self.video_output_draw_name)
            
        print("[Info] Camera status {alias}:{s}".format(alias=self.alias, s=self.cap.isOpened()))
        
        
    def start(self):
        self.th = []
        self.write_log("[Info] Start the program.")
        
        if self.is_threading:
            self.th.append(threading.Thread(target = self.video_capture))
            self.th.append(threading.Thread(target = self.prediction))
        else:
            self.th.append(threading.Thread(target = self.prediction))
        
        for t in self.th:
            t.start()

    
    def video_capture_wo_threading(self): 
        self.ret, self.frame = self.cap.read() 
            
        if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong
            print("[Info] Video detection is finished...")
            self.stop()            
        else:
            if self.save_video_original:
                self.save_video_frame(self.frame)
    
    
    def video_capture(self):
        t = threading.currentThread()   
        time.sleep(5) # waiting for loading yolo model
        
        while getattr(t, "do_run", True):
            self.ret, self.frame = self.cap.read() 
            
            if not self.cap.isOpened() or not self.ret: # if camera connecting is wrong 
                print("[Error] Reconnecting:{group}:{alias}:{url} ".format(group=self.group, alias=self.alias, url=self.video_url))
                self.retry += 1
                self.cap.release()
                time.sleep(5)
                self.cap = cv2.VideoCapture(self.video_url)
                time.sleep(5)
                
                if self.retry % 3 == 0:
                    self.write_log("[Error] Reconnecting to the camera.")
                    print("[Error] Restarting:{group}:{alias}:{url} ".format(group=self.group, alias=self.alias, url=self.video_url))
                    if self.auto_restart:
                        try:                            
                            self.restart()
                        except Exception as e:
                            self.write_log("[Error] Restart the program failed: {e}".format(e=e))
                            print("[Error] Can not restart:{group}:{alias}:{url} ".format(group=self.group, alias=self.alias, url=self.video_url))
                            time.sleep(10)
            else:
                if self.save_video_original:
                    self.save_video_frame(self.frame)

                    
    def prediction(self):        
        network, class_names, class_colors = darknet.load_network(
            config_file = self.config_file,
            data_file = self.data_file,
            weights = self.weights_file)
        
        last_time = time.time() # to compute the fps
        cnt = 0  # to compute the fps
        predict_time_sum = 0  # to compute the fps              
        t = threading.currentThread() # get this function threading status
        
        # compare two image different        
        bc_img = None # backgrond image        
        self.last_save_bc_time = time.time()
        
        while getattr(t, "do_run", True):            
            cnt+=1
            
            if not self.is_threading:
                self.video_capture_wo_threading()
                
                if self.skip_frame != None and cnt%self.skip_frame != 0:
                    continue
            
            
            if not self.cap.isOpened() or not self.ret:                
                #print("[Info] Waiting for reconnecting...")
                time.sleep(1)
                continue
                
            self.frame_id += 1
            
            frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB) # original image
            darknet_image = darknet.make_image(self.W, self.H, 3)
            darknet.copy_image_from_bytes(darknet_image, frame_rgb.tobytes())
            
            predict_time = time.time() # get start predict time
            self.detections = darknet.detect_image(network, class_names, darknet_image, thresh=self.thresh)
            predict_time_sum +=  (time.time() - predict_time) # add sum predict time
            
#             darknet.print_detections(self.detections, True) # print detection
            darknet.free_image(darknet_image)
    
            # filter the scope and target class   
            self.detect_target = detect_filter(self.detections, self.target_classes, self.vertex)  
            # convert to BGR image
            image = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)            
            
            save_path_img = None
            save_path_img_orig = None
            save_video_draw_path = None
            
            
            # draw the image with object tracking or object detection
            if self.obj_trace: 
                image = self.object_tracker(image)    
            else:
                image = draw_boxes(self.detections, image, class_colors, self.target_classes, self.vertex)
            
            
            if self.draw_polygon: 
                image = draw_polylines(image, self.vertex)  # draw the polygon
            
            
            # save the difference image if occurred
            if self.catch_miss:
                if time.time() - self.last_save_bc_time > 600:
                    self.is_save_bc = True
                
                # if the background should be update and no person in the image 
                if self.is_save_bc and len(self.detections) == 0:
                    self.vote_bc["frame"].append(image)
                    self.vote_bc["frame_id"].append(self.frame_id)                    
                    
                    # if get the five background image
                    if len(self.vote_bc["frame"]) == 7:
                        # check whether have five continous "no-person" frame
                        if max(self.vote_bc["frame_id"]) - min(self.vote_bc["frame_id"]) == 6:
                            self.vote_bc["candidate_frame"].append(self.vote_bc["frame"][2]) # get the middle frame as candidate
                        
                        # check whether the two candidate is different
                        if len(self.vote_bc["candidate_frame"]) == 2:
                            _, score, save_diff = self.diff_img(
                                self.vote_bc["candidate_frame"][0], self.vote_bc["candidate_frame"][1])
                            
                            if save_diff == False:
                                bc_img = self.vote_bc["candidate_frame"][1]
                                self.diff_pre_bbox["bbox"] = [] # reset the bbox record
                                self.diff_pre_bbox["nums"] = [] # reset the bbox record                    
                                self.is_save_bc = False
                                self.last_save_bc_time = time.time()
                                
                            self.vote_bc["candidate_frame"] = []
                            
                        self.vote_bc["frame"] = []
                        self.vote_bc["frame_id"] = []                        
                 
                if self.is_save_bc:
                    # put the info on the image
                    cv2.putText(image, "Waiting for catch background",(50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)                
                else:
                    # if the background was decided      
                    image, score, save_diff = self.diff_img(bc_img, image)       
            
            
            # save draw bbox image
            if self.save_img and len(self.detect_target) > 0:   
                save_path_img = self.save_img_draw(image)           
            else:
                save_path_img_all = self.save_img_all(self.frame)

                
            
            # save oiginal image
            if self.save_img_original and len(self.detect_target) > 0:
                save_path_img_orig = self.save_img_orig(self.frame)
            
            
            # set counter for save after omage
            if self.is_save_after_img and len(self.detect_target) > 0:
                self.count_after_frame = 30   
                
                
            # save after image
            if self.count_after_frame > 0:
                save_path_after_img = self.save_after_img(self.frame)
                self.count_after_frame -= 1               
            
            
            # save video with draw            
            if self.save_video:
                save_video_draw_path = self.save_video_draw(image)
            
            
            # callback function for user            
            if self.is_count_id == True and len(self.count_id) > 0:
                self.is_return_results = False                
                max_id = max(self.count_id, key=self.count_id.get)
                
                if self.count_id[max_id] >= 3 : 
                    self.is_return_results = True                   
                    self.count_id[max_id] = 0
                   
                
            if len(self.detect_target) > 0:
                self.cnt_person_frame += 1
                
                if self.is_count_id == False or \
                    (self.is_count_id == True and self.is_return_results == True):
                    self.__trigger_callback(save_path_img, self.group, self.alias, self.detect_target)                  
            
            
            # Compute FPS
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
            
        self.tracker_motpy.step(detections=[Detection(box=b, score=s, class_id=l) for b, s, l in zip(boxes, scores, class_ids)])
        tracks = self.tracker_motpy.active_tracks(min_steps_alive=3)
        
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
        output_tracks = self.tracker.update(np.array(boxes), np.array(confidence), np.array(class_ids))
        
        self.detect_target = [] # re-assigned each bbox
        for track in output_tracks:
            frame, idx, bb_left, bb_top, bb_width, bb_height, confidence, x, y, z = track
            assert len(track) == 10
#             print(track)
            bbox = (bb_left+bb_width/2, bb_top+bb_height/2,  bb_width, bb_height)
            self.detect_target.append((self.target_classes[0], confidence, bbox, idx)) # put the result to detect_target 
            
            # count id number and determine if post results            
            if self.is_count_id == True:
                if self.count_id.get(idx) == None:
                    self.count_id[idx] = 0
                else:
                    self.count_id[idx] += 1
                
            # assigen each id for a color
            if self.bbox_colors.get(idx) == None:
                self.bbox_colors[idx] = (random.randint(0, 255),
                                        random.randint(0, 255),
                                        random.randint(0, 255))
                
            cv2.rectangle(image, (int(bb_left), int(bb_top)), (int(bb_left+bb_width), int(bb_top+bb_height)), self.bbox_colors[idx], 2) # draw the bbox
            image = draw_tracks(image, output_tracks) # draw the id
            
            # put the score and class to the image
            txt = str(r[0]) + " "+ str(confidence)
            cv2.putText(image, txt, (int(bb_left), int(bb_top-7)) ,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5, color=self.bbox_colors[idx])        
        
        return image
        
    
    def diff_img(self, bc_img, image):        
        diff_bbox = []
        is_diff = False
        save_diff = False       
            
        before_gray = cv2.cvtColor(bc_img, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # compute SSIM between two images
        (score, diff) = structural_similarity(before_gray, after_gray, data_range=1000.0, full=True)   
#         (score, diff) = structural_similarity(before_gray, after_gray, full=True)   
        diff = (diff * 255).astype("uint8")

        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        # iteration all different region
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:
                lx_diff, ly_diff, w_diff, h_diff = cv2.boundingRect(c)
                rx_diff = int(lx_diff + w_diff)
                ry_diff = int(ly_diff + h_diff)
                x_diff = int(lx_diff + w_diff/2)
                y_diff = int(ly_diff + h_diff/2)
                is_diff = True
                
                # delete the unreasonable diff_box area and ratio
                if self.vertex != None and not self.is_save_bc:
                    # check the diff_box whether in the vertex first
                    if not(is_in_hull(self.vertex, (lx_diff, ly_diff)) or is_in_hull(self.vertex, (lx_diff, ry_diff))\
                        or is_in_hull(self.vertex, (rx_diff, ly_diff)) or is_in_hull(self.vertex, (rx_diff, ry_diff))):  
#                     if not(is_in_hull(self.vertex, (x_diff, y_diff))):
                        is_diff = False
                
                # check the diff_box area and ration
                if w_diff * h_diff < 20000 or w_diff * h_diff > 40000 or \
                   h_diff < w_diff * 1.2 or h_diff > w_diff * 5:
                    is_diff = False
                else: 
                    # iteration all yolo detected bbox, then check the bbox iou whether overlapping
                    for det in self.detect_target:
                        x, y, w, h = det[2]
                        iou = bb_intersection_over_union((bbox2points([x, y, w, h])), 
                                                      bbox2points([x_diff, y_diff, w_diff, h_diff]))
                        # if the yolo detected bbox is overlapping the diff_bbox, then neglect
                        if iou > 0.1:
                            is_diff = False
                            break                    
                
                # if the region in the image is different
                if is_diff:
                    # count the diff_box numbers                    
                    if [x_diff, y_diff, w_diff, h_diff] in self.diff_pre_bbox["bbox"]:
                        self.diff_pre_bbox["nums"][self.diff_pre_bbox["bbox"].index([x_diff, y_diff, w_diff, h_diff])] += 1
                    else:
                        self.diff_pre_bbox["bbox"].append([x_diff, y_diff, w_diff, h_diff])
                        self.diff_pre_bbox["nums"].append(0)                    
                    
                    max_id = self.diff_pre_bbox["nums"].index(max(self.diff_pre_bbox["nums"]))
                    
                    # check diff_box whether in the same region (Background may contain the object)
                    if self.diff_pre_bbox["nums"][max_id] >= 2:
                        self.is_save_bc = True
                    else:  
                        # save the diff_bbox                        
                        diff_bbox.append(' '.join([str(0), str(x_diff/self.W), str(y_diff/self.H), 
                                                   str(w_diff/self.W), str(h_diff/self.H)]))
                        cv2.rectangle(image, (lx_diff, ly_diff), (lx_diff + w_diff, ly_diff + h_diff),
                                    (255,0,255), 4)                        
                        save_diff = True        
                        self.cnt_diff += 1
        
        # if found the diff image
        if save_diff and not self.is_save_bc:
            # save the yolo detect_target bbox
            for det in self.detections:
                x, y, w, h = det[2]
                diff_bbox.append(' '.join([str(0), str(x/self.W), str(y/self.H), 
                                       str(w/self.W), str(h/self.H)]))
            with open(os.path.join(self.output_dir_diff_original_img, str(self.frame_id) + ".txt"), 'w') as f:
                f.write('\n'.join(diff_bbox))
                
            cv2.imwrite(os.path.join(self.output_dir_diff_original_img, str(self.frame_id) + ".jpg"), self.frame)    
            cv2.imwrite(os.path.join(self.output_dir_diff_draw_img, str(self.frame_id) + ".jpg"), image)
            
        cv2.putText(image, str(self.cnt_diff) + "/" + str(self.cnt_person_frame + self.cnt_diff), 
                    (self.W-200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        
        return image, score, save_diff
        
        
    def set_listener(self, on_detection):
        self.detection_listener = on_detection
        
        
    def __trigger_callback(self, save_path_img, group, alias, detect):        
        if self.detection_listener is None:
            return         
        self.detection_listener(save_path_img, group, alias, detect)
  

    def get_current_frame(self):
        return self.frame

        
    def stop(self):        
        for t in self.th:
            t.do_run = False
#             t.join()
            
        self.write_log("[Info] Stop the program.")
        self.cap.release()
        
        print('[Info] Stop the program: Group:{group}, alias:{alias}, URL:{url}'\
              .format(group=self.group, alias=self.alias, url=self.video_url))      
        
        
    def restart(self):
        self.stop()        
        self.write_log("[Info] Restart the program")
        restart()
        
        
    def write_log(self, msg):     
        f= open('log.txt', "a")    
        f.write("{msg}, Time:{time}, Group:{group}, alias:{alias}, URL:{url} \n"\
                .format(time=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), \
                 group=self.group, alias=self.alias, url=self.video_url, msg=msg))
        f.close()
       
    
    def print_msg(self, msg):
        if self.display_message == True:            
            print(msg)
            
            
    def save_video_frame(self, frame):        
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video, get_current_date_string(), 
                        get_current_hour_string())
            if self.alias == "":                
                video_path_name = os.path.join(video_path, get_current_hour_string() + ".avi")                
            else:
                video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".avi")   
            
        else: # video input, so output wihtout time directory
            video_path_name = self.video_output_name
        
        if not os.path.exists(video_path_name):
            if self.is_threading: create_dir(video_path)                        
            self.frame_original = cv2.VideoWriter(video_path_name, self.fourcc, 30.0, (self.W, self.H))
            print("[Info] {alias} Set video frame writer. Width={W}, Height={H} ".format(alias=self.alias, W=self.W, H=self.H))
            
        self.frame_original.write(frame)
        
        return video_path_name
    
    
    def save_video_draw(self, frame):        
        if self.is_threading:
            video_path = os.path.join(self.output_dir_video_draw, get_current_date_string(), 
                        get_current_hour_string())
            if self.alias == "":                
                video_path_name = os.path.join(video_path, get_current_hour_string() + ".avi")                
            else:
                video_path_name = os.path.join(video_path, get_current_hour_string() +"_"+ self.alias + ".avi")   
            
        else: # video input, so output wihtout time directory
            video_path_name = self.video_output_draw_name            
            
        if not os.path.exists(video_path_name):
            if self.is_threading: create_dir(video_path)
            self.frame_draw = cv2.VideoWriter(video_path_name, self.fourcc, 20.0, (self.W, self.H))
            print("[Info] {alias} Set video draw writer. Width={W}, Height={H} ".format(alias=self.alias, W=self.W, H=self.H))
            
        self.frame_draw.write(frame)
        
        return video_path_name
    
    def save_img_all(self, image):
        # if input is video, dir will not separate into the time directory
        if not self.is_threading:
            img_path = self.output_dir_img_draw
        else:
            img_path = os.path.join(self.output_dir_img_draw, get_current_date_string(), get_current_hour_string())     
        if not os.path.exists(img_path):
            create_dir(img_path)
            
        ftime = time.strftime("%Y-%m-%d_%H-%M-%S-", time.localtime())
        mSec = str(time.time())[11:13]
        ftime = ftime + mSec
        
        if self.alias == "":
            self.frame_name = os.path.join(img_path, ftime + ".jpg")
        else:
            self.frame_name = os.path.join(img_path, ftime + ".jpg")    
                  
        cv2.imwrite(self.frame_name, image)
        
        return self.frame_name
    
    
    def save_img_draw(self, image):              
        # if input is video, dir will not separate into the time directory
        if not self.is_threading:
            img_path = self.output_dir_img_draw
        else:
            img_path = os.path.join(self.output_dir_img_draw, get_current_date_string(), get_current_hour_string())     
        
        if not os.path.exists(img_path):
            create_dir(img_path)      
        
        ftime = time.strftime("%Y-%m-%d_%H-%M-%S-", time.localtime())
        mSec = str(time.time())[11:13]
        ftime = ftime + mSec

        if self.alias == "":
            img_path_name = os.path.join(img_path, ftime + ".jpg")
        else:
            img_path_name = os.path.join(img_path, ftime + ".jpg")               
                  
        cv2.imwrite(img_path_name, image)
        
        return img_path_name
    
    
    def save_img_orig(self, image):        
        # if input is video, dir will not separate into the time directory
        if not self.is_threading:
            img_path = self.output_dir_img
        else:
            img_path = os.path.join(self.output_dir_img, get_current_date_string(), get_current_hour_string())
        
        if not os.path.exists(img_path):
            create_dir(img_path)            
            
        if self.alias == "":
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) + ".jpg")
            txt_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) + ".txt")
        else:
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) +"_"+ self.alias + ".jpg")            
            txt_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) +"_"+ self.alias + ".txt")            
                
        cv2.imwrite(img_path_name, image)
        
        # save class bbox to txt file
        txt_list = []
        f= open(txt_path_name, "w")
        
        for bbox in self.detect_target:
            objclass = self.target_classes.index(bbox[0])
            x = bbox[2][0] / self.W
            y = bbox[2][1] / self.H
            w = bbox[2][2] / self.W
            h = bbox[2][3] / self.H
            txt_list.append(' '.join([str(objclass), str(x), str(y), str(w), str(h)]))
        f.write('\n'.join(txt_list))
        f.close()
        
        return img_path_name
    
    
    def save_after_img(self, image):        
        # if input is video, dir will not separate into the time directory
        if not self.is_threading:
            img_path = self.output_dir_img_after
        else:
            img_path = os.path.join(self.output_dir_img_after, get_current_date_string(), get_current_hour_string())
        
        if not os.path.exists(img_path):
            create_dir(img_path)            
            
        if self.alias == "":
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) + ".jpg")            
        else:
            img_path_name = os.path.join(img_path, str(self.frame_id).zfill(5) +"_"+ self.alias + ".jpg") 
                
        cv2.imwrite(img_path_name, image)
        
        return img_path_name
