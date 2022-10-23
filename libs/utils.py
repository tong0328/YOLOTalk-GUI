import datetime
import os
import shutil
import sys
import json
import cv2
import matplotlib.path as mplPath
import numpy as np
from darknet import darknet
from skimage.metrics import structural_similarity

# import below is jim's YOLOtalk code
import sys
sys.path.append("..") 
from darknet import darknet
from libs.utils import *





def detect_filter(detections, target_classes, vertex):

    results = []
    
    for label, confidence, bbox in detections:        
        left, top, right, bottom = bbox2points(bbox)
        
        # filter the target class
        if target_classes != None:
            if label not in target_classes:
                continue            
        
        # filter the bbox base on the vertex
        if vertex == None:            
            results.append((label, confidence, bbox, None))
        else:
            center_x = (left + right)/2
            center_y = (top + bottom)/2
            if is_in_hull(vertex,(left, top)) or is_in_hull(vertex,(left, bottom))\
                or is_in_hull(vertex,(right, top)) or is_in_hull(vertex,(right, bottom))\
                or is_in_hull(vertex,(center_x, center_y)):          
                results.append((label, confidence, bbox, None))

    return  results
    

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def draw_boxes(detections, image, colors, target_classes, vertex):
    """
    Target will be drawed if vertex is None or target is in vertex
    """
    # dedraw_polylinestections = detect_filter(detections, target_classes, vertex)
    
    for label, confidence, bbox, _ in detections:        
        left, top, right, bottom = bbox2points(bbox)        
        
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
        
    return image


def draw_polylines(img, vertex):
    vertex_key = list(vertex.keys())
    
    for each_vertex_key in vertex_key:
        if vertex[each_vertex_key] != None:
            pts = np.array(vertex[each_vertex_key], np.int32)
            red_color = (0, 0, 255) # BGR
            cv2.polylines(img, [pts], isClosed=True, color=red_color, thickness=3) # BGR
    
    return img


def get_current_date_string():
    now_dt = datetime.datetime.now()
    return "{:04d}-{:02d}-{:02d}".format(now_dt.year, now_dt.month, now_dt.day)


def get_current_hour_string():
    now_dt = datetime.datetime.now()
    return "{:02d}".format(now_dt.hour)


def create_dir(output_dir):
    try:
        path = os.path.join(output_dir)
        os.makedirs(path, exist_ok=True)
        return path
    
    except OSError as e:
        print("Create dirictory error:",e)
        

def del_dir(output_dir, expire_day):    
    for i in range(expire_day, expire_day+10):
        yesterday_dir =  os.path.join(output_dir, ((datetime.datetime.today() - datetime.timedelta(days=i)).strftime("%Y-%m-%d")))
        if os.path.exists(yesterday_dir):
            try:
                shutil.rmtree(yesterday_dir)
                print("[Info] Delete ", yesterday_dir, " successful.")
            except OSError as e:
                pass

                
# Delete the folder all file and create new folder
def re_make_dir(path):    
    try:
        shutil.rmtree(path)        
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
        
    os.makedirs(path)
    
    
def restart():    
    os.execv(sys.executable, ['python'] + sys.argv)


# https://www.tutorialspoint.com/what-s-the-fastest-way-of-checking-if-a-point-is-inside-a-polygon-in-python
# Returns true if the point p lies 
# inside the polygon[] with n vertices
def is_in_hull(vertex:list, p:tuple) -> bool:    
    poly_path = mplPath.Path(np.array(vertex))
    
    return poly_path.contains_point(p)


def bb_intersection_over_union(boxA, boxB):
    
    boxA = [int(x) for x in bbox2points(boxA)]
    boxB = [int(x) for x in bbox2points(boxB)]
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou


# calculate the average value in specific image channel
def avg_color_img(value):
    average_color_row = np.average(value, axis=0) # average each row first
    average_color = np.average(average_color_row, axis=0)
    
    return average_color


def get_img_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    average_color = avg_color_img(v)   

    return average_color


# https://github.com/amusi/Non-Maximum-Suppression/blob/master/nms.py
def nms(all_class_names, detections, detections2, threshold=0.5):
    
    if len(detections) == 0 and len(detections2) == 0:
        return []
    
    for det in detections2:
        detections.append(det)        
    
    nms_detections = []     
    picked_boxes = []
    picked_score = []
    
    for name in all_class_names:
        bounding_boxes  = [] # for return the original results
        boxes = [] # for compute NMS
        confidence_score = []   
        picked_boxes = []
        picked_score = []
        
        for det in detections:
            if det[0] == name:                
                boxes.append(bbox2points(det[2])) 
                bounding_boxes.append(det[2]) 
                confidence_score.append(round(float(det[1])/100, 4))
                
        if len(bounding_boxes) == 0:
            continue
            
        boxes = np.array(boxes)
        score = np.array(confidence_score)
        
        # coordinates of bounding boxes
        start_x = boxes[:, 0]
        start_y = boxes[:, 1]
        end_x = boxes[:, 2]
        end_y = boxes[:, 3]
        
        # Compute areas of bounding boxes
        areas = (end_x - start_x + 1) * (end_y - start_y + 1)

        # Sort by confidence score of bounding boxes
        order = np.argsort(score)

        # Iterate bounding boxes
        while order.size > 0:
            # The index of largest confidence score
            index = order[-1]

            # Pick the bounding box with largest confidence score
            picked_boxes.append(bounding_boxes[index])
            picked_score.append(confidence_score[index])

            # Compute ordinates of intersection-over-union(IOU)
            x1 = np.maximum(start_x[index], start_x[order[:-1]])
            x2 = np.minimum(end_x[index], end_x[order[:-1]])
            y1 = np.maximum(start_y[index], start_y[order[:-1]])
            y2 = np.minimum(end_y[index], end_y[order[:-1]])

            # Compute areas of intersection-over-union
            w = np.maximum(0.0, x2 - x1 + 1)
            h = np.maximum(0.0, y2 - y1 + 1)
            intersection = w * h
            # Compute the ratio between intersection and union
            ratio = intersection / (areas[index] + areas[order[:-1]] - intersection)
            
            left = np.where(ratio < threshold)
            order = order[left]
        
        for i in range(len(picked_boxes)):
            nms_detections.append((name, str(picked_score[i]*100), picked_boxes[i]))
    
    return nms_detections


def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))


# https://stackoverflow.com/questions/56183201/detect-and-visualize-differences-between-two-images-with-opencv-python
def SSIM(bc_img, image):        

    before_gray = cv2.cvtColor(bc_img, cv2.COLOR_BGR2GRAY)
    after_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # compute SSIM between two images
    (score, diff) = structural_similarity(before_gray, after_gray, win_size=25, full=True)
    SSIM_img = (diff * 255).astype("uint8")
    
    return score, SSIM_img
