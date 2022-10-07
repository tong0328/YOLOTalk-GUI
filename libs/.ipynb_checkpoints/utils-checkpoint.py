import cv2
import numpy as np
import datetime
import os
import shutil
import sys
import matplotlib.path as mplPath

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
        elif is_in_hull(vertex,(left, top)) or is_in_hull(vertex,(left, bottom))\
            or is_in_hull(vertex,(right, top)) or is_in_hull(vertex,(right, bottom)):          
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
    Target will be drawed if vertex is None or bounding box in vertex
    """
    detections = detect_filter(detections, target_classes, vertex)
    
    for label, confidence, bbox, _ in detections:        
        left, top, right, bottom = bbox2points(bbox)        
        
        cv2.rectangle(image, (left, top), (right, bottom), colors[label], 1)
        cv2.putText(image, "{} [{:.2f}]".format(label, float(confidence)),
                        (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label], 2)
        
    return image

def draw_polylines(img, vertex):
    if vertex != None:
        pts = np.array([vertex], np.int32)
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
#         path = os.path.join(output_dir, get_current_date_string(), get_current_hour_string())
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
                print("[Error] Delete video error:",e)
                
                
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
