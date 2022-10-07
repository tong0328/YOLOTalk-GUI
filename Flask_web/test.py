import cv2


cap = cv2.VideoCapture("rtsp://lab610:lab610@140.113.131.8:554/stream1")  
print(cap.isOpened())

while True:
    ret, frame = cap.read()
    