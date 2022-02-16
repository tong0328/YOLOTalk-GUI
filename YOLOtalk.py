from flask import Flask, render_template,Response ,jsonify, session
import time
import cv2

app = Flask(__name__)
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
#yield 設計來的目的，就是為了單次輸出內容。我們可以把 yield 暫時看成 return，
#但是這個 return 的功能只有單次。而且，一旦我們的程式執行到 yield 後，程式就會把值丟出，並暫時停止。
#「b」是指bytes literal，也就是byte格式的字串
        

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()