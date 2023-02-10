from flask import Flask,render_template,request,redirect,url_for
import cv2
import numpy as np
import face_recognition as face_recognition
import os
from datetime import datetime

app = Flask(__name__)
path = 'imagesAttendance'
imgBackGround = cv2.imread('Resources/background.png')


images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)


@app.route("/", methods=['GET','POST'])
def home():
    if request.method == 'POST':
        name = request.form['fname']
        reg_no = request.form['reg_no']
        dept = request.form['dept']

        cap = cv2.VideoCapture(0) # use 0 if you only have front facing camera
        cv2.waitKey(5000)
        ret, frame = cap.read() #read one frame
        print(frame.shape)
        cap.release() # release the VideoCapture object.

        cv2.imshow('image', frame)
        status = cv2.imwrite(path+'/'+reg_no+'.jpg',frame)
        if cv2.waitKey(0) & 0xff == ord('q'): # press q to exit
            cv2.destroyAllWindows()
        return render_template('home.html',fname=name,reg_no=reg_no,dept=dept)

    else:
        return render_template("home.html")


def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        print(encode)
        encodeList.append(encode)
    return encodeList

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

#### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# def captureScreen(bbox=(300,300,690+300,530+300)):
#     capScr = np.array(ImageGrab.grab(bbox))
#     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#     return capScr

images = []
classNames = []

@app.route('/login')
def login():
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
		


    

