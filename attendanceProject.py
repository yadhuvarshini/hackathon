import cv2
import numpy as np
import face_recognition as face_recognition
import os
from datetime import datetime
from flask import Flask, jsonify, render_template

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])
def home():
    if request.method == 'POST':
        name = request.form['fname']
        reg_no = request.form['reg_no']
        dept = request.form['dept']
        return render_template('home.html', name=name, reg_no=reg_no, dept=dept)
    else:
        return render_template('home.html')
# from PIL import ImageGrab


path = 'imagesAttendance'

images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# reg_no = input()

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        try:
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
            return encodeList
        except:
            error = []
            print("found no faces")
            return error

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

if reg_no in classNames:
    print(len(classNames))
    print("already In")
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        #img = captureScreen()
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

        for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
            #print(faceDis)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] > 0.50:
                name = classNames[matchIndex].upper()
                markAttendance(name)

            else: 
                name='unknown'
                cap = cv2.VideoCapture(0) # use 0 if you only have front facing camera
                cv2.waitKey(5000)
                ret, frame = cap.read() #read one frame
                print(frame.shape)
                cap.release() # release the VideoCapture object.

                cv2.imshow('image', frame)
                status = cv2.imwrite(path+'/'+name+'.jpg',frame)

            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)

        cv2.imshow('Webcam',img)
        cv2.waitKey(1)
        # cap.release()
        
if cv2.waitKey(0) & 0xff == ord('q'): # press q to exit
    cv2.destroyAllWindows()

else:
    cap = cv2.VideoCapture(0) # use 0 if you only have front facing camera
    cv2.waitKey(5000)
    ret, frame = cap.read() #read one frame
    print(frame.shape)
    cap.release() # release the VideoCapture object.

    cv2.imshow('image', frame)
    status = cv2.imwrite(path+'/'+reg_no+'.jpg',frame)
    if cv2.waitKey(0) & 0xff == ord('q'): # press q to exit
        cv2.destroyAllWindows()

print("hio")

if __name__ == '__main__':
    app.run(debug=True)
