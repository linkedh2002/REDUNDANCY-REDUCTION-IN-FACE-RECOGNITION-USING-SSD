#Required Libaries
import cv2
import numpy as np
import face_recognition
import os
from PIL import ImageGrab
#Importing Data
path = 'Data'
images = []
classNames = []
myList = os.listdir(path)
#print(myList)
#Encoding the Data and Training
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print(classNames)
#Encoding the Data and Training
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

#def captureScreen(bbox=(300,300,690+300,530+300)):
#    capScr = np.array(ImageGrab.grab(bbox))
#    capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
#    return capScr

encodeListKnown = findEncodings(images)
print(len(encodeListKnown))
print('Encoding Complete')

#cap = cv2.VideoCapture(0)

while True:
    #success, img = cap.read()
    img = cv2.imread('test 5.jpg')
    #img = captureScreen()
    imgS = cv2.resize(img,(0,0),None,1,1)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            #y1, x2, y2, x1 = y1 * 2, x2 * 2, y2 * 2, x1 * 2
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1, y2-30), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img,name,(x1+9,y2-9),cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),2)

        else:
            name = 'Unknown'
        # print(name)
            y1, x2, y2, x1 = faceLoc
            #y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2)

    cv2.imshow('Face Detected',img)
    cv2.waitKey(1)