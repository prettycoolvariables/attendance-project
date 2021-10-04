import cv2
import numpy
import os
import streamlit
import json
imageplace2=streamlit.empty()
databaseopen=open("database.json",'r+')
dbreadwrite=json.load(databaseopen)

def trainer():
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    path="face"
    print("trainer test")
    def gimage(path):
        ids=[]
        faces=[]
        for i in os.listdir(path):
            a=cv2.imread("face/"+i,0) 
            faces.append(a)
            #cv2.imshow("image",a)
            #cv2.waitKey(500)
            id=i.split(".")[1]
            ids.append(int(id))
        return (faces,ids)   
    face,hello=gimage(path)
    print(face,hello)
    recognizer.train(face,numpy.array(hello))
    recognizer.write("facenew.xml")
        


def finder():
    camera=cv2.VideoCapture(0)
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    facedetector=cv2.CascadeClassifier("face.xml") 
    recognizer.read("facenew.xml")
    attendance=[]
    while True:
        success,frame=camera.read()
        if success:
            frame1=cv2.cvtColor(frame,cv2,cv2.COLOR_BGR2RGB)
            img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            face=facedetector.detectMultiScale(img,minNeighbors=10)
            if len(face):
                for x,y,w,h in face:
                    cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
                    plate=img[y:y+h,x:x+w]
                    id,percentdata=recognizer.predict(plate)
                    print(id)
                    percent=round(abs(100-percentdata))
                    if percent>=45:
                        # print(dbreadwrite)
                        cv2.putText(frame1,str(dbreadwrite[str(id)]),(x+5,y-5),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
                        cv2.putText(frame1,str(percent)+"%",(x+5,y+h+22),cv2.FONT_HERSHEY_PLAIN,2,(255,0,0),2)
                        #attendance.append(d[id])                      
            imageplace2.image(frame1,width=700)
            

# attendset=set(attendance)
# print(attendset)  
# f=open("data.txt",'a')  
# f.write("\n"+str(xt))
# for i in attendset:
#     f.write("\n"+str(i))  
# f.close()