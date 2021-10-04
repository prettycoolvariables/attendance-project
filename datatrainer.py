import cv2
import streamlit

imageplace=streamlit.empty()

global count
# idno=0
count=0
camera=cv2.VideoCapture(0)
facedetector=cv2.CascadeClassifier("face.xml")

def dataset(name,idno):
    global count
    #idno+=1
    #print(name)
    print(idno)
    while True:
        success,frame=camera.read()
        if success:
            img=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            frame1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            face=facedetector.detectMultiScale(img,minNeighbors=10)
            #print(face)
            if len(face):
                for x,y,w,h in face:
                    cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
                    plate=frame[y:y+h,x:x+w]
                    count+=1

                    if count<=30:
                        print("test...........")
                        #cv2.imshow("plate",plate)
                        #cv2.waitKey(200)
                        cv2.imwrite("face/user"+"."+str(idno)+"."+str(count)+".jpg",plate)
                        cv2.imwrite("test.jpeg",plate)
                if count>30:
                    camera.release()
                    break
            imageplace.image(frame1,width=400)          



