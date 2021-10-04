import cv2
import streamlit
import datatrainer
import facetrainer
import json

name=streamlit.sidebar.text_input("name")
a=streamlit.sidebar.button("register") 

if a:
    databaseopen=open("database.json",'r+')
    dbreadwrite=json.load(databaseopen)
    dbreadwrite.update({len(dbreadwrite)+1:name})
    databaseopen.seek(0)
    json.dump(dbreadwrite,databaseopen)
    datatrainer.dataset(name,len(dbreadwrite))
    streamlit.title("get ready")
    print(dbreadwrite)
    facetrainer.trainer()    
facetrainer.finder()