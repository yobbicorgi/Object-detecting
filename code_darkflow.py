# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:08:20 2021

@author: User
"""

from darkflow.net.build import TFNet
import cv2
import numpy as np
import argparse
import pafy

#%%
url = "https://www.youtube.com/watch?v=eflsH8ry3Xw&t=1300s&vq=hd1440" # youtube 영상 링크
video = pafy.new(url)
best = video.getbest(preftype="mp4")
cap = cv2.VideoCapture(best.url)

# cap = cv2.VideoCapture('../sample_racing.mp4') # Input Video
model_path = "./cfg/yolo.cfg"
weights_path = "./bin/yolov2.weights"
name_path="./cfg/coco.names"

options = {"model": model_path, "load": weights_path, "threshold": 0.3, "gpu": 0.7}
tfnet = TFNet(options)

width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

LABELS = open(name_path).read().strip().split("\n")
np.random.seed(3)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8") #요소별 박스 컬러 랜덤 설정
color={name:value for name, value in zip(LABELS,COLORS)}

fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('../output/output.avi',fourcc, 30, (int(width), int(height)))
cnt=0
while True:
    cnt+=1
    ret, frame = cap.read()
    print('Frame Number : '+str(cnt)+'   Progressing : ' +str(cnt/length*100)[:4] +'%')

    if ret == True:
        frame = np.asarray(frame)
        results = tfnet.return_predict(frame)
                
        for result in results:
            top_x = result['topleft']['x']
            top_y = result['topleft']['y']
    
            btm_x = result['bottomright']['x']
            btm_y = result['bottomright']['y']
    
            confidence = result['confidence']
            label = result['label'] + " " + str(round(confidence, 3))
    
            if confidence > options['threshold']:
                c=color[result['label']]
                frame = cv2.rectangle(frame, (top_x, top_y), (btm_x, btm_y), (int(c[0]),int(c[1]),int(c[2])), 3)
                # frame = cv2.rectangle(frame, (top_x, top_y), (btm_x, btm_y), (0,0,250), 2)
                frame = cv2.rectangle(frame, (top_x - 1, top_y), (top_x + len(label) * 11+15 , top_y - 20), (0, 0, 0), -1)
                frame = cv2.putText(frame, label, (top_x, top_y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL , 0.9, (250, 250, 250), 1, cv2.LINE_AA)
        
        out.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()