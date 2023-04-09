# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 20:46:55 2022

@author: kamipakistan
"""
import numpy as np
import cv2

conf_threshold = 0.5
NMS_threshlod = 0.3

cam_feed = cv2.VideoCapture(0)

cam_feed.set(cv2.CAP_PROP_FRAME_WIDTH, 650)
cam_feed.set(cv2.CAP_PROP_FRAME_HEIGHT, 750)

## Loading Coco Names file
classesFile = "coco.names"
with open(classesFile, "rt") as file:
    classNames = file.read().rstrip('\n').split('\n')
print(f"classNames = {classNames}")

net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
# declaring that we are using opencv as backend
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# declaring to use CPU
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

model = cv2.dnn.DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

while cam_feed.isOpened():
    ret, frame = cam_feed.read()
    classes, scores, boxes = model.detect(frame, conf_threshold, NMS_threshlod)
    for classId, score, box in zip(classes, scores, boxes):
        print(classId)
        cv2.rectangle(frame, box, (0, 255, 0))
        print(box)
        cv2.putText(frame, f'{classNames[classId].upper()} {int(score * 100)}%', (box[0], box[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cam_feed.release()
cv2.destroyAllWindows()
