import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2
import uuid
import os
import time

model = torch.hub.load('ultralytics/yolov5','custom',path= 'yolov5/runs/train/exp4/weights/last.pt',force_reload=True)


# accesses the webcam, reads the frames, closes if q is pressed
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    results = model(frame)

    cv2.imshow('Drowsiness Detector (q to exit)',np.squeeze(results.render()))

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows
