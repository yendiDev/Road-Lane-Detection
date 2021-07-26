import matplotlib.pyplot as plt
import cv2
import numpy as np
import math
from utils import process_frame


cap = cv2.VideoCapture('lane_video1.mp4')

#_, frame = cap.read()

#cv2.imwrite('lane2_image.png', frame)

while cap.isOpened():
    _, frame = cap.read()

    frame = process_frame(frame)

    cv2.imshow('Lane Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()