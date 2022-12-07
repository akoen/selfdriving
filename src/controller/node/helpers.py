import cv2
import numpy as np

def detectMotion(frame, prev_frame, bounds=(0,0,0,0), threshold=10):
    def togray(mat): return cv2.cvtColor(mat, cv2.COLOR_BGR2GRAY)
    def blur(mat): return cv2.GaussianBlur(mat, (5, 5), sigmaX=0)

    rows, cols, channels = frame.shape

    x1,x2,y1,y2 = bounds
    frame_diff = cv2.absdiff(
        blur(togray(frame)), blur(togray(prev_frame)))

    motion_frame = cv2.threshold(
        frame_diff, 20, 255, type=cv2.THRESH_BINARY)[1]

    motion = np.sum(motion_frame[x1:cols-x2, y1:rows-y2]) 
    print(motion)

    motion_frame = cv2.rectangle(motion_frame, (x1, y1), (cols-x2, rows-y2), (255,255,255), 2)
    return motion_frame, motion > threshold