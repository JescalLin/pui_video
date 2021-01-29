#手動模式 按空白鍵逐格拍攝 Enter結束錄影

import numpy as np
import cv2


cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640,  480))

while True:
    ret, image_np = cap.read()
    cv2.imshow('cam', cv2.resize(image_np, (640, 480)))
    key = cv2.waitKey(90)
    if key == 32:
        out.write(image_np)
        print("catch")
    if key == 13:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        break
