import cv2 
import numpy as np
from imutils import grab_contours

cap = cv2.VideoCapture(2)
cap.set(3, 640)
cap.set(4, 480)

while True:
    ret, frame = cap.read()
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #                        B  G   R
    lower_yellow = np.array([22, 93, 0], np.uint8)
    upper_yellow =  np.array([45, 255, 255], np.uint8)
    

    
    lower_red = np.array([0,150,0],np.uint8)
    upper_red = np.array([10,255,255],np.uint8)
    
    lower_blue = np.array([100,150,0],np.uint8)
    upper_blue = np.array([140,255,255],np.uint8)
    
    lower_orange = np.array([5, 50, 50],np.uint8)
    upper_orange =  np.array([15, 255, 255],np.uint8)
    
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white =  np.array([0,0,255], dtype=np.uint8)
    
    
    mask_yellow = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_frame,  (36, 25, 25), (70, 255,255))
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)
    mask_blue = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    mask_orange = cv2.inRange(hsv_frame, lower_orange, upper_orange)
    mask_white = cv2.inRange(hsv_frame, lower_white, upper_white)
    
    
    cont_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_yellow = grab_contours(cont_yellow)
    
    cont_green = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_green = grab_contours(cont_green)
    
    cont_red = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_red = grab_contours(cont_red)
    
    cont_blue = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_blue = grab_contours(cont_blue)
    
    cont_orange = cv2.findContours(mask_orange, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_orange = grab_contours(cont_orange)
    
    cont_white = cv2.findContours(mask_white, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cont_white = grab_contours(cont_white)
    
    for cont in cont_yellow:
        area_yellow = cv2.contourArea(cont)
        if area_yellow >1500 and area_yellow < 5000:
            cv2.drawContours(frame, [cont], -1, (0,255,0),3)
            M = cv2.moments(cont)
            cx_yellow = int(M["m10"] /M['m00'])
            cy_yellow = int(M["m01"] /M["m00"])
            
            cv2.circle(frame, (cx_yellow, cy_yellow), 7, (255,255,255),-1)
            
    
    
    for cont in cont_green:
        area_green = cv2.contourArea(cont)
        if area_green >5000:
            cv2.drawContours(frame, [cont], -1, (0,255,0),3)
            M = cv2.moments(cont)
            cx_green = int(M["m10"] /M['m00'])
            cy_green = int(M["m01"] /M["m00"])
            
            cv2.circle(frame, (cx_green, cy_green), 7, (255,255,255),-1)
            
            
    for cont in cont_red:
        area_red = cv2.contourArea(cont)
        if area_red >5000:
            cv2.drawContours(frame, [cont], -1, (0,255,0),3)
            M = cv2.moments(cont)
            cx_red = int(M["m10"] /M['m00'])
            cy_red = int(M["m01"] /M["m00"])
            
            cv2.circle(frame, (cx_red, cy_red), 7, (255,255,255),-1)
            
            
            
    for cont in cont_blue:
        area_blue = cv2.contourArea(cont)
        if area_blue >5000:
            cv2.drawContours(frame, [cont], -1, (0,255,0),3)
            M = cv2.moments(cont)
            cx_blue = int(M["m10"] /M['m00'])
            cy_blue = int(M["m01"] /M["m00"])
            
            cv2.circle(frame, (cx_blue, cy_blue), 7, (255,255,255),-1)
            
            
    for cont in cont_orange:
        area_orange = cv2.contourArea(cont)
        if area_orange >5000:
            cv2.drawContours(frame, [cont], -1, (0,255,0),3)
            M = cv2.moments(cont)
            cx_orange = int(M["m10"] /M['m00'])
            cy_orange = int(M["m01"] /M["m00"])
            
            cv2.circle(frame, (cx_orange, cy_orange), 7, (255,255,255),-1)
            
            
    for cont in cont_white:
        area_white = cv2.contourArea(cont)
        if area_white >5000:
            cv2.drawContours(frame, [cont], -1, (0,255,0),3)
            M = cv2.moments(cont)
            cx_white = int(M["m10"] /M['m00'])
            cy_white = int(M["m01"] /M["m00"])
            
            cv2.circle(frame, (cx_white, cy_white), 7, (255,255,255),-1)
   
    
    cv2.imshow("result", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()