import cv2
import time
import HandTrackingModule as htm
import numpy as np
import os

overlayList=[]

brushThickness = 15
eraserThickness = 100
drawColor=(255,0,255)

xp, yp = 0, 0
imgCanvas = np.zeros((720, 1280, 3), np.uint8)


folderPath="Header"
myList=os.listdir(folderPath)
#print(myList)
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)#inserting images one by one in the overlayList
header=overlayList[0]#storing 1st image 
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

detector = htm.handDetector(detectionCon=0.50,maxHands=1)

while True:

    # 1. Import image
    success, img = cap.read()
    img=cv2.flip(img,1)
    
    # 2. Find Hand Landmarks
    img = detector.findHands(img,draw=True)
    lmList,bbox = detector.findPosition(img, draw=False)
    
    if len(lmList)!=0:
        print(lmList)
        x1, y1 = lmList[8][1],lmList[8][2]# tip of index finger
        x2, y2 = lmList[12][1],lmList[12][2]# tip of middle finger
        
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        #print(fingers)

        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp,yp=0,0
            if y1 < 125:
                if 100 < x1 < 250:
                    header = overlayList[0]
                    drawColor = (255, 0, 255)
                elif 350 < x1 < 500:
                    header = overlayList[1]
                    drawColor = (235, 206, 135)
                elif 550 < x1 < 730:#if i m clicking at green brush
                    header = overlayList[2]
                    drawColor = (0, 0, 255)
                elif 820 < x1 < 950:#if i m clicking at eraser
                    header = overlayList[3]
                    drawColor = (0, 255, 0)
                elif 1065 < x1 < 1233:
                    header = overlayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawColor, cv2.FILLED)


        
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 5, drawColor, -1)
            
            if xp == 0 and yp == 0:
                xp, yp = x1, y1 
            
            
            #eraser
            if drawColor == (0, 0, 0):
                cv2.line(img, (xp, yp), (x1, y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, eraserThickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, brushThickness)
            xp,yp=x1,y1 
           
           
    
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    
    
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    
    img = cv2.bitwise_and(img,imgInv)
    
    
    img = cv2.bitwise_or(img,imgCanvas)


    
    img[0:125,0:1280]=header

    cv2.imshow("Image", img)
    #cv2.imshow("Canvas", imgCanvas)
    #cv2.imshow("Inv", imgInv)
    if cv2.waitKey(1) & 0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()