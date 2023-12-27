import cv2
import numpy as np
import hand_detection as htm
import time
import autopy

########################
wCam, hCam = 640, 480
wScr, hScr = autopy.screen.size()
frameR = 50
smoothening = 20
########################

plocX, plocY = 0, 0 
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
pTime = 0
detector = htm.handDetector(maxHands = 1)
while True:
    try:
        sucess, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList, bbox = detector.findPosition(img)
        if(len(lmList)!=0):
            cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam -frameR), (255, 255, 255), 3)
            x1, y1 = lmList[8][1:]
            x2, y2 = lmList[12][1:]
            fingers = detector.fingersUp()
            # print(fingers)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            if fingers[2] == 1 and fingers[3] ==0:
                x3 =np.interp(x1, (frameR, wCam-frameR), (0, wScr))
                y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
                clocX = plocX + (x3-plocX)/smoothening
                clocY = plocY + (y3-plocY)/smoothening
                autopy.mouse.move(x3, y3)
                cv2.circle(img, (x1,y1), 15, (128,0,0), cv2.FILLED)
                plocX, plocY = clocX, clocY
            if fingers[2] == 1 and fingers[3] ==1:
                length, img, lineInfo =detector.findDistance(8, 12, img)
                print(length)
                if length <= 50:
                    cv2.circle(img, (lineInfo[4], lineInfo[5]), 15, (0,255,0), cv2.FILLED)
                    autopy.mouse.click()

        cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),  2)
        cv2.imshow("IMAGE", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(img, str(int(fps)), (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0),  2)
            cv2.imshow("IMAGE", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print({e})
        continue

cap.release()
cv2.destroyAllWindows()