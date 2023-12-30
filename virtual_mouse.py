import cv2
import numpy as np
import hand_detection as htm
import time
import autopy

# Constants for camera and screen dimensions
CAM_WIDTH, CAM_HEIGHT = 640, 480
SCREEN_WIDTH, SCREEN_HEIGHT = autopy.screen.size()
FRAME_MARGIN = 50
SMOOTHENING_FACTOR = 20

# Initial positions for mouse control
prev_loc_x, prev_loc_y = 0, 0
curr_loc_x, curr_loc_y = 0, 0

# Set up the video capture
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
k1 =0 
k2 = -1
k3 = 0
# Initialize variables for frame processing
prev_time = 0
detector = htm.handDetector(maxHands=1)

while True:
    try:
        success, img = cap.read()
        img = cv2.flip(img, 1)

        # Detect hands and find landmarks
        img = detector.findHands(img)
        lm_list, bbox = detector.findPosition(img)

        if k3 ==0:
            print("WEBCAM HAS STARTED....PRESS 'Q' or 'Ctrl + C' TO EXIT")
            k3 =-1
        if len(lm_list) != 0:
            # Draw a rectangle around the area of interest
            cv2.rectangle(img, (FRAME_MARGIN, FRAME_MARGIN), (CAM_WIDTH - FRAME_MARGIN, CAM_HEIGHT - FRAME_MARGIN),
                          (255, 255, 255), 3)

            x1, y1 = lm_list[8][1:]
            x2, y2 = lm_list[12][1:]
            fingers = detector.fingersUp()

            # Calculate frame rate
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            if fingers[2] == 1 and fingers[3] == 0:
                if(k1 ==0):
                    print("TRACKING MODE")
                    k2 = 0
                    k1 =-1
                # Map hand coordinates to screen coordinates
                x3 = np.interp(x1, (FRAME_MARGIN, CAM_WIDTH - FRAME_MARGIN), (0, SCREEN_WIDTH))
                y3 = np.interp(y1, (FRAME_MARGIN, CAM_HEIGHT - FRAME_MARGIN), (0, SCREEN_HEIGHT))
                curr_loc_x = prev_loc_x + (x3 - prev_loc_x) / SMOOTHENING_FACTOR
                curr_loc_y = prev_loc_y + (y3 - prev_loc_y) / SMOOTHENING_FACTOR
                autopy.mouse.move(x3, y3)
                cv2.circle(img, (x1, y1), 15, (128, 0, 0), cv2.FILLED)
                prev_loc_x, prev_loc_y = curr_loc_x, curr_loc_y

            if fingers[2] == 1 and fingers[3] == 1:
                # Perform a mouse click if distance between fingers is small
                length, img, line_info = detector.findDistance(8, 12, img)
                if k2==0:
                    print("SELECTION MODE")
                    k2 =-1
                    k1 =0
                if length <= 50:
                    cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                    autopy.mouse.click()

            # Display frame rate
            cv2.putText(img, f"FPS: {int(fps)}", (20, 40), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)

        # Display the image
        cv2.imshow("IMAGE", img)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print({e})
        continue

# Release the video capture and close all windows
cap.release()
cv2.destroyAllWindows()
