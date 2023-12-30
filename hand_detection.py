import cv2
import mediapipe as mp
import time
import math
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=2, detection_con=0.5, track_con=0.5):
        # Initialize HandDetector with configuration parameters
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.track_con = track_con
        self.mp_hands = mp.solutions.hands
        # Initialize MediaPipe Hands module
        self.hands = self.mp_hands.Hands(static_image_mode=self.mode,
                                         max_num_hands=self.max_hands,
                                         min_detection_confidence=self.detection_con,
                                         min_tracking_confidence=self.track_con)
        self.mp_draw = mp.solutions.drawing_utils
        self.tip_ids = [4, 8, 12, 16, 20]

    def find_hands(self, img, draw=True):
        # Convert image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Process the image with the Hand module
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            # Draw landmarks and connections if hands are detected
            for hand_lms in results.multi_hand_landmarks:
                if draw:
                    drawing_spec = self.mp_draw.DrawingSpec
                    landmark_drawing_spec = drawing_spec(color=(0, 128, 0))
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS,
                                                landmark_drawing_spec=landmark_drawing_spec,
                                                connection_drawing_spec=drawing_spec(color=(0, 0, 180)))

        return img

    def find_position(self, img, hand_no=0, draw=True):
        x_list = []
        y_list = []
        bbox = []
        lm_list = []

        if self.hands.results.multi_hand_landmarks:
            # Extract information about the detected hand
            my_hand = self.hands.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(my_hand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                x_list.append(cx)
                y_list.append(cy)
                lm_list.append([id, cx, cy])

                if draw:
                    cv2.circle(img, (cx, cy), 5, (0, 128, 0), cv2.FILLED)

            xmin, xmax = min(x_list), max(x_list)
            ymin, ymax = min(y_list), max(y_list)
            bbox = xmin, ymin, xmax, ymax

            if draw:
                # Draw bounding box around the hand
                cv2.rectangle(img, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20),
                              (0, 255, 0), 2)
        return lm_list, bbox

    def fingers_up(self):
        fingers = []
        # Check if each finger is up or down
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(0, 4):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def find_distance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            # Draw line and circles to represent distance
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]

def main():
    p_time = 0
    c_time = 0
    cap = cv2.VideoCapture(0)
    print("WebCam turned on and recording")
