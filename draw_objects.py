
#!/usr/bin/env python3

# Imports
import argparse
import cv2
import numpy as np
from time import ctime
import json
from colorama import Fore, Style, init
from functools import partial
import math
from copy import copy


drawing = False  # true if mouse is pressed
pt1_x, pt1_y = None, None
copied = False
copied_image = None
flag = 0
past_x, past_y = None, None
holding = False
finished = False
refPt = []



#Function for calculating distance between two points in a plane 
def distance(current_location, previous_location):
    return int(math.sqrt(
        math.pow(current_location[0] - previous_location[0], 2) + math.pow(current_location[1] - previous_location[1],
                                                                           2)))

#Used for creation of ellipses
def angle(x1, x2, y1, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))



# mouse callback function from drawing on canvs with mouse over video feed
# mouse callback function for drawing on canvas with mouse
def main_objects(event, x, y, flags, param,image_name, img, shape, color, thickness):
    global pt1_x, pt1_y, drawing, copied, copied_image
    global holding, finished
    global refPt
    print(refPt)
    # user presses the left button
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt1_x, pt1_y = x, y
        refPt.append((x, y))
    # starts drawing
    elif event == cv2.EVENT_MOUSEMOVE:
        
        if drawing:
            
            # after the rectangle, he disappears if the button wasnt pressed
            copied_image = img.copy()
            if shape == 1:
                if not copied:
                    copied_image = img.copy()
                    copied = True
                cv2.rectangle(copied_image, refPt[0], (x, y), color, thickness)
            if shape == 2:
                if not copied:
                    copied_image = img.copy()
                    copied = True
                cv2.circle(copied_image, refPt[0], distance((x, y), refPt[0]), color, thickness)
            if shape == 3:
                if not copied:
                    copied_image = img.copy()
                    copied = True
                cv2.ellipse(copied_image, refPt[0], (abs(x - pt1_x), abs(y - pt1_y)),
                            angle(pt1_x, x, pt1_y, y),
                            0., 360, color, thickness)
            cv2.imshow(image_name, copied_image)
            

    # stops drawing
    elif event == cv2.EVENT_LBUTTONUP:
        drawing, copied = False, False
        
        if shape == 1:
            cv2.rectangle(img, refPt[0], (x, y), color=color, thickness=thickness)
        if shape == 2:
            cv2.circle(img, refPt[0], distance((x, y), refPt[0]), color, thickness)
        if shape == 3:
            cv2.ellipse(img, refPt[0], (abs(x - pt1_x), abs(y - pt1_y)), angle(pt1_x, x, pt1_y, y), 0.,
                        360, color,
                        thickness)
        refPt = []


    



