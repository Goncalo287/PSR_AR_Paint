
#!/usr/bin/env python3

# Imports
import argparse
import cv2
import numpy as np
from time import ctime
import json
from colorama import Fore, Style, init
from functools import partial

refPt = []

final_boundaries = []
image = None

cv2.ellipse(image, center_coordinates, axesLength,
           angle, startAngle, endAngle, color, thickness)

def click_and_crop(event, x, y, flags, param, key):
    global refPt, image
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt = [(x, y)]
    elif event == cv2.EVENT_LBUTTONUP:
        refPt.append((x, y))
        final_boundaries.append((refPt[0],refPt[1]))
        if key == ord('s'):
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
        if key == ord('s'):
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
        if key == ord('s'):
            cv2.rectangle(image, refPt[0], refPt[1], (0, 255, 0), 2)
            cv2.imshow("image", image)
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        clone = image.copy()
        cv2.rectangle(clone, refPt[0], (x, y), (0, 255, 0), 2)
        cv2.imshow("image", clone)


def main_draw(image_name, key):
    global image
    image = image_name
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", partial(click_and_crop, key=key))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return (final_boundaries)