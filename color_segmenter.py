#!/usr/bin/env python3


# Imports
import cv2
import numpy as np
import json
from colorama import Fore, Style, init
from functools import partial

init(autoreset=True)        # Initialize colorama


def readJsonFile(filename):
    with open(filename, 'r') as f:
        limits = json.load(f)
    return limits['limits']


def getLimits(window_name):
    min_b = cv2.getTrackbarPos('min B', window_name)
    min_g = cv2.getTrackbarPos('min G', window_name)
    min_r = cv2.getTrackbarPos('min R', window_name)

    max_b = cv2.getTrackbarPos('max B', window_name)
    max_g = cv2.getTrackbarPos('max G', window_name)
    max_r = cv2.getTrackbarPos('max R', window_name)

    min = np.array([min_b, min_g, min_r], np.uint8)
    max = np.array([max_b, max_g, max_r], np.uint8)

    return min, max


def onTrackbar(val):    # replaced by getLimits() but
    pass                # required to create trackbars


def mouseClick(event, x, y, flags, param, window_name, img_dict):

    if event == cv2.EVENT_LBUTTONDOWN:
        
        image = img_dict['image']
        b, g, r = image[y, x]
    
        range = 30

        min_b = b - range if b > range else 0
        max_b = b + range if b < 255-range else 255
        min_g = g - range if g > range else 0
        max_g = g + range if g < 255-range else 255
        min_r = r - range if r > range else 0
        max_r = r + range if r < 255-range else 255

        limits = {'min B': min_b, 'max B': max_b, 'min G': min_g, 'max G': max_g, 'min R': min_r, 'max R': max_r}

        for limit in limits:
            cv2.setTrackbarPos(limit, window_name, limits[limit])


def main():

    # Start message: title + keys
    print(Fore.CYAN + '_'*30 + '\n\n       COLOR SEGMENTER\n' + '_'*30 + Fore.RESET +
            '\n\n Save and exit: ' + Fore.CYAN + 'w / ENTER' + Fore.RESET +
            '\n Exit without saving: ' + Fore.CYAN + 'q / ESC' + Fore.RESET +
            '\n Pick color in camera: ' + Fore.CYAN + 'Mouse Left Click\n')


    # Create windows
    name_segmented = 'Segmented'
    name_original = 'Original'
    cv2.namedWindow(name_segmented, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_original, cv2.WINDOW_AUTOSIZE)


    # Get limits from file
    file_name = 'limits.json'

    try:
        limits = readJsonFile(file_name)
        print(Fore.CYAN + ' Loading limits from ' + Style.BRIGHT + file_name)
    except FileNotFoundError:
        limits = {'B': {'max': 200, 'min': 100}, 'G': {'max': 200, 'min': 100}, 'R': {'max': 200, 'min': 100}}


    # Create trackbars {'name': count}
    trackbars = {'min B': limits['B']['min'],
                    'max B': limits['B']['max'],
                    'min G': limits['G']['min'],
                    'max G': limits['G']['max'],
                    'min R': limits['R']['min'],
                    'max R': limits['R']['max']}

    for tb in trackbars:
        cv2.createTrackbar(tb, name_segmented, trackbars[tb], 255, onTrackbar)


    # Select camera and get first frame
    capture = cv2.VideoCapture(0)
    _, image = capture.read()
    image = cv2.flip(image, 1)
    img_dict = {'image': image}


    # Move windows to starting pos
    cv2.moveWindow(name_original, 100, 0)
    cv2.moveWindow(name_segmented, image.shape[1] + 120, 0)


    # Create mouse callback
    cv2.setMouseCallback(name_original, partial(mouseClick, window_name=name_segmented, img_dict=img_dict))


    while True:

        # Update image from camera
        _, image = capture.read()
        image = cv2.flip(image, 1)
        img_dict['image'] = image
        cv2.imshow(name_original, image)


        # Update segmented image
        min, max = getLimits(name_segmented)
        image_thresholded = cv2.inRange(image, min, max)
        cv2.imshow(name_segmented, image_thresholded)


        # Keyboard inputs
        key = cv2.waitKey(10) & 0xFF        # Only read last byte (prevent numlock)

        if key == ord('w') or key == 13:    # w or ENTER
            min = min.tolist()
            max = max.tolist()

            dict = {'limits': {'B': {'max': max[0], 'min': min[0]},
                                'G': {'max': max[1], 'min': min[1]},
                                'R': {'max': max[2], 'min': min[2]}}}

            with open(file_name, 'w') as file_handle:
                print(Fore.CYAN + '\n Saved results to ' + Style.BRIGHT + file_name + '\n')
                json.dump(dict, file_handle, indent=4)
            break

        elif key == ord('q') or key == 27:  # q or ESC
            print(Fore.RED + '\n Closed without saving\n')
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()