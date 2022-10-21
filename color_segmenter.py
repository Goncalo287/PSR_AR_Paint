#!/usr/bin/env python3


# Imports
import cv2
import numpy as np
import json
from colorama import Fore, Style, init

init(autoreset=True)        # Initialize colorama


def getLimits(window_name):
    min_b = cv2.getTrackbarPos('min B', window_name)
    min_g = cv2.getTrackbarPos('min G', window_name)
    min_r = cv2.getTrackbarPos('min R', window_name)

    max_b = cv2.getTrackbarPos('max B', window_name)
    max_g = cv2.getTrackbarPos('max G', window_name)
    max_r = cv2.getTrackbarPos('max R', window_name)

    min = (min_b, min_g, min_r)
    max = (max_b, max_g, max_r)

    min = np.array(min, np.uint8)
    max = np.array(max, np.uint8)

    return min, max


def onTrackbar(val):    # replaced by getLimits() but
    pass                # required to create trackbars


def main():

    # Create windows
    name_segmented = 'Segmented'
    name_original = 'Original'
    cv2.namedWindow(name_segmented, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_original, cv2.WINDOW_AUTOSIZE)


    # Create trackbars {'name': count}
    trackbars = {'min B': 100, 'max B': 200, 'min G': 100, 'max G': 200, 'min R': 100, 'max R': 200}
    for tb in trackbars:
        cv2.createTrackbar(tb, name_segmented, trackbars[tb], 255, onTrackbar)


    # Select camera
    capture = cv2.VideoCapture(0)
    _, image = capture.read()


    # Start message: title + keys
    print(Fore.CYAN + "_"*30 + "\n\n       COLOR SEGMENTER\n" + "_"*30 + Fore.RESET +
            '\n\n Save and exit: ' + Fore.CYAN + 'w / ENTER' + Fore.RESET +
            '\n Exit without saving: ' + Fore.CYAN + 'q / ESC\n' + Fore.RESET)


    while True:

        # Update image from camera
        _, image = capture.read()
        image = cv2.flip(image, 1)
        cv2.imshow('Original', image)


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

            file_name = 'limits.json'
            with open(file_name, 'w') as file_handle:
                print(Fore.CYAN + '\n Saved results to ' + Style.BRIGHT + file_name + "\n")
                json.dump(dict, file_handle)
            break

        elif key == ord('q') or key == 27:  # q or ESC
            print(Fore.RED + '\n Closed without saving\n')
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()