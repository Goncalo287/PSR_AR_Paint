#!/usr/bin/env python3

# Imports
import argparse
import cv2
import numpy as np
import json
from colorama import Fore, Style, init

init(autoreset=True)        # Initialize colorama

def readJsonFile(filename):
    with open(filename, 'r') as f:
        limits = json.load(f)
    return limits['limits']

def main():
    ### define and read arguments
    parser = argparse.ArgumentParser(description='AR Paint usage')
    parser.add_argument('-j', '--json', type=str, default='limits.json',
        required=False,
        help='Full path to json file.')

    args = vars(parser.parse_args())

    json_file = args['json']

    limits = readJsonFile(json_file)
    print(limits)

    ## Capture Video
    name_original = "Original"
    capture = cv2.VideoCapture(0)
    _, image = capture.read()

    while True:
        # Update image from camera
        _, image = capture.read()
        cv2.imshow(name_original, image)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xFF        # Only read last byte (prevent numlock)
        
        ## Create Blank Canvas
        canvas_window = "Canvas"
        canvas_height = image.shape[0]
        canvas_width = image.shape[1]
        canvas_channels = image.shape[2]
        canvas = np.zeros((canvas_height,canvas_width,canvas_channels), dtype=np.uint8)
        canvas.fill(255)
        cv2.imshow(canvas_window, canvas)

        if key == ord('q') or key == 27:  # q or ESC
            break    

    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()