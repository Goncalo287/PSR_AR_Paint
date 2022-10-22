#!/usr/bin/env python3

# Imports
import argparse
import cv2
import numpy as np
from time import ctime
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

    ## Initialize useful variables
    pencil_color = (0,0,0)
    pencil_size = 1

    ## Capture Video
    name_original = "Original"
    capture = cv2.VideoCapture(0)
    _, image = capture.read()
    
    ## Create Blank Canvas
    canvas_window = "Canvas"
    canvas_height = image.shape[0]
    canvas_width = image.shape[1]
    canvas_channels = image.shape[2]
    canvas = np.zeros((canvas_height,canvas_width,canvas_channels), dtype=np.uint8)
    canvas.fill(255)
    cv2.imshow(canvas_window, canvas)

    while True:
        # Update image from camera
        _, image = capture.read()
        cv2.imshow(name_original, image)

        ## Update Canvas
        #TODO: Update canvas image with the "drawings"

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xFF        # Only read last byte (prevent numlock)
    
        if key == ord('q') or key == 27:  # q or ESC - quit without saving
            break  
        elif key == ord('c'): # c - clear the canvas
            canvas = np.zeros((canvas_height,canvas_width,canvas_channels), dtype=np.uint8)
            canvas.fill(255)
            cv2.imshow(canvas_window, canvas)
        elif key == ord('w'): # w - save the current canvas
            drawing_filename = f"drawing_{ctime().replace(' ','_')}.png"
            cv2.imwrite(drawing_filename,canvas)
        elif key == ord('r'): # r - change pencil color to red
            pencil_color = (0, 0, 255)
        elif key == ord('g'): # g - change pencil color to green
            pencil_color = (0, 255, 0)
        elif key == ord('b'): # b - change pencil color to blue
            pencil_color = (255, 0, 0)
        elif key == ord('+'): # + - increase pencil size
            pencil_size += 1
        elif key == ord('-'): # - - decrease pencil size
            pencil_size -= 1 if pencil_size > 0 else 1

    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()