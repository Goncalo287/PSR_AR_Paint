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

def getLimits(limits):
    min_b = limits['B']['min']
    min_g = limits['G']['min']
    min_r = limits['R']['min']

    max_b = limits['B']['max']
    max_g = limits['G']['max']
    max_r = limits['R']['max']

    min = np.array([min_b, min_g, min_r], np.uint8)
    max = np.array([max_b, max_g, max_r], np.uint8)

    return min, max

def main():
    ### define and read arguments
    parser = argparse.ArgumentParser(description='AR Paint usage')
    parser.add_argument('-j', '--json', type=str, default='limits.json',
        required=False,
        help='Full path to json file.')
    parser.add_argument('-usp', '--use_shake_prevention', action='store_true', default=False,
    required=False,
    help='Activate shake prevention.')

    args = vars(parser.parse_args())

    json_file, use_shake_prevention = args['json'], args['use_shake_prevention']
    print(json_file, use_shake_prevention)

    limits = readJsonFile(json_file)
    print(limits)

    ## Initialize useful variables
    pencil_color = (0,0,0)
    pencil_size = 1

    ## Capture Video
    capture = cv2.VideoCapture(0)
    _, image = capture.read()

    # Create windows
    name_segmented = 'Segmented'
    name_original = 'Original'
    cv2.namedWindow(name_segmented, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_original, cv2.WINDOW_AUTOSIZE)

    
    ## Create Blank Canvas
    canvas_window = "Canvas"
    canvas_height = image.shape[0]
    canvas_width = image.shape[1]
    canvas_channels = image.shape[2]
    canvas = np.zeros((canvas_height,canvas_width,canvas_channels), dtype=np.uint8)
    canvas.fill(255)
    cv2.imshow(canvas_window, canvas)

    last_x = -1
    last_y = -1
    camera_mode = False

    while True:
        ## Update image from camera
        _, image = capture.read()
        cv2.imshow(name_original, image)

        ## Read Json file
        limits = readJsonFile(json_file)

        ## Update Segmented Image
        min, max = getLimits(limits)
        image_thresholded = cv2.inRange(image, min, max)
        cv2.imshow(name_segmented, image_thresholded)

        ## Get largest segmented component
        _, thresh = cv2.threshold(image_thresholded,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        connectivity = 4
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh , connectivity , cv2.CV_32S)
        max_idx = 1
        for i in range(1, len(stats)):
            max_idx = i if stats[i][4] > stats[max_idx][4] \
                else max_idx
        x = -1
        y = -1
        if len(centroids) > 1 \
             and (centroids[max_idx][0] + 0.5 != image.shape[1] / 2 \
             or centroids[max_idx][1] + 0.5 != image.shape[0] / 2):
            x = int(centroids[max_idx][0])
            y = int(centroids[max_idx][1])
        print(f'x: {x}, y: {y}')
        if x != -1 and y != -1:
            # Drawing cross on the webcam feed
            cv2.line(image, (int(x-10), int(y)), (int(x+10), int(y)), (0, 0, 255), 1)
            cv2.line(image, (int(x), int(y-10)), (int(x), int(y+10)), (0, 0, 255), 1)            
            cv2.imshow(name_original, image)


        ## TODO: Another window with only the largest component

        ## Update Canvas
        cv2.line(canvas, (int(x),int(y)), (last_x, last_y), pencil_color, pencil_size) if last_x != -1 and x != -1 \
            else cv2.line(canvas, (x,y), (x,y), pencil_color, pencil_size)
        last_x = x
        last_y = y

        # Update image in canvas (check camera mode)
        if camera_mode:

            # Overlay canvas drawing on camera (exclude background color with threshold)
            canvas_cam = image.copy()
            canvas_gray= cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, canvas_thresh = cv2.threshold(canvas_gray, 254, 255, cv2.THRESH_BINARY)
            canvas_cam[canvas_thresh==0] = canvas[canvas_thresh==0]
            cv2.imshow(canvas_window, canvas_cam)
        else:
            cv2.imshow(canvas_window, canvas)

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
        elif key == ord('s'): # s - draw a square
            #TODO: Advanced Funcionality
            pass
        elif key == ord('e'): # e - draw an ellipse
            #TODO: Advanced Funcionality
            pass
        elif key == ord('o'): # o - draw a circle
            #TODO: Advanced Funcionality
            pass
        
        elif key == ord('m'): # m - toggle camera Mode
            camera_mode = False if camera_mode else True

    cv2.destroyAllWindows()




if __name__ == '__main__':
    main()