#!/usr/bin/env python3

# Imports
import argparse
import cv2
import numpy as np
from time import ctime
import json
from colorama import Fore, Style, init
from math import sqrt
from functools import partial

init(autoreset=True)        # Initialize colorama


def args():
    parser = argparse.ArgumentParser(description='AR Paint usage')
    parser.add_argument('-j', '--json', type=str, default='limits.json',
        required=False,
        help='Full path to json file.')
    parser.add_argument('-usp', '--use_shake_prevention', action='store_true', default=False,
        required=False,
        help='Activate shake prevention.')

    args = vars(parser.parse_args())

    return args['json'], args['use_shake_prevention']


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


def showResized(name, image, resize):

    (h, w) = image.shape[:2]

    h = int(h * resize)
    w = int(w * resize)

    image = cv2.resize(image, (w, h))
    cv2.imshow(name, image)


def getCentroid(image_thresh):
    _, thresh = cv2.threshold(image_thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    connectivity = 4
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh , connectivity , cv2.CV_32S)
    max_idx = 1
    for i in range(1, len(stats)):
        max_idx = i if stats[i][4] > stats[max_idx][4] \
            else max_idx

    x = -1
    y = -1

    if len(centroids) > 1 \
            and (centroids[max_idx][0] + 0.5 != image_thresh.shape[1] / 2 \
            or centroids[max_idx][1] + 0.5 != image_thresh.shape[0] / 2):
        x = int(centroids[max_idx][0])
        y = int(centroids[max_idx][1])

    return x, y


def drawCentroid(image, x, y):
    if x != -1 and y != -1:
        # Drawing cross on the webcam feed
        cv2.line(image, (int(x-10), int(y)), (int(x+10), int(y)), (0, 0, 255), 1)
        cv2.line(image, (int(x), int(y-10)), (int(x), int(y+10)), (0, 0, 255), 1)            

    return image


def drawLine(canvas, pencil, usp):
    x, y = pencil['x'], pencil['y']
    last_x, last_y = pencil['last_x'], pencil['last_y']

    if last_x != -1 and last_y != -1:
        if usp:
            distance = sqrt((last_x-x)**2+(last_y-y)**2)
            if distance > 100:
                last_x = int(x)
                last_y = int(y)
        cv2.line(canvas, (int(x),int(y)), (last_x, last_y), pencil['color'], pencil['size'])

    return canvas


def mouseMove(event, x, y, flags, params, pencil):
    if pencil['use_mouse']:
        if event == cv2.EVENT_MOUSEMOVE:
            pencil['x'] = x
            pencil['y'] = y


def main():

    # ---------------------
    #    Initialization
    # ---------------------

    ### define and read arguments
    json_file, use_shake_prevention = args()

    try:
        limits = readJsonFile(json_file)
        print('Load limits from ' + json_file)
    except FileNotFoundError:
        print("File not found: " + json_file)
        exit()

    ## Capture Video
    capture = cv2.VideoCapture(0)
    _, image = capture.read()
    image = cv2.flip(image, 1)


    # Create windows
    name_original = 'Original'
    name_segmented = 'Segmented'
    name_largest = 'Largest Object'
    name_canvas = 'Canvas'

    cv2.namedWindow(name_original, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_segmented, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_largest, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_canvas, cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow(name_original, 100, 100)
    cv2.moveWindow(name_segmented, image.shape[1] + 120, 100)
    cv2.moveWindow(name_largest, image.shape[1] + 120, int(image.shape[0]/2) + 200)
    cv2.moveWindow(name_canvas, int(image.shape[1]*1.5 + 200), 100)


    ## Create Blank Canvas
    canvas = np.zeros(image.shape, dtype=np.uint8)
    canvas.fill(255)
    cv2.imshow(name_canvas, canvas)


    # Initial pencil properties
    pencil = {'x': -1,
                'y': -1,
                'last_x': -1,
                'last_y': -1,
                'color': (0, 0, 0),
                'size': 1,
                'use_mouse': False}

    cv2.setMouseCallback(name_canvas, partial(mouseMove, pencil=pencil))
    camera_mode = False


    # ---------------------
    #      Main loop
    # ---------------------


    while True:
        ## Update image from camera
        _, image = capture.read()
        image = cv2.flip(image, 1)

        ## Update Segmented Image
        min, max = getLimits(limits)
        image_thresholded = cv2.inRange(image, min, max)
        showResized(name_segmented, image_thresholded, 0.5)


        ## Find centroid and draw cross in camera
        if not pencil['use_mouse']:
            pencil['x'], pencil['y'] = getCentroid(image_thresholded)
        image = drawCentroid(image, pencil['x'], pencil['y'])
        cv2.imshow(name_original, image)

        ## Draw in canvas
        canvas = drawLine(canvas, pencil, use_shake_prevention)


        ## TODO: Another window with only the largest component


        # Save position
        pencil['last_x'] = pencil['x']
        pencil['last_y'] = pencil['y']

        # Update image in canvas (check camera mode)
        if camera_mode:

            # Overlay canvas drawing on camera (exclude background color with threshold)
            canvas_cam = image.copy()
            canvas_gray= cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, canvas_thresh = cv2.threshold(canvas_gray, 254, 255, cv2.THRESH_BINARY)
            canvas_cam[canvas_thresh==0] = canvas[canvas_thresh==0]
            cv2.imshow(name_canvas, canvas_cam)
        else:
            cv2.imshow(name_canvas, canvas)

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xFF        # Only read last byte (prevent numlock)
    
        if key == ord('q') or key == 27:  # q or ESC - quit without saving
            break

        elif key == ord('c'): # c - clear the canvas
            canvas.fill(255)

        elif key == ord('w'): # w - save the current canvas
            drawing_filename = f"drawing_{ctime().replace(' ','_')}.png"
            cv2.imwrite(drawing_filename, canvas)

        elif key == ord('r'): # r - change pencil color to red
            pencil['color'] = (0, 0, 255)

        elif key == ord('g'): # g - change pencil color to green
            pencil['color'] = (0, 255, 0)

        elif key == ord('b'): # b - change pencil color to blue
            pencil['color'] = (255, 0, 0)

        elif key == ord('+'): # + - increase pencil size
            pencil['size'] += 1

        elif key == ord('-'): # - - decrease pencil size
            pencil['size'] -= 1 if pencil['size'] > 1 else 0

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

        elif key == ord('n'): # n - toggle mouse input
            pencil['use_mouse'] = False if pencil['use_mouse'] else True

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()