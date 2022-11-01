#!/usr/bin/env python3

# Imports
import argparse
import cv2
import numpy as np
from time import ctime
import json
from colorama import Fore, Style, init
import math
from functools import partial
#from draw_objects import main_objects

init(autoreset=True)        # Initialize colorama


def args():
    parser = argparse.ArgumentParser(description='AR Paint usage')
    parser.add_argument('-j', '--json', type=str, default='limits.json',
        required=False,
        help='Full path to json file.')
    parser.add_argument('-usp', '--use_shake_prevention', action='store_true', default=False,
        required=False,
        help='Activate shake prevention.')
    parser.add_argument('-p', '--paint', type=str, 
        required=False,
        help='Choose an image to paint. Options: Fire, Turtle & Bird')

    args = vars(parser.parse_args())

    return args['json'], args['use_shake_prevention'], args['paint']


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


def getCentroid(image_thresh, name_largest):
    _, thresh = cv2.threshold(image_thresh,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    connectivity = 4
    num_labels, output, stats, centroids = cv2.connectedComponentsWithStats(thresh , connectivity , cv2.CV_32S)
    max_idx = 1
    for i in range(1, len(stats)):
        max_idx = i if stats[i][4] > stats[max_idx][4] \
            else max_idx

    area = stats[max_idx][4] if len(centroids) > 1 else 0
    min_area = 100

    # Show largest component
    largest_component = np.zeros(output.shape)
    if area > min_area:
        largest_component[output == max_idx] = 255
    showResized(name_largest, largest_component, 0.5)

    x = -1
    y = -1

    if area > min_area and len(centroids) > 1 \
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
            distance = math.sqrt((last_x-x)**2+(last_y-y)**2)
            if distance > 100:
                last_x = int(x)
                last_y = int(y)
        cv2.line(canvas, (int(x),int(y)), (last_x, last_y), pencil['color'], pencil['size'])

    return canvas


def canvasMode(image, canvas, camera_mode, paint = None):

    if camera_mode:
        # Overlay canvas drawing on camera (exclude background color with threshold)
        canvas_cam = image.copy()
        canvas_gray= cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_thresh = cv2.threshold(canvas_gray, 254, 255, cv2.THRESH_BINARY)
        canvas_cam[canvas_thresh==0] = canvas[canvas_thresh==0]
        return canvas_cam
    elif paint is not None:
        # Overlay canvas drawing on paint
        canvas_paint = paint.copy()
        canvas_gray= cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, canvas_thresh = cv2.threshold(canvas_gray, 254, 255, cv2.THRESH_BINARY)
        canvas_paint[canvas_thresh==0] = canvas[canvas_thresh==0]
        return canvas_paint
    else:
        return canvas

def calculateAccuracy(canvas, painted_image):

    # convert the images to HSV format
    canvas_hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(canvas_hsv, (0, 50, 70), (9, 255, 255))
    green = cv2.inRange(canvas_hsv, (36, 25, 25), (70, 255, 255))
    blue = cv2.inRange(canvas_hsv, (110, 50, 50), (130, 255, 255))
    painted_hsv = cv2.cvtColor(painted_image, cv2.COLOR_BGR2HSV)
    red_painted = cv2.inRange(painted_hsv, (0, 50, 70), (9, 255, 255))
    green_painted = cv2.inRange(painted_hsv, (36, 25, 25), (70, 255, 255))
    blue_painted = cv2.inRange(painted_hsv, (110, 50, 50), (130, 255, 255))

    # Get the canvas painted areas
    bitwise_and_red = cv2.bitwise_and(red_painted, red)
    bitwise_and_green = cv2.bitwise_and(green_painted, green)
    bitwise_and_blue = cv2.bitwise_and(blue_painted, blue)
    # Get the painted_image painted areas
    bitwise_or_red = cv2.bitwise_or(red, red_painted)
    bitwise_or_green = cv2.bitwise_or(green, green_painted)
    bitwise_or_blue = cv2.bitwise_or(blue, blue_painted)

    # calculate red accuracy
    bitwise_or_red[bitwise_or_red > 0] = 1
    bitwise_and_red[bitwise_and_red > 0] = 1
    red_painted = sum(sum(bitwise_and_red))
    total_red = sum(sum(bitwise_or_red))
    red_accuracy = (red_painted / total_red) * 100

    # calculate green accuracy
    bitwise_or_green[bitwise_or_green > 0] = 1
    bitwise_and_green[bitwise_and_green > 0] = 1
    green_painted = sum(sum(bitwise_and_green))
    total_green = sum(sum(bitwise_or_green))
    green_accuracy = (green_painted / total_green) * 100

    # calculate blue accuracy
    bitwise_or_blue[bitwise_or_blue > 0] = 1
    bitwise_and_blue[bitwise_and_blue > 0] = 1
    blue_painted = sum(sum(bitwise_and_blue))
    total_blue = sum(sum(bitwise_or_blue))
    blue_accuracy = (blue_painted / total_blue) * 100

    # calculate painting accuracy
    painting_accuracy = (blue_painted + green_painted + red_painted) / (total_red + total_blue + total_green) * 100

    print('\n'*50)
    print(Fore.RED + 'Current RED Accuracy: ' + str(red_accuracy) + Style.RESET_ALL)
    print(Fore.GREEN + 'Current GREEN Accuracy: ' + str(green_accuracy) + Style.RESET_ALL)
    print(Fore.BLUE + 'Current BLUE Accuracy: ' + str(blue_accuracy) + Style.RESET_ALL)
    print('Current PAINTING Accuracy: '+ f"{Fore.RED if painting_accuracy < 50 else Fore.GREEN }" + str(painting_accuracy) + Style.RESET_ALL)

'''
def calculateAccuracy(image_to_paint, image_painted, canvas): #TODO: Improve this method
    diff = cv2.absdiff(image_to_paint, image_painted)

    diff = diff.astype(np.uint8)

    ## Get initial difference between the images
    initial_percentage = 100 - ((np.count_nonzero(diff) * 100) / diff.size)

    diff = cv2.absdiff(image_painted, canvas)
    
    diff = diff.astype(np.uint8)

    ## Get final difference between images and subtract the initial difference
    accuracy = (100 - (np.count_nonzero(diff) * 100) / diff.size)
    print(accuracy - initial_percentage)
    accuracy = accuracy - initial_percentage if accuracy - initial_percentage >= 0 else 0
    return (accuracy *100)/(100-initial_percentage) if accuracy <= 100-initial_percentage else 100

    ## Convert to percentage from 0-100%
    accuracy = (accuracy * 100) / (100-initial_percentage)

    return accuracy
'''

#Function for calculating distance between two points in a plane 
def distance(current_location, previous_location):
    return int(math.sqrt(
        math.pow(current_location[0] - previous_location[0], 2) + math.pow(current_location[1] - previous_location[1],
                                                                           2)))

#Used for creation of ellipses
def angle(x1, x2, y1, y2):
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def mouseMove(event, x, y, flags, params, pencil, shape):
    if pencil['use_mouse']:

        img = shape['canvas']
        color = pencil['color']
        thickness = pencil['size']
        start_point = (shape['start_x'], shape['start_y'])

        if event == cv2.EVENT_LBUTTONDOWN:

            # start drawing and save starting point
            shape['drawing'] = True
            shape['start_x'] = x
            shape['start_y'] = y
 
        elif event == cv2.EVENT_MOUSEMOVE:

            # Move pencil position
            pencil['x'] = x
            pencil['y'] = y


            # Draw shapes
            if shape['drawing']:
                copied_image = img.copy()
                if shape['shape'] == 1:
                    cv2.rectangle(copied_image, start_point, (x, y), color, thickness)
                if shape['shape'] == 2:
                    cv2.circle(copied_image, start_point, distance((x, y), start_point), color, thickness)
                if shape['shape'] == 3:
                    cv2.ellipse(copied_image, start_point, (abs(x - start_point[0]), abs(y - start_point[1])),
                                angle(start_point[0], x, start_point[1], y),
                                0., 360, color, thickness)
                shape['canvas_drawing'] = copied_image
            else:
                shape['canvas_drawing'] = img

        elif event == cv2.EVENT_LBUTTONUP:
            shape['drawing'] = False

            if shape['shape'] == 1:
                cv2.rectangle(img, start_point, (x, y), color=color, thickness=thickness)
            if shape['shape'] == 2:
                cv2.circle(img, start_point, distance((x, y), start_point), color, thickness)
            if shape['shape'] == 3:
                cv2.ellipse(img, start_point, (abs(x - start_point[0]), abs(y - start_point[1])), angle(start_point[0], x, start_point[1], y), 0.,
                            360, color,
                            thickness)

            shape['start_x'] = -1
            shape['start_y'] = -1


def drawShape(pencil, shape):
    x = pencil['x']
    y = pencil['y']

    img = shape['canvas']
    color = pencil['color']
    thickness = pencil['size']
    start_point = (shape['start_x'], shape['start_y'])

    copied_image = img.copy()
    if shape['shape'] == 1:
        cv2.rectangle(copied_image, start_point, (x, y), color, thickness)
    if shape['shape'] == 2:
        cv2.circle(copied_image, start_point, distance((x, y), start_point), color, thickness)
    if shape['shape'] == 3:
        cv2.ellipse(copied_image, start_point, (abs(x - start_point[0]), abs(y - start_point[1])),
                    angle(start_point[0], x, start_point[1], y),
                    0., 360, color, thickness)
    shape['canvas_drawing'] = copied_image





def main():

    # ---------------------
    #    Initialization
    # ---------------------

    ### define and read arguments
    json_file, use_shake_prevention, paint = args()

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
    name_largest = 'Largest Component'
    name_canvas = 'Canvas'

    cv2.namedWindow(name_original, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_segmented, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_largest, cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(name_canvas, cv2.WINDOW_AUTOSIZE)

    cv2.moveWindow(name_original, 100, 100)
    cv2.moveWindow(name_segmented, image.shape[1] + 120, 100)
    cv2.moveWindow(name_largest, image.shape[1] + 120, int(image.shape[0]/2) + 230)
    cv2.moveWindow(name_canvas, int(image.shape[1]*1.5 + 190), 140)

    ## Create Blank Canvas
    canvas = np.zeros(image.shape, dtype=np.uint8)
    canvas.fill(255)
    cv2.imshow(name_canvas, canvas)


    if paint:
        paint = paint.lower()
        if paint in ['fire', 'turtle', 'bird']:
            image_to_paint = cv2.imread(f'drawings/{paint}-blank.png')
            image_to_paint = cv2.resize(image_to_paint, (canvas.shape[1], canvas.shape[0]), interpolation = cv2.INTER_AREA)
            painted_image = cv2.imread(f'drawings/{paint}-painted.png')
            painted_image = cv2.resize(painted_image, (canvas.shape[1], canvas.shape[0]), interpolation = cv2.INTER_AREA)
        else:
            print(f'{Fore.RED}Paint not available. Choose one of the following: Fire')
            return


    # Initial pencil properties
    pencil = {  'x': -1,
                'y': -1,
                'last_x': -1,
                'last_y': -1,
                'color': (255, 0, 0),
                'size': 10,
                'use_mouse': False}
    
    shape = {   'shape': 0,
                'drawing': False,
                'start_x': -1,
                'start_y': -1,
                'canvas': canvas,
                'canvas_name': name_canvas,
                'canvas_drawing': canvas}


    cv2.setMouseCallback(name_canvas, partial(mouseMove, pencil=pencil, shape=shape))
    camera_mode = False
    circle = False
    ellipse = False
    square = False

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


        ## Find centroid and draw largest component
        centroid = getCentroid(image_thresholded, name_largest)
        image = drawCentroid(image, centroid[0], centroid[1])
        cv2.imshow(name_original, image)

        if not pencil['use_mouse']:
            pencil['x'], pencil['y'] = centroid


        ## Draw in canvas
        if pencil['x'] != -1 and pencil['y'] != -1 and shape['shape'] == 0:
            canvas = drawLine(canvas, pencil, use_shake_prevention)

        # Save position
        pencil['last_x'] = pencil['x']
        pencil['last_y'] = pencil['y']


        # Draw with centroid
        if shape['shape'] != 0 and not pencil['use_mouse']:
            drawShape(pencil, shape)

        # Use normal canvas or shape from mousecallback
        canvas_updated = shape['canvas_drawing'] if shape['shape'] != 0 else canvas


        # Update image in canvas (check camera mode)
        final_image = canvasMode(image, canvas_updated, camera_mode, paint = image_to_paint if paint else None)

        cv2.imshow(name_canvas, final_image)
        if paint:
            calculateAccuracy(final_image, painted_image)
            #accuracy = calculateAccuracy(image_to_paint, painted_image, final_image)
            #print(f'Accuracy: {accuracy}%')

        # Keyboard inputs
        key = cv2.waitKey(10) & 0xFF        # Only read last byte (prevent numlock)
    
        if key == ord('q') or key == 27:  # q or ESC - quit without saving
            break

        elif key == ord('c'): # c - clear the canvas
            canvas.fill(255)

        elif key == ord('w'): # w - save the current canvas
            drawing_filename = f"drawing_{ctime().replace(' ','_')}.png"
            cv2.imwrite(drawing_filename, final_image)

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
            square = not square
            if square:
                shape['shape'] = 1
                shape['start_x'] = pencil['x']
                shape['start_y'] = pencil['y']
            elif not square:
                canvas = shape['canvas_drawing']
                shape['canvas'] = canvas
                shape['shape'] = 0

        elif key == ord('e'): # e - draw an ellipse
            ellipse = not ellipse
            if ellipse:
                shape['shape'] = 3
                shape['start_x'] = pencil['x']
                shape['start_y'] = pencil['y']
            if not ellipse:
                canvas = shape['canvas_drawing']
                shape['canvas'] = canvas
                shape['shape'] = 0

        elif key == ord('o'): # o - draw a circle
            circle = not circle
            if circle:
                shape['shape'] = 2
                shape['start_x'] = pencil['x']
                shape['start_y'] = pencil['y']
            elif not circle:
                canvas = shape['canvas_drawing']
                shape['canvas'] = canvas
                shape['shape'] = 0

        elif key == ord('m'): # m - toggle camera Mode
            camera_mode = False if camera_mode else True

        elif key == ord('n'): # n - toggle mouse input
            pencil['use_mouse'] = False if pencil['use_mouse'] else True

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()