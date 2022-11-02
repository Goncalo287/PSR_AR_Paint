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

init(autoreset=True)        # Initialize colorama


def args():
    '''
    Define and parse command line arguments

    Returns:
        str: Path to json file, bool: Use Shake Detection Activation, bool: Paint mode Activation
    '''
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
    '''
    Read json file

    Args:
        filename (str): Path to json file

    Returns:
        dict: Minimum and maximum color values
    '''
    with open(filename, 'r') as f:
        limits = json.load(f)
    return limits['limits']


def getLimits(limits):
    '''
    Get Colors Maximum and Minimum values

    Args:
        limits (dict): Dictionary with color limits

    Returns:
        list: Minimum value of colors
        list: Maximum value of colors
    '''
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
    '''
    Show resized image

    Args:
        name (str): Window name
        image (numpy.ndarray): Image to Show
        resize (float): Resize scale
    '''

    (h, w) = image.shape[:2]

    h = int(h * resize)
    w = int(w * resize)

    image = cv2.resize(image, (w, h))
    cv2.imshow(name, image)


def getCentroid(image_thresh, name_largest):
    '''
    Get largest component centroid coordinates

    Args:
        image_thresh (numpy.ndarray): thresholded image
        name_largest (str): window name

    Returns:
        float: x coordinate
        float: y coordinate
    '''
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
    '''
    Draw centroid on image

    Args:
        image (numpy.ndarray): Current Webcam Image
        x (float): Centroid X coordinate
        y (float): Centroid Y coordinate

    Returns:
        numpy.ndarray: Image with an cross drawn on centroid
    '''
    if x != -1 and y != -1:
        # Drawing cross on the webcam feed
        cv2.line(image, (int(x-10), int(y)), (int(x+10), int(y)), (0, 0, 255), 1)
        cv2.line(image, (int(x), int(y-10)), (int(x), int(y+10)), (0, 0, 255), 1)            

    return image


def drawLine(canvas, pencil, usp):
    '''
    Draw line on canvas

    Args:
        canvas (numpy.ndarray): Canvas image
        pencil (dict): Pencil Properties
        usp (bool): Use Shake Prevention Mode

    Returns:
        numpy.ndarray: Canvas with line drawn
    '''
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
    '''
    Compose a final image depending on the mode selected by the user

    Args:
        image (numpy.ndarray): Camera Image
        canvas (numpy.ndarray): Canvas Image
        camera_mode (bool): Camera Mode
        paint (bool, optional): Paint Mode. Defaults to None.

    Returns:
        numpy.ndarray: Composed Image
    '''
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


def getColorAccuracy(bitwise_and, bitwise_or):
    '''
    Calculate color accuracy

    Args:
        bitwise_and (_type_): _description_
        bitwise_or (_type_): _description_

    Returns:
        float: color accuracy
        float: painted pixels
        float: total painted pixels
    '''
    bitwise_and[bitwise_and > 0] = 1
    bitwise_or[bitwise_or > 0] = 1
    color_painted = sum(sum(bitwise_and))
    total_color = sum(sum(bitwise_or))
    color_accuracy = (color_painted / total_color) * 100

    return color_accuracy, color_painted, total_color


def calculateAccuracy(canvas, painted_image):
    '''
    Calculate Paiting Accuracy

    Args:
        canvas (numpy.ndarray): Canvas Image
        painted_image (numpy.ndarray): Original Painted Image
    '''

    ## Define lower and upper color values
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])

    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])

    lower_green = np.array([50,100,100])
    upper_green = np.array([70,255,255])

    ## Create masks for the colors on canvas
    canvas_hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV)
    red = cv2.inRange(canvas_hsv, lower_red, upper_red)
    green = cv2.inRange(canvas_hsv, lower_green, upper_green)
    blue = cv2.inRange(canvas_hsv, lower_blue, upper_blue)

    ## Create masks for the colors on the painted image
    painted_hsv = cv2.cvtColor(painted_image, cv2.COLOR_BGR2HSV)
    red_painted = cv2.inRange(painted_hsv, lower_red, upper_red)
    green_painted = cv2.inRange(painted_hsv, lower_green, upper_green)
    blue_painted = cv2.inRange(painted_hsv, lower_blue, upper_blue)

    red_accuracy, red_painted, total_red = getColorAccuracy(cv2.bitwise_and(red_painted, red), cv2.bitwise_or(red, red_painted))
    green_accuracy, green_painted, total_green = getColorAccuracy(cv2.bitwise_and(green_painted, green), cv2.bitwise_or(green, green_painted))
    blue_accuracy, blue_painted, total_blue = getColorAccuracy(cv2.bitwise_and(blue_painted, blue), cv2.bitwise_or(blue, blue_painted))

    # calculate painting accuracy
    painting_accuracy = (blue_painted + green_painted + red_painted) / (total_red + total_blue + total_green) * 100

    print('\n'*50)
    print(Fore.RED + 'Current RED Accuracy: ' + str(red_accuracy) + Style.RESET_ALL)
    print(Fore.GREEN + 'Current GREEN Accuracy: ' + str(green_accuracy) + Style.RESET_ALL)
    print(Fore.BLUE + 'Current BLUE Accuracy: ' + str(blue_accuracy) + Style.RESET_ALL)
    print('Current PAINTING Accuracy: '+ f"{Fore.RED if painting_accuracy < 50 else Fore.GREEN }" + str(painting_accuracy) + Style.RESET_ALL)


def distance(current_location, previous_location):
    '''
    Calculate distance between two points in a plane

    Args:
        current_location (tuple): X and Y coordinates of the current position
        previous_location (tuple): X and Y coordinates of the previous position

    Returns:
        int: distance between the two points
    ''' 

    return int(math.sqrt(
        math.pow(current_location[0] - previous_location[0], 2) + math.pow(current_location[1] - previous_location[1],
                                                                           2)))

def drawShape(image, pencil, shape):
    '''
    Draws shapes in image
    Args:
        image (numpy.ndarray): where the shapes are drawn
        pencil (dict): dictionary with pencil properties
        shape (dict): dictionary with shape properties

    Returns:
        numpy.ndarray: Image with the shapes drawn
    '''

    x = pencil['x']
    y = pencil['y']
    color = pencil['color']
    thickness = pencil['size']
    start_point = (shape['start_x'], shape['start_y'])

    if shape['id'] == 1:
        cv2.rectangle(image, start_point, (x, y), color, thickness)

    elif shape['id'] == 2:
        cv2.circle(image, start_point, distance((x, y), start_point), color, thickness)

    elif shape['id'] == 3:
        size_x = (x - start_point[0])//2
        size_y = (y - start_point[1])//2
        center_x = start_point[0] + size_x
        center_y = start_point[1] + size_y

        cv2.ellipse(image, (center_x, center_y), (abs(size_x), abs(size_y)),
                    0,
                    0., 360, color, thickness)
    return image


def mouseMove(event, x, y, flags, params, pencil, shape):
    '''
    Mouse Move Event

    Args:
        event, x, y, flags, params: Default mouseCallBack parameters
        pencil (dict): dictionary with pencil properties
        shape (dict): dictionary with shape properties
    Returns:
        nothing, input dictionaries are updated instead    
    '''

    if pencil['use_mouse']:

        if event == cv2.EVENT_LBUTTONDOWN:

            # start drawing and save starting point
            shape['drawing'] = True
            shape['start_x'] = x
            shape['start_y'] = y
 
        elif event == cv2.EVENT_MOUSEMOVE:

            # Move pencil position
            pencil['x'] = x
            pencil['y'] = y

            img = shape['canvas']

            # Draw shapes
            if shape['drawing']:
                copied_image = img.copy()
                shape['canvas_drawing'] = drawShape(copied_image, pencil, shape)
            else:
                shape['canvas_drawing'] = img

        elif event == cv2.EVENT_LBUTTONUP:
            shape['drawing'] = False

            shape['canvas'] = shape['canvas_drawing']
            shape['start_x'] = -1
            shape['start_y'] = -1


def selectShape(pencil, shape, new_id):
    '''
    Select shape to be drawn

    Args:
        pencil (dict): dictionary with pencil properties
        shape (dict): dictionary with shape properties
        new_id (int): new shape id

    Returns:
        int: Id of the shape to be drawn 
    '''

    # If already drawing this shape, save it to canvas
    if shape['id'] == new_id:
        shape['canvas'] = shape['canvas_drawing']
        shape['id'] = 0

    # If drawing something else, stop it and start this shape
    else:
        shape['id'] = new_id
        shape['start_x'] = pencil['x']
        shape['start_y'] = pencil['y']
    
    # Return updated canvas
    return shape['canvas']



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
        try:
            image_to_paint = cv2.imread(f'drawings/{paint}-blank.png')
            image_to_paint = cv2.resize(image_to_paint, (canvas.shape[1], canvas.shape[0]), interpolation = cv2.INTER_AREA)
            painted_image = cv2.imread(f'drawings/{paint}-painted.png')
            painted_image = cv2.resize(painted_image, (canvas.shape[1], canvas.shape[0]), interpolation = cv2.INTER_AREA)
        except FileNotFoundError:
            print(f'{Fore.RED}Paint not available. Choose one of the following: Fire, Turtle, Bird')
            exit()


    # Initial pencil properties
    pencil = {  'x': -1,
                'y': -1,
                'last_x': -1,
                'last_y': -1,
                'color': (255, 0, 0),
                'size': 10,
                'use_mouse': False}
    
    shape = {   'id': 0,
                'drawing': False,
                'start_x': -1,
                'start_y': -1,
                'canvas': canvas,
                'canvas_name': name_canvas,
                'canvas_drawing': canvas}


    cv2.setMouseCallback(name_canvas, partial(mouseMove, pencil=pencil, shape=shape))
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


        ## Find centroid and draw largest component
        centroid = getCentroid(image_thresholded, name_largest)
        image = drawCentroid(image, centroid[0], centroid[1])
        cv2.imshow(name_original, image)


        # Save centroid coordinates to pencil properties. if use_mouse == True, this is handled by setMouseCallback
        if not pencil['use_mouse']:
            pencil['x'], pencil['y'] = centroid


        ## Draw in canvas
        if pencil['x'] != -1 and pencil['y'] != -1:

            # Normal pencil
            if shape['id'] == 0:
                canvas = drawLine(canvas, pencil, use_shake_prevention)

            # Draw shapes, if use_mouse == True, this is handled by setMouseCallback
            else:
                if not pencil['use_mouse'] and shape['start_x'] != -1 and shape['start_y'] != -1:
                    shape['canvas_drawing'] = drawShape(shape['canvas'].copy(), pencil, shape)


        # Pick a canvas to show: normal or drawing shape (temporary shapes on top of canvas)
        canvas = shape['canvas']
        canvas_updated = shape['canvas_drawing'] if shape['id'] != 0 else canvas


        # Update image in canvas (check camera mode and paint mode)
        final_image = canvasMode(image, canvas_updated, camera_mode, paint = image_to_paint if paint else None)
        cv2.imshow(name_canvas, final_image)


        # Calculate accuracy
        if paint:
            calculateAccuracy(final_image, painted_image)


        # Save current pencil position to use in the next cycle
        pencil['last_x'] = pencil['x']
        pencil['last_y'] = pencil['y']


        # Keyboard inputs
        key = cv2.waitKey(10) & 0xFF        # Only read last byte (prevent numlock)
    
        if key == ord('q') or key == 27:  # q or ESC - quit without saving
            break

        elif key == ord('c'): # c - clear the canvas and stop shape drawing
            canvas.fill(255)
            shape['drawing'] = False
            shape['id'] = 0
            shape['start_x'], shape['start_y'] = -1, -1
            shape['canvas_drawing'] = canvas

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
            canvas = selectShape(pencil, shape, 1)

        elif key == ord('o'): # o - draw a circle
            canvas = selectShape(pencil, shape, 2)

        elif key == ord('e'): # e - draw an ellipse
            canvas = selectShape(pencil, shape, 3)

        elif key == ord('m'): # m - toggle camera Mode
            camera_mode = not camera_mode

        elif key == ord('n'): # n - toggle mouse input and stop shape drawing
            pencil['use_mouse'] = not pencil['use_mouse']
            shape['drawing'] = False
            shape['id'] = 0
            shape['start_x'], shape['start_y'] = -1, -1
            shape['canvas_drawing'] = canvas

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()