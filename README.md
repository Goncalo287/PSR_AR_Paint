# PSR Assignment 2 - AR PAINT
## P1 - Group 2
GONÇALO IVAN RAMOS ANACLETO 93394

HENRIQUE CARVALHO SOUSA 98324

JOÃO PEDRO PEREIRA TOMÁS 93366

## Introduction
This applications allows the user to draw on an image by moving a colored object in front of the laptop's camera.

The tools used for this effect are the ones that were made for the past classes (python, opencv, etc).

We developed two separate scripts, one called color_segmenter.py, where we can choose the color range of the colored object to be tracked and the other called ar_paint.py that allows the user to drwan on the image.

## Features

* Reading command line arguments.
* Reading the json file with the limits for color segmentation
* Setup video capture
* Creating a white image (canvas) to draw the same size as the images all received from the camera
* Getting a mask with the pixels with the chosen color
* Getting only the largest component from the mask
* Calculating and defining the centroid of the largest component on the camera
* Highlighting the largest component on the camera
* Using the centroid to paint on the canvas
* Change pencil properties with key bindings
* ADVANCED FUNCIONALITIES
  - Use Shake Detection
  - Camera Mode to use video stream as the canvas background
  - Draw Shapes on the canvas
  - Numbered painting to be colored
  - Evaluation of the picture paiting
* EXTRAS:
  - The user can click on the camera while in the color segmenter to choose the limits
  - Getting the evaluation of the picture for each of the colors
  - Can draw with the mouse with a toggle

## How To Use

To clone and run this application, you'll need [Git](https://git-scm.com) and the packages that are used on our python scripts.

```bash
# Clone this repository
$ git clone https://github.com/Goncalo287/PSR_AR_Paint.git

# Go into the repository
$ cd PSR_AR_Paint

# Run the color segmenter
$ python color_segmenter.py

# Run the program to paint on an image
$ python ar_paint.py [args]

# You can run this to understand which arguments it accepts
$ python ar_paint.py --help
```

## Color Segmenter Key Bindings
* w / ENTER
  - Save and exit
* q / ESC
  - Exit without saving
* Mouse Left Click
  - Pick color in camera

## AR Paint Key Bindings
* w / ENTER
  - Save current canvas on file
* q / ESC
  - Exit without saving
* c
  - Clear the canvas
* r
  - Change pencil color to red
* g
  - Change pencil color to green
* '+'
  - Increase pencil size by 1
* '-'
  - Decrease pencil size by 1
* s
  - Press one time to start drawing a square. Press the second time to stop drawing
* o
  - Press one time to start drawing a circle. Press the second time to stop drawing
* e
  - Press one time to start drawing an ellipse. Press the second time to stop drawing
* m
  - Toggle Camera Mode
* n
  - Toggle Mouse Input and stop Shape Drawing



















