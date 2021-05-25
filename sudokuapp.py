import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import cv2
import numpy as np

import sudokusolver
from tensorflow.keras.models import load_model
from image_processing import * 

def initialize_digit_prediction():
    detection_model = load_model("myModel.h5")
    return detection_model


pathImage = "/Users/mubarekabdela/Desktop/Fall2020Projects/Sudoku/download.jpeg"

model = initialize_digit_prediction()


def solve(pathImage):
    heightImg = 450
    widthImg = 450
    #prep the image
    # image = cv2.imread(pathImage)
    image = cv2.imread(pathImage)
    image = cv2.resize(image, (widthImg, heightImg))
    imageBlank = np.zeros((heightImg, widthImg,3), np.uint8) # blank test for debugging ?
    imageThreshold = pre_process(image)

    imageContours = image.copy()
    imageBigContour= image.copy()
    contours, hierarchy = cv2.findContours(imageThreshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imageContours, contours, -1, (0, 255, 0), 3)
    biggest_square, max_area = biggest_contour(contours)
    if biggest_square.size != 0:
        #preparing and sorting image for warping as corners may not
        # be ordered accordingly
        biggest_square = reorder_points(biggest_square)
        # draws contours at square of the sudoku
        cv2.drawContours(imageBigContour, biggest_square, -1, (0, 0, 255),25)
        point1 = np.float32(biggest_square) # prepare the points/coordinates
        # strctures all the points at each corner accordingly.
        point2 = np.float32([[0, 0],[widthImg, 0], [0,heightImg],[widthImg, heightImg]])
        matrix = cv2.getPerspectiveTransform(point1, point2)
        imageWarp = cv2.warpPerspective(image, matrix, (widthImg, heightImg))
        imageDetectedDigits = imageBlank.copy()
        imageWarpColored = cv2. cvtColor(imageWarp, cv2.COLOR_BGR2GRAY)

        image_solved_digits = imageBlank.copy()
        boxes = split_boxes(imageWarpColored)
        numbers = get_prediction(boxes, model)
        # I can probably check if the list of numbers exist in a db
        # if they do, I can return the solved board
        # this can optimize redundant computations
        imageDetectedDigits = display_numbers(imageDetectedDigits, numbers, color=(255, 0, 255))
        numbers = np.asarray(numbers)
        # if the number is greater than 0 it will put 0 
        # otherwise, it will put in the same position. 
        position_array = np.where(numbers > 0, 0,1)
        sudoku_board = np.array_split(numbers,9)
        try:
            sudokusolver.solve(sudoku_board)
        except:
            pass
        # change from np array to regular list
        regular_list = []
        for rows in sudoku_board:
            for element in rows:
                regular_list.append(element)
    
        # this eliminates overlaying of elements that 
        # are already populated in our sudoku board 
        # when we populate input values from the board with zero 
        # it allows us to overlay only non-populated grids.
        input_numbers = regular_list * position_array 
        image_solved_digits = display_numbers(image_solved_digits, input_numbers)

        # prepare the points and fit the solved board
        # to the original picture 
        first_point = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
        second_point = np.float32(biggest_square)

        matrix = cv2.getPerspectiveTransform(first_point, second_point)
        image_in_warp_colored = image.copy() 
        image_in_warp_colored = cv2.warpPerspective(image_solved_digits, matrix, (widthImg, heightImg))
        weight_perspective = cv2.addWeighted(image_in_warp_colored, 1, image, 0.5, 1)
        imageDetectedDigits = draw_grid(imageDetectedDigits)
        imageDetectedDigits = draw_grid(imageDetectedDigits)
        return [[image,imageThreshold,imageContours,imageBigContour],
        [imageDetectedDigits,image_solved_digits,image_in_warp_colored,weight_perspective]]



# imageArray = ([image,imageThreshold,imageContours,imageBigContour],
# [imageDetectedDigits,image_solved_digits,image_in_warp_colored,weight_perspective])
imageArray = solve(pathImage)
stackedImage = stack_images(imageArray, 1)
cv2.imshow('Stacked Images', stackedImage)
cv2.waitKey(0)