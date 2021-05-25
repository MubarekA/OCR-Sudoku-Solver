import cv2
import numpy as np
# preprocess the image
def pre_process(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert IMage to Gray scale
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 1) # Add Gaussian blur
    image_threshold = cv2.adaptiveThreshold(image_blur,255,1,1,11,2)#apply adaptivve threshold
    # src,maxvalue,adaptive method, thrsholdtype, blocksize, C, d
    return image_threshold 

# stack all the images in one window
# noinspection PyPep8Naming
def stack_images(Array_of_image, scale):
    rows = len(Array_of_image)
    columns = len(Array_of_image[0])
    # isinstance check if its an instance of an object
    rowsAvailable = isinstance(Array_of_image[0], list)
    width = Array_of_image[0][0].shape[1]
    height = Array_of_image[0][0].shape[0]
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,columns):
                Array_of_image[x][y]= cv2.resize(Array_of_image[x][y], (0, 0), None, scale, scale)
                if len(Array_of_image[x][y].shape)== 2:
                    #come back to this part
                    Array_of_image[x][y]=cv2.cvtColor(Array_of_image[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(Array_of_image[x])
            hor_con[x] = np.concatenate(Array_of_image[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            Array_of_image[x] = cv2.resize(Array_of_image[x], (0, 0), None, scale, scale)
            if len(Array_of_image[x].shape) == 2:
                Array_of_image[x] = cv2.cvtColor(Array_of_image[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(Array_of_image)
        hor_con= np.concatenate(Array_of_image)
        ver = hor
    return ver

def biggest_contour(given_contours):
    biggest_grid = np.array([])
    max_given_area = 0
    for contour in given_contours:
        # check each area of contour
        #if its area is small its most
        #likey noise.
        area = cv2.contourArea(contour)
        if area > 50:
            permieter =  cv2.arcLength(contour, True)
            # find its corners by appoximating poly count count
            corners = cv2.approxPolyDP(contour, 0.02 * permieter, True)
            #if its the max area and its a square/rectangle
            # it will replace and return the max
            if area > max_given_area and len(corners)==4:
                biggest_grid = corners
                max_given_area=area
    # return corner points and maximum area of their contours
    return biggest_grid, max_given_area

def reorder_points(corner_points):
    corner_points = corner_points.reshape((4, 2))
    corner_points_new = np.zeros((4, 1, 2), dtype=np.int32)
    add = corner_points.sum(1)
    # lowest value will be 0,0 or top left of our
    # square
    corner_points_new[0] = corner_points[np.argmin(add)]
    # the max value will be our bottom right of our square
    corner_points_new[3] = corner_points[np.argmax(add)]
    difference = np.diff(corner_points, axis=1)
    # the positive val in diff will be the top right side
    #while -ve val is bottom left
    corner_points_new[1]= corner_points[np.argmin(difference)]
    corner_points_new[2]= corner_points[np.argmax(difference)]
    return corner_points_new

def split_boxes(image):
    rows = np.vsplit(image,9)
    all_boxes = []
    for row in rows:
        columns = np.hsplit(row,9)
        for box in columns:
            all_boxes.append(box)
    return all_boxes


def get_prediction(boxes, model):
    one_dimensional_suoku = []
    for image_element in boxes:
        # preproccessing image
        image_element = np.asarray(image_element)
        image_element = image_element[4:image_element.shape[0] - 4, 4:image_element.shape[1] - 4]
        image_element = cv2.resize(image_element, (28, 28))
        image_element = image_element / 255
        image_element = image_element.reshape(1, 28, 28, 1)
        prediction = model.predict(image_element)
        element = np.argmax(prediction, axis=-1)
        highest_probable_value = np.amax(prediction)
        # if the prediction is higher than 80%
        # its most likely the right digit
        # in the case that its lower than 80%
        # we can safely assume that its an empty grid
        # in our sudoku board and will assume its zero
        if highest_probable_value > 0.8:
            one_dimensional_suoku.append(element[0])
        else:
            one_dimensional_suoku.append(0)
    # this returns the elements in the board as a 1d array
    # len of the list should be 81
    return one_dimensional_suoku

def display_numbers(image, numbers, color= (0,255,0)):
    width = int(image.shape[1]/9)
    height = int(image.shape[0]/9)
    for row in range(0,9):
        for column in range(0,9):
            if numbers[(column*9)+row] != 0:
                cv2.putText(image, str(numbers[(column * 9)+row]),
                (row*width+int(width/2)-10, int((column+0.8)*height)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                2, color, 2, cv2.LINE_AA)
    return image
    
def draw_grid(image):
    Width = int(image.shape[1]/9)
    Height = int(image.shape[0]/9)
    for i in range (0,9):
        point_one = (0,Height*i)
        point_two = (image.shape[1],Height*i)
        point_three = (Width * i, 0)
        point_four = (Width*i,image.shape[0])
        cv2.line(image, point_one, point_two, (255, 255, 0),2)
        cv2.line(image, point_three, point_four, (255, 255, 0),2)
    return image