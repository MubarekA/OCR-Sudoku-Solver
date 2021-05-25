import numpy as np 
import cv2  
#Loading the model back:
from tensorflow import keras
model = keras.models.load_model('/Users/mubarekabdela/Desktop/Fall2020Projects/Sudoku')
width= 640 
height = 480 

cap = cv2.VideoCapture(1)
cap.set(3, width)
cap.set(4, height)

def pre_process_image(image):
    image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # distributes light in picture evenly
    image = cv2.equalizeHist(image)
    image = image/255
    return image 

while True:
    succeess,imageOriginal= cap.read()
    img = np.asarray(imageOriginal)
    img = cv2.resize(img(320,320)) 
    img = pre_process_image(img)
    cv2.imshow("processed Image",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break 

