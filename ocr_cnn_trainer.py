import numpy as np 
import cv2 
import os 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt 
from keras.preprocessing.image import ImageDataGenerator as idg
from keras.utils.np_utils import to_categorical 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D,MaxPooling2D 
from tensorflow.keras.layers import Dropout,Flatten, Dense 
from tensorflow.keras.optimizers import Adam 

import pickle

path = 'myData'
testRatio= 0.2
validationRatio = 0.2
images = []
class_num = []
myList = os.listdir(path)
num_of_Classes = len(myList)
image_dimesions = (32,32,3)
batch_size_val = 50 
epochs_val=10
# stepsper_epoch=2000


for i in range(0, num_of_Classes):
    myPictureList = os.listdir(path+"/"+str(i))
    print('EACH PICS: ',len(myPictureList))
    for j in myPictureList:
        current_image = cv2.imread(path+"/"+str(i)+"/"+j)
        #resize them to save computational power when building
        current_image = cv2.resize(current_image, (32,32)) 
        images.append(current_image)
        class_num.append(i)

images = np.array(images)
class_num = np.array(class_num)

# split the data  test_size=20% test 80% training 
#we need an external library to shuffle and test accordingly 
# otherwise, 80% would have been 0-7 of the dataset.
X_train,X_test,y_train, y_test = train_test_split(images,class_num,test_size=testRatio)
X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=validationRatio)
print(X_train.shape) 
print(X_test.shape)
print(X_validation.shape)

# we are locating indexes with from class 0.
#keeps number of samples from each class
num_of_samples=[]
for sample in range(0,num_of_Classes):
    num_of_samples.append(len(np.where(y_train==sample)[0]))

print(num_of_samples)
# plt.figure(figsize=(10,5))
# plt.bar(range(0,num_of_Classes),num_of_samples)
# plt.title("Number of Images in each class")
# plt.xlabel("Class ID")
# plt.ylabel("Number of Images")
# plt.show()

def pre_process_image(image):
    image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # distributes light in picture evenly
    image = cv2.equalizeHist(image)
    image = image/255
    return image 

# preproccess all images in X_train and store the
# processed images back in X_train
X_train = np.array(list(map(pre_process_image,X_train)))
X_test = np.array(list(map(pre_process_image,X_test)))
X_validation = np.array(list(map(pre_process_image,X_validation)))
#since the CNN needs depth we add it. 
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)
# to make the data set more generic do some vairations 
# some of which are zoom,rotation and so on.
data_generation = idg(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,rotation_range=10)
print('x trainnn: ',len(X_train))
data_generation.fit(X_train)
y_train = to_categorical(y_train,num_of_Classes)
y_test = to_categorical(y_test,num_of_Classes)
y_validation = to_categorical(y_validation,num_of_Classes) 
def my_model():
    num_filters=60 
    size_of_filter1=(5,5)
    size_of_filter2 = (3,3)
    size_of_pool=(2,2)
    num_of_nodes = 500 
    model = Sequential() 
    model.add((Conv2D(num_filters,size_of_filter1,input_shape=(image_dimesions[0],image_dimesions[1],1),activation='relu')))
    model.add((Conv2D(num_filters,size_of_filter1,activation='relu')))

    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add((Conv2D(num_filters//2,size_of_filter2,activation='relu')))
    model.add((Conv2D(num_filters//2,size_of_filter2,activation='relu')))

    model.add(MaxPooling2D(pool_size=size_of_pool))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_of_nodes, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_of_Classes,activation='softmax'))
    model.compile(Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    return model

model = my_model() 
print(model.summary())
stepsper_epoch=len(X_train)//batch_size_val

history = model.fit(data_generation.flow(X_train,y_train,batch_size=batch_size_val),
steps_per_epoch=stepsper_epoch,epochs=epochs_val,
validation_data=(X_validation,y_validation),shuffle=1)

plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')
plt.show()

score = model.evaluate(X_test,y_test,verbose=0)
print('Test Score = ', score[0])
print('Test Accuracy =', score[1])

model.save('/Users/mubarekabdela/Desktop/Fall2020Projects/Sudoku')

