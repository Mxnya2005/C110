# import the opencv library
import cv2
import tensorflow as tf
import numpy as np


model=tf.keras.models.load_model("keras_model.h5")

# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame by frame
    ret, frame = vid.read()
    img=cv2.resize(frame,(224,224)) #the way model learnt it (224,224)
    test_Img=np.array(img,dtype=np.float32) #img converted to this array,indivisual values should be of float32 type
    test_Img=np.expand_dims(test_Img,axis=0) #converting 3d into 4d, expand the dimentions of array
    normalised_img=test_Img/255
    prediction=model.predict(normalised_img)
    print("Prediction",prediction)
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # Quit window with spacebar
    key = cv2.waitKey(1)
    
    if key == 32:
        break
  
# After the loop release the cap object
vid.release()

# Destroy all the windows
cv2.destroyAllWindows()