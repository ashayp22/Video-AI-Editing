from keras.models import load_model
import cv2
import numpy as np
from datetime import datetime


#load in the models
model = load_model('model83.h5') #convolutional neural network for classifying grin

#haar cascades for finding the mouth and face
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_mcs_mouth.xml')

#get the input
cap = cv2.VideoCapture('inputvideo.mp4')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.mp4',fourcc, 30, (int(cap.get(3)), int(cap.get(4))))

#Read the Brown Coat of Arms and resize
coat = cv2.imread("coat.png")
coat = cv2.resize(coat, (0,0), fx=1.65, fy=1.65)
print(coat.shape)
num = 0

#Start reading the video
while cap.isOpened():
    ret, img = cap.read()

    #start the pre-processing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale

    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #gets the parts of the image that contains a face

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2) #draw a rectangle
        roi_gray = gray[y:y + h, x:x + w]  # gets the region
        roi_color = img[y:y + h, x:x + w]
        mouth = smile_cascade.detectMultiScale(roi_gray)

        # boundaries for the actual mouth
        good_x = -1
        good_y = -1
        good_w = -1
        good_h = -1

        for (ex, ey, ew, eh) in mouth:
            if ey > good_y:  # the boundary containing the actual mouth will be as low as possible on the face
                good_x = ex
                good_y = ey
                good_w = ew
                good_h = eh

        if good_x == -1:
            continue

        #obtain the mouth
        mouth = roi_gray[good_y: good_y + good_h, good_x:good_x + good_w]

        #resize the mouth to fit the cnn's input size
        mouth = cv2.resize(mouth, (52, 32))

        mouth = np.array(mouth)

        #reshape the mouth
        mouth = mouth.reshape((1, 32, 52, 1))
        mouth = mouth.astype("float32") / 255

        #input into cnn
        val = model.predict(mouth)
        val = val[0]

        y = np.where(val == np.amax(val))[0][0] #obtain the output

        txt = ''

        #if y is 0, there isn't a grin detected
        if y == 0:
            txt = 'no grin'
        else:
            txt = ' grin'
            mult = 3

            #a grin has been detected, so we must show more of the grin
            if num < coat.shape[1]:
                if num + mult < coat.shape[1]:
                    num += mult
                else:
                    num = coat.shape[1]

        #place a rectangle over the mouth
        cv2.rectangle(roi_color, (good_x, good_y), (good_x + good_w, good_y + good_h), (0, 255, 0), 2)

        #label no grin or grin over the mouth's bounding box
        cv2.putText(img, txt, (x+150, h+175), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255), lineType=cv2.LINE_AA)

    #add logo

    if num > 0:

        #here, we are adding the coats in all four corners. We use a addWeight calculation
        # where dst = x * alpha + y * beta + gamma, where x is the original, y is the coat, alpha is 1, and beta is 1

        top_left_coat = cv2.addWeighted(img[0:coat.shape[0], 0:num, :], 1, coat[0:coat.shape[0], 0:num, :], 1, 0)
        img[0:coat.shape[0], 0:num] = top_left_coat

        bottom_left_coat = cv2.addWeighted(img[img.shape[0] - coat.shape[0]:img.shape[0], 0:num, :], 1, coat[0:coat.shape[0], 0:num, :], 1, 0)
        img[img.shape[0] - coat.shape[0]:img.shape[0], 0:num] = bottom_left_coat

        top_right_coat = cv2.addWeighted(img[0:coat.shape[0], img.shape[1] - coat.shape[1]:img.shape[1] - coat.shape[1] + num, :], 1,
                                      coat[0:coat.shape[0], 0:num, :], 1, 0)
        img[0:coat.shape[0], img.shape[1] - coat.shape[1]:img.shape[1] - coat.shape[1] + num] = top_right_coat

        bottom_right_coat = cv2.addWeighted(img[img.shape[0] - coat.shape[0]:img.shape[0], img.shape[1] - coat.shape[1]:img.shape[1] - coat.shape[1] + num, :], 1,
                                      coat[0:coat.shape[0], 0:num, :], 1, 0)
        img[img.shape[0] - coat.shape[0]:img.shape[0], img.shape[1] - coat.shape[1]:img.shape[1] - coat.shape[1] + num] = bottom_right_coat


    cv2.imshow('img', img)

    out.write(img)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()