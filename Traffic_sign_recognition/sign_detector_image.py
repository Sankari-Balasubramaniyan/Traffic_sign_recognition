import cv2 as cv
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model

img = None
outputs = None
images = []

classes = ['prohibitory', 'danger', 'mandatory', 'other']
CNN_classes = ["Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)","Speed limit (120km/h)","No passing","No passing for vehicles over 3.5 metric tons","Right-of-way at the next intersection","Priority road","Yield","Stop","No vehicles","Vehicles over 3.5 metric tons prohibited","No entry","General caution","Dangerous curve to the left","Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow","Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory","End of no passing","End of no passing by vehicles over 3.5 metric tons"]

#Load the configuration and weights of the model
net = cv.dnn.readNetFromDarknet('model/yolov3-my-tiny.cfg', 'model/yolov3-my-tiny_final.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

#Get the final detector layer
last_layer = net.getLayerNames()
last_layer = [last_layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
font = cv.FONT_HERSHEY_SIMPLEX

CNN = load_model("model/classifier_v29.h5")

def load_image(direc):
    for imagename in os.listdir(direc):
        img = cv.imread(os.path.join(direc,imagename))
        if img is not None:
            images.append(img)
    return images

def main():
    all_images = load_image('images')
    count = 0
    for i in range(len(all_images)):
        img = all_images[i]
        H, W = img.shape[:2]
        
        #Convert image to blob before feeding into the model
        img1 = img.copy()
        blob = cv.dnn.blobFromImage(img, 1/255.0, (288, 288), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward(last_layer)
        outputs = np.vstack(outputs)

        bounding_boxes = []
        conf = []
        class_num = []

        confidence_threshold = 0.5

        for out in outputs:
            scores = out[5:]
            class_no = np.argmax(scores)
            confidence = scores[class_no]
            if confidence > confidence_threshold:
                x, y, w, h = out[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                bounding_boxes.append([*p0, int(w), int(h)])
                conf.append(float(confidence))
                class_num.append(class_no)

        indices = cv.dnn.NMSBoxes(bounding_boxes, conf, confidence_threshold, confidence_threshold-0.1)
        if len(indices) > 0:
            for j in indices.flatten():
                (x, y) = (bounding_boxes[j][0], bounding_boxes[j][1])
                (w, h) = (bounding_boxes[j][2], bounding_boxes[j][3])
                crop_img = img[y-1:y+h+1, x-1:x+w+1]
                if len(crop_img) > 0:
                    crop_img = cv.resize(crop_img, (32, 32))
                    crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2RGB)
                    crop_img =  crop_img.reshape(-1,32,32,3)
                    prediction = np.argmax(CNN.predict(crop_img))
                    label = str(CNN_classes[prediction])
                cv.rectangle(img1, (x, y), (x + w, y + h), [3, 98, 243], 2)
                cv.putText(img1, label, (x-40, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1, [43, 28, 243], 2)
                #[3, 98, 243]
                print("The given frame has ", label)
        cv.imshow("Image",img1)
        cv.waitKey(0)
        #plt.imsave(f'output_{count}.png',cv.cvtColor(img1, cv.COLOR_BGR2RGB))
        count = count + 1

if __name__ == "__main__":
    main()
