import cv2 as cv
import numpy as np
import time
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import load_model
import skimage.morphology as morp
from skimage.filters import rank

img = None
outputs = None

classes = ['prohibitory', 'danger', 'mandatory', 'other']
colors = [[102, 220, 225], [95, 179, 61], [234, 203, 92], [3, 98, 243]]

net = cv.dnn.readNetFromDarknet('model/yolov3-my-tiny.cfg', 'model/yolov3-my-tiny_final.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

last_layer = net.getLayerNames()
last_layer = [last_layer[i - 1] for i in net.getUnconnectedOutLayers()]
font = cv.FONT_HERSHEY_SIMPLEX

def load_video(path):
    cap = cv.VideoCapture(path)
    frames = 0
    time_taken = 0
    #width= int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    #height= int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    
    #writer= cv.VideoWriter('output.mp4', cv.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

    CNN = load_model("Classifier.h5")
    CNN_classes = ["Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)","Speed limit (120km/h)","No passing","No passing for vehicles over 3.5 metric tons","Right-of-way at the next intersection","Priority road","Yield","Stop","No vehicles","Vehicles over 3.5 metric tons prohibited","No entry","General caution","Dangerous curve to the left","Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow","Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory","End of no passing","End of no passing by vehicles over 3.5 metric tons"]


    while(cap.isOpened()):
        ret, img = cap.read()

        if not ret:
            break  

        frames += 1
        H, W = img.shape[:2]
        blob = cv.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        outputs = net.forward(last_layer)
        end = time.time()
        outputs = np.vstack(outputs)
        time_taken += end-start

        print('Frame number {0} took {1:.4f} seconds'.format(frames, end - start))

        bounding_boxes = []
        conf = []
        class_num = []

        confidence_threshold = 0.2

        for output in outputs:
            scores = output[5:]
            class_no = np.argmax(scores)
            confidence = scores[class_no]
            if confidence > confidence_threshold:
                x, y, w, h = output[:4] * np.array([W, H, W, H])
                p0 = int(x - w//2), int(y - h//2)
                p1 = int(x + w//2), int(y + h//2)
                bounding_boxes.append([*p0, int(w), int(h)])
                conf.append(float(confidence))
                class_num.append(class_no)

        indices = cv.dnn.NMSBoxes(bounding_boxes, conf, confidence_threshold, confidence_threshold-0.1)
        if len(indices) > 0:
            for i in indices.flatten():
                (x, y) = (bounding_boxes[i][0], bounding_boxes[i][1])
                (w, h) = (bounding_boxes[i][2], bounding_boxes[i][3])
                color = [int(c) for c in colors[class_num[i]]]
                crop_img = img[y:y+h, x:x+w]
                if len(crop_img) >0:
                    crop_img = cv.resize(crop_img, (32, 32))
                    crop_img = cv.cvtColor(crop_img, cv.COLOR_BGR2RGB)
                    crop_img =  crop_img.reshape(-1,32,32,3)
                    prediction = np.argmax(CNN.predict(crop_img))
                    label = str(CNN_classes[prediction])
                #text = "{}: {:.4f}".format(classes[class_num[i]], conf[i])
                cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv.putText(img, label, (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv.imshow("Image",img)
        #writer.write(img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    #writer.release()
    cv.destroyAllWindows()

    return frames, time_taken

def main():
    f,t = load_video("images/traffic-sign-to-test.mp4")
    print('Total number of frames', f)
    print('Total amount of time {:.4f} seconds'.format(t))
    print('FPS:', round((f / t), 1))

if __name__ == "__main__":
    main()