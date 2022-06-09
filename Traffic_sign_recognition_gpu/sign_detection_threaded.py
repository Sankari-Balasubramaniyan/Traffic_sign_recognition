#!/usr/bin/python3
from __future__ import print_function
import time
import cv2
import numpy as np
from process_video import GetVideoFrames
import tensorflow as tf
import tensorflow.keras as keras

####Initialise variables####

img = None
outputs = None
frames_list = []
f_no = 0
td = 0 
tc = 0
process_time = 0
NoneType = type(None)
classes = ['prohibitory', 'danger', 'mandatory', 'other']
CNN_classes = ["Speed limit (20km/h)","Speed limit (30km/h)","Speed limit (50km/h)","Speed limit (60km/h)","Speed limit (70km/h)","Speed limit (80km/h)","End of speed limit (80km/h)","Speed limit (100km/h)","Speed limit (120km/h)","No passing","No passing for vehicles over 3.5 metric tons","Right-of-way at the next intersection","Priority road","Yield","Stop","No vehicles","Vehicles over 3.5 metric tons prohibited","No entry","General caution","Dangerous curve to the left","Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow","Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory","End of no passing","End of no passing by vehicles over 3.5 metric tons"]

####Initiate the model####

net = cv2.dnn.readNetFromDarknet('model/yolov3-my-tiny.cfg', 'model/yolov3-my-tiny_final.weights')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
last_layer = net.getLayerNames()
last_layer = [last_layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
font = cv2.FONT_HERSHEY_SIMPLEX
pv = GetVideoFrames("images/traffic-sign-to-test.mp4").start()
img = pv.frames_queue.get()
blob = cv2.dnn.blobFromImage(img, 1/255.0, (288, 288), swapRB=True, crop=False)
net.setInput(blob)
net.forward(last_layer)
CNN = keras.models.load_model("model/classifier_v15_2.h5")

print("***Loaded and initiated the model***")

####Start detection capturing the frame by frame####

while type(img) is not NoneType:
    start_1 = time.time()
    H, W = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (288, 288), swapRB=True, crop=False)
    net.setInput(blob) 
    start = time.time()
    outputs = net.forward(last_layer)
    end = time.time()
    td += end-start
    print('[DETECTION INFO] Frame number {0} took {1:.4f} seconds'.format(f_no+1, end - start))
    outputs = np.vstack(outputs)

    bounding_boxes = []
    conf = []

    confidence_threshold = 0.5

    for output in outputs:
        scores = output[5:]
        class_no = np.argmax(scores)
        confidence = scores[class_no]
        if confidence > confidence_threshold:
            x, y, w, h = output[:4] * np.array([W, H, W, H])
            p0 = int(x - w//2), int(y - h//2)
            bounding_boxes.append([*p0, int(w), int(h)])
            conf.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bounding_boxes, conf, confidence_threshold, confidence_threshold-0.1)
    if len(indices) > 0:
        for i in indices.flatten():
            (x, y) = (bounding_boxes[i][0], bounding_boxes[i][1])
            (w, h) = (bounding_boxes[i][2], bounding_boxes[i][3])
            crop_img = img[y:y+h, x:x+w]
            if len(crop_img) >0:
                crop_img = cv2.resize(crop_img, (32, 32))
                crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                crop_img =  np.float32(crop_img.reshape(-1,32,32,3))
                start = time.time()
                prediction = np.argmax(CNN.predict(crop_img))
                end = time.time()
                tc += end-start
                print('[CLASSIFICATION INFO] Frame number {0} took {1:.4f} seconds'.format(f_no+1, end - start))
                label = str(CNN_classes[prediction])

            cv2.rectangle(img, (x, y), (x + w, y + h),  [3, 98, 243], 2)
            cv2.putText(img, label, (x, y - 5), font, 0.5, [43, 28, 243], 1)

    keras.backend.clear_session()    
    cv2.imshow("Frames",img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    img = pv.frames_queue.get() 
    f_no += 1
    end_1 = time.time()
    process_time += end_1-start_1
    #writer.write(img)
#writer.release()
print("[INFO] Elasped time: {:.2f}".format(process_time))
print("[INFO] Number of frames totally process per second : {:.2f}".format(f_no/process_time))
print("[INFO] Number of frames detected per second - FPS : {:.2f}".format(f_no/td))
print("[INFO] Number of frames classified per second - Inference time : {:.2f}".format(f_no/tc))
print("[INFO] Total time taken to process(detect + classify) one frame : {:.2f}".format(process_time/f_no))
cv2.destroyAllWindows() 

