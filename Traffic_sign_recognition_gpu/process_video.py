#!/usr/bin/python3
from __future__ import print_function
from threading import Thread
import cv2
import numpy as np
import time
from queue import Queue

class GetVideoFrames:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        (self.grab, self.frame) = self.cap.read()
        self.frames_queue = Queue()
        self.width = Queue()
        self.height = Queue()
        self.width.put(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.height.put(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.frames_queue.put(self.frame)
        self.video_finish = False
        
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
        
    def update(self):
        while True:
            if self.video_finish:
                return
            (self.grab, self.frame) = self.cap.read()
            self.frames_queue.put(self.frame)
            self.width.put(int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
            self.height.put(int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            if not self.grab:
                self.stop()
        
    def stop(self):
        self.video_finish = True
        print("***Frames have been read***")
