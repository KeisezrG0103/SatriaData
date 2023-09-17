import keras_ocr
import imutils as im
import cv2
import numpy as np
import os
import math
import matplotlib.pyplot as plt

image_directory = "E:\SatdatRapi\V2\Test_2"
recognizer = keras_ocr.recognition.Recognizer()
#load model.sleep
recognizer.model.load_weights("E:\SatdatRapi\V2\model_sleepy_weights.h5")
pipeline = keras_ocr.pipeline.Pipeline()
