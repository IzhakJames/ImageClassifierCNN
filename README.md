Using TensorFlow Library, this project is on an image classifier using Convolutional Neural Network for traffic signs in Singapore. The aim of the model is to find applicability in the development of Autonomous vehicles in identifying traffic information during drives excluding highway.

| Label | Signs                                |
|-------|--------------------------------------|
| 0     | END OF EXP WAY                       |
| 1     | NO JAYWALKING                        |
| 2     | TP CAMERA ZONE                       |
| 3     | SPD LIMIT 90                         |
| 4     | TURN LEFT                            |
| 5     | ERP                                  |
| 6     | SPD LIMIT 70                         |
| 7     | U TURN LANE                          |
| 8     | SPLIT WAY                            |
| 9     | STOP                                 |
| 10    | SPD LIMIT 50                         |
| 11    | CURVE RIGHT ALIGNMENT MARKER         |
| 12    | ZEBRA CROSSING                       |
| 13    | RAIN SHELTER                         |
| 14    | NO ENTRY                             |
| 15    | KEEP LEFT                            |
| 16    | PARKING AREA FOR MOTORCARS           |
| 17    | PEDESTRIAN USE CROSSING              |
| 18    | RESTRICTED ZONE AHEAD                |
| 19    | CURVE LEFT ALIGNMENT MARKER          |
| 20    | START OF EXP WAY                     |
| 21    | GIVE WAY                             |
| 22    | NO VEH OVER HEIGHT 4.5               |
| 23    | SPD LIMIT 40                         |
| 24    | SLOW SPEED                           |
| 25    | ROAD HUMP                            |
| 26    | NO LEFT TURN                         |
| 27    | ONE WAY RIGHT                        |
| 28    | ONE WAY LEFT                         |
| 29    | SLOW DOWN                            |
| 30    | MERGE                                |
| 31    | NO RIGHT TURN                        |

There are two code-base jupyter notebooks:

1. Image processing - Preprocessing of images to prepare different types of processed images to be fit to the model
2. Development of model - Developed a CNN to be trained on various types of processed images from raw (gray scaled) to normalised(color scaled)

Libraries need for code to run, please ensure to install any missing modules using: "pip install <module>"
import pandas as pd
import random
import pickle
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten,BatchNormalization, Dropout, Lambda, Conv2D, MaxPool2D
from tensorflow.keras.utils import plot_model
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.preprocessing import LabelEncoder
