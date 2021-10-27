import pandas as pd
import os
from sklearn import preprocessing
from collections import deque
import random
import numpy as np
import time
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint

#Desgin
#have the model train on previous days data and beforehand

#Then use the model on ibpy file to fire a trade


#use the model every minute?

#pull data out of the ibpy db to use into the model
