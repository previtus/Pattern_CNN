wget http://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz

extract ...

rm dtd-r1.0.1/dtd/images/waffled/.directory

pip install keras

pip install tensorflow

pip install matplotlib

pip install pydot

sudo apt-get install graphviz

pip install sklearn

pip install pandas seaborn



# test it with:

import sys
from os import listdir
from os.path import isfile, join
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import random
import keras
from matplotlib import pyplot as plt
from keras import applications
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from visualize_history import visualize_history
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sn
import pandas  as pd