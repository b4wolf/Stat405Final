import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from multiprocessing import cpu_count
from joblib import Parallel
from joblib import delayed

# Define function to load data
def load_data():
    # Load metadata file
    metadata = pd.read_csv("HAM10000_metadata.csv")
    # Load image data
    images = np.load("HAM10000_images.npz")['arr_0']
    # Encode labels
    le = LabelEncoder()
    le.fit(metadata['dx'])
    labels = le.transform(metadata['dx'])
    labels = np_utils.to_categorical(labels)
    return images, labels

# Define function to create model
def create_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(450, 600, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Load data
X, y = load_data()


