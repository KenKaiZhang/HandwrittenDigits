import os
import cv2
import pickle
import random
import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Dropout, Flatten, MaxPooling2D

TRAINDATADIR = "D:\\MathOpDataset2\\train"
TESTDATADIR = "D:\\MathOpDataset2\\test"
CATEGORIES = {'0':0, '1':1, '2':2, '3':3,
              '4':4, '5':5, '6':6, '7':7,
              '8':8, '9':9, '+':10, '-':11,
              'x':12, '%':13, 'dec':14, '=':15}
IMG_SIZE = 60

def create_dataset(path):
    dataset = []
    for category in CATEGORIES:
        label_path = os.path.join(path, category)
        class_num = CATEGORIES[category]
        for img in os.listdir(label_path):
            img_array = cv2.imread(os.path.join(label_path, img), cv2.IMREAD_GRAYSCALE)
            img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            dataset.append([img_array, class_num])
    return dataset

def split_dataset(dataset):
    x, y = [], []
    random.shuffle(dataset)
    for img, label in dataset:
        x.append(img)
        y.append(label)
    x = np.array(x).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(y)
    x = tf.keras.utils.normalize(x, axis=1)
    return x, y

def generate_model(x,y):
    model = Sequential()

    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D((2,2)))
    model.add(Flatten(input_shape=(28,28)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(16, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x, y, epochs=10)

    return model

def save_dataset(dataset, name):
    with open(f'Dataset\\{name}.pickle', 'wb') as f:
        pickle.dump(dataset, f)
        f.close()

def save_model(model, name):
    model.save(f'{name}.model')

def load_dataset(name):
    with open(f'Dataset\\{name}.pickle', 'rb') as f:
        dataset = pickle.load(f)
        f.close()
    return dataset

def load_saved_model(name):
    return load_model(f'{name}.model')

def predict_digit(digit, model):
    img = cv2.imread(f'Digits\\{digit}', cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.array(img).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    prediction = model.predict(img)
    print(f'PREDICTION: {np.argmax(prediction)}, ACTUAL: {digit}')

#-------------Creating Dataset----------------# 
# train, test = create_dataset(TRAINDATADIR), create_dataset(TESTDATADIR)

# x_train, y_train = split_dataset(train)
# x_test, y_test = split_dataset(test)

#-------------Saving Dataset----------------# 
# save_dataset(x_train, 'x_train')
# save_dataset(y_train, 'y_train')
# save_dataset(x_test, 'x_test')
# save_dataset(y_test, 'y_test')

#-------------Creating Model----------------# 
# model = generate_model(x_train, y_train)
# model.evaluate(x_test, y_test)

#-------------Saving Dataset----------------# 
# save_model(model, 'handwritten')

#-------------Load dataset----------------# 
x_test = load_dataset('x_test')
y_test = load_dataset('y_test')

#-------------Load Model----------------# 
model = load_saved_model('handwritten')

for digit in os.listdir('Digits'):
    predict_digit(digit, model)



