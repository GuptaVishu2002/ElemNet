# Larger CNN for the MNIST Dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from collections import Counter
from tensorflow.keras.layers import Input
import re, os, csv, math, operator
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

#Contains 86 elements (Without Noble elements as it does not forms compounds in normal condition)
elements = ['H','Li','Be', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl',
            'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe','Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge',
            'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd',
            'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er','Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 
            'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu']

# import training data 
def load_data(csvname):
    # load in data
    data = np.asarray(pd.read_csv(csvname))

    # import data and reshape appropriately
    X = data[:,0:-1]
    y = data[:,-1]
    y.shape = (len(y),1)
    
    return X,y

def convert(lst): 
    res_dct = {lst[i]: lst[i + 1] for i in range(0, len(lst), 2)} 
    return res_dct 

separate = re.compile('[A-Z][a-z]?|\d+\.\d')

def correction(x_train):
    new_x = []
    for i in range (0,x_train.shape[0]):
        new_x.append(separate.findall(x_train[i][0])) 
    new_x = np.asarray(new_x)
    new_x.shape = (len(new_x),1)    
    dict_x = convert(new_x[0][0])
    input_x = []
    for i in range (0,new_x.shape[0]):
        input_x.append(convert(new_x[i][0]))
        
    in_elements = np.zeros(shape=(len(input_x), len(elements)))
    comp_no = 0

    for compound in input_x:
        keys = compound.keys()
        for key in keys:
            in_elements[comp_no][elements.index(key)] = compound[key]
        comp_no+=1  

    data = in_elements    
    
    return data

# load data
x_train, y_train = load_data('dataset/train_set.csv')
x_test, y_test = load_data('dataset/test_set.csv')

new_x_train = correction(x_train)
new_x_test = correction(x_test)

new_y_train = y_train
new_y_test = y_test

new_y_train.shape = (len(new_y_train),)
new_y_test.shape = (len(new_y_test),)

#batch_size = new_x_train.shape[0]
#num_input = new_x_train.shape[1]

#in_chem = Input(shape=(num_input,))

def elemnet_model():
	# create model
    model = Sequential()
    model.add(Dense(1024, activation='relu', input_dim=86))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(1-0.8))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(1-0.9))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(1-0.7))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(1-0.8))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='linear'))

	# Compile model
    adam = optimizers.Adam(lr=0.0001)
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=adam, metrics=['mean_absolute_error'])
    return model
# build the model
model = elemnet_model()
# Fit the model
model.fit(new_x_train, new_y_train,verbose=1, validation_data=(new_x_test, new_y_test), epochs=1000, batch_size=32)
# Final evaluation of the model
#pred_y = model.predict(new_x_test)
#mae_score = mean_absolute_error(new_y_test, pred_y)
#print(mae_score)

#scores = model.evaluate(new_x_test, new_y_test, verbose=0)
#print("MAE Error : %.2f%%" % (100-scores[1]*100))