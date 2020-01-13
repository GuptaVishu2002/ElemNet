# Larger CNN for the MNIST Dataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from collections import Counter
from tensorflow.keras.layers import Input
import re, os, csv, math, operator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import autokeras as ak

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

new_x_train = new_x_train.astype(dtype = 'float32')
new_y_train = new_y_train.astype(dtype = 'float32')
new_x_test = new_x_test.astype(dtype = 'float32')
new_y_test = new_y_test.astype(dtype = 'float32')

#new2_x = tf.convert_to_tensor(new_x_train)
#new2_y = tf.convert_to_tensor(new_y_train)

batch_size1 = new_x_train.shape[0]
num_input1 = new_x_train.shape[1]

input_node = ak.Input(shape=(86,))
output_node = ak.DenseBlock(num_layers=17)(input_node)
output_node = ak.RegressionHead(output_dim=1, loss="mean_absolute_error", metrics="mean_absolute_error")(output_node)
clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=100)
clf.fit(new_x_train, new_y_train)

#input_node = ak.Input()
#output_node = ak.ImageBlock(
#    block_type='denseblock'
#    )(input_node)
#output_node = ak.RegressionHead()(output_node)
#clf = ak.AutoModel(inputs=input_node, outputs=output_node, max_trials=1000)
#clf.fit(new_x_train, new_y_train)