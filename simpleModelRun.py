import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

filename = 'Full_DATA_r12.csv'  # file with data. I've included the full dataset, and the training & test sets (That have had the binaries cut off

data = pd.read_csv(filename)

cutoff = 0.15  # Set this to the maximum allowed relative error

data = data[(data['Parallax_error'] < cutoff)]


"""Now we have to cut all the extra data that model doesn't use for predictions but that we still may want for 
plotting """

Apparent = data.pop('Apparent')
Extinction = data.pop('Extinction')
parallax_error = data.pop('Parallax_error')
parallax = data.pop('parallax')
g = data.pop('G')
G_mag = g - 5 * np.log10((1/(parallax/1000)) / 10)
BP = data.pop('BP')
RP = data.pop('RP')
vscatter = data.pop('vscatter')

distance = (1/(parallax/1000))  # in Parsecs


train_stats = pd.read_csv('trainstats.csv')  # reads file with stats used during model training

labels = data.pop('Abs_MAG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_data = norm(data)


modelfile = 'singlelayerNet_15perc_cutBinaries.h5'  # file that contains the trained model

model = tf.keras.models.load_model(modelfile)  # loads model into Tensorflow model object

predictions = model.predict(normed_data).flatten()  # uses data with model to predict the absolute magnitude of stars

distance_prediction = 10*10**((predictions+Extinction-Apparent)/-5)

distance_regression = (distance - distance_prediction) / distance



