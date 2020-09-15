

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('DatawithID_noabs_mag.csv')

Apparent = data.pop('Apparent')
Extinction = data.pop('Extinction')
id = data.pop('APOGEE_ID')

# sns.pairplot(train_dataset[["Abs_MAG", "TEFF", "Grav", "Metal"]], diag_kind="kde")


train_stats = pd.read_csv('trainstats.csv', index_col=0)


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_test_data = norm(data)

model = tf.keras.models.load_model('singlelayerNet_15perc_noBinaries.h5')

predictions = model.predict(normed_test_data).flatten()
distance_prediction = 10*10**((predictions+Extinction-Apparent)/-5)


data.insert(0,'APOGEE_ID', id.values, True)
data.insert(4,'Apparent', Apparent.values, True)
data.insert(5,'Extinction', Extinction.values, True)
data.insert(6,'Absolute Magnitude Prediction', predictions, True)
data.insert(7,'Distance prediction [pc]', distance_prediction, True)

data.to_csv("culculateddataset.csv")