import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('Full_DATA_r12.csv')

data = data[(data['Parallax_error'] < 0.15)]

Apparent = data.pop('Apparent')
Extinction = data.pop('Extinction')
parallax_error = data.pop('Parallax_error')
parallax = data.pop('parallax')
g = data.pop('G')
G_mag_train = g - 5 * np.log10((1/(parallax/1000)) / 10)
BP = data.pop('BP')
RP = data.pop('RP')
vscatter_train = data.pop('vscatter')



# sns.pairplot(train_dataset[["Abs_MAG", "TEFF", "Grav", "Metal"]], diag_kind="kde")

train_stats = data.describe()
train_stats.pop("Abs_MAG")

train_stats = train_stats.transpose()

test_labels = data.pop('Abs_MAG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_test_data = norm(data)

model = tf.keras.models.load_model('singlelayerNet_15perc_noflattening.h5')

test_predictions = model.predict(normed_test_data).flatten()
perc_errors = (test_labels - test_predictions) / test_predictions

distance_test = (1/(parallax/1000))

distance_prediction = 10*10**((test_predictions+Extinction-Apparent)/-5)
regression = (distance_test - distance_prediction) / distance_test


Starstocut = data[(data['Grav'] < 5) & (data['Grav'] > 4)& (regression>0.15)]

data = pd.read_csv('Full_DATA_r12.csv')

data_withoutBinaries = data.drop(Starstocut.index)

data_withoutBinaries.to_csv('noBinaries_data_r12.csv', index=False)