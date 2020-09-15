import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


data = pd.read_csv('r13_withCN.csv')

data = data[(data['Parallax_error'] < 0.15)]

id = data.pop('APOGEE_ID')
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

train_stats = pd.read_csv('trainstats_withcarbon.csv', index_col=0)


test_labels = data.pop('Abs_MAG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_test_data = norm(data)

model = tf.keras.models.load_model('62320withIds.h5')
carbon = normed_test_data.pop('carbon')
nitrogen = normed_test_data.pop('nitrogen')
test_predictions = model.predict(normed_test_data).flatten()
perc_errors = (test_labels - test_predictions) / test_predictions

distance_test = (1/(parallax/1000))

distance_prediction = 10*10**((test_predictions+Extinction-Apparent)/-5)
regression = (distance_test - distance_prediction) / distance_test


Starstocut = data[(data['Grav'] < 5) & (data['Grav'] > 4)& (regression>0.15)]


plt.figure()
plt.scatter(data['TEFF'],test_labels, s = 0.01)
plt.scatter(data['TEFF'][(data['Grav'] < 5) & (data['Grav'] > 4)& (regression>0.15)],test_labels[(data['Grav'] < 5) & (data['Grav'] > 4)& (regression>0.15)], s = 0.01)
plt.xlim(7500, 3000)
plt.xscale('log')
plt.ylim(10, -12)
plt.xlabel('Temp (k)')
plt.ylabel('Absolute Magnitude (K-band)')
plt.title('HR diagram with binaries')
plt.show()

# data = pd.read_csv('r13_withCN.csv')
#
# data_withoutBinaries = data.drop(Starstocut.index)
#
# data_withoutBinaries.to_csv('r13_withCN_nobinaries.csv', index=False)