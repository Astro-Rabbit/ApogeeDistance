import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

data = pd.read_csv('Full_DATA.csv')

data = data[(data['Abs_MAG'] < 100) & (data['TEFF'] > 0) & (data['Grav'] > 0) & (data['Metal'] > -9000) & (
            data['Abs_MAG'] > -40)]

stats = data.describe()
stats.pop("Abs_MAG")
stats = stats.transpose()

data_labels = data.pop('Abs_MAG')

def norm(x):
    return (x - stats['mean']) / stats['std']

normed_data = norm(data)

model = tf.keras.models.load_model('test_model_1219.h5')

test_predictions = model.predict(normed_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(data_labels, test_predictions)
plt.xlabel('True Values [Abs_Mag]')
plt.ylabel('Predictions [Abs_Mag]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
