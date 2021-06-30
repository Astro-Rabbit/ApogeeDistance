import pandas as pd
import tensorflow as tf
# import tensorflow_docs as tfdocs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('APOKASC_Massmodel_withAddonData.csv')

data.pop('Mass_err')
data.pop('source')

train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

id = train_dataset.pop('SDSS_ids')
# id2 = train_dataset.pop('APOKASC_ids')


id_test = test_dataset.pop('SDSS_ids')

# id_test2 = test_dataset.pop('APOKASC_ids')

# Raw_grav = train_dataset.pop('Raw_Grav')
# Raw_grav_test = test_dataset.pop('Raw_Grav')
#
# APOK_Grav = train_dataset.pop('APOK_Grav')
# APOK_Grav_test = test_dataset.pop('APOK_Grav')
#
# Raw_TEFF = train_dataset.pop('Raw_TEFF')
# Raw_TEFF_test = test_dataset.pop('Raw_TEFF')


train_stats = train_dataset.describe()
train_stats.pop("Mass")

train_stats = train_stats.transpose()

train_stats.to_csv('trainstatsMassModel.csv',  index=True)

train_labels = train_dataset.pop('Mass')
test_labels = test_dataset.pop('Mass')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)



def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=[len(normed_train_data.keys())]),
        tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)

EPOCHS = 500

history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[early_stop])

test_predictions = model.predict(normed_test_data).flatten()
train_predictions = model.predict(normed_train_data).flatten()

model.save('Masses.h5')

perc_errors = (test_predictions - test_labels) / test_labels

# a = plt.axes(aspect='equal')
# plt.scatter(test_labels, test_predictions, alpha=0.3, s=1)
# plt.xlabel('True Values [SolarMass]')
# plt.ylabel('Predictions [SolarMass]')
# lims = [0, 5]
# _ = plt.plot(lims, lims)
# plt.show()
#
# plt.scatter(test_labels, perc_errors, alpha=0.3, s=1)
# plt.xlabel('True Values [SolarMass]')
# plt.ylabel('Predictions [SolarMass]')
# plt.plot([0, 3.5], [0, 0])
# plt.xlim(0,4)
# plt.ylim(-2.5,2.5)
# plt.show()
#
plt.figure()
plt.scatter(train_dataset['TEFF'], train_dataset['Grav'], s=1)
plt.xlim(7500, 3000)
plt.ylim(5.5, 0)
plt.xscale('log')
plt.xlabel('Temp (k)')
plt.ylabel('Grav')
plt.title('HR diagram')
plt.show()

data = pd.read_csv('APOKASC_Massmodel_withAddonData.csv')
data.pop('SDSS_ids')
mass = data.pop('Mass')
mass_err  = data.pop('Mass_err')
s = data.pop('source')
norm_data = norm(data)

pred = model.predict(norm_data).flatten()
perc = (pred - mass) / mass

perc_o = (pred - mass) / mass



# plt.figure()
# MAD = []
# model2_err_un = []
# x = np.arange(1, 5, 0.25)
# for i in x:data
#     temp = perc_errors[(data['Grav'] < i) & (data['Grav'] > i - 0.25)].dropna()
#     MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
# plt.plot(x, MAD)
# plt.xlabel('Log G')
# plt.ylabel('MAD of predicted Mass')
# plt.title('MAD of predicted Mass per Log G')
# # plt.savefig('MAD_grav.png')
# plt.show()
#
# [M,B] = np.polyfit(x,MAD, 1)
#
#
# data['bounded'] = np.zeros(len(data))

plt.scatter(mass, perc, alpha=0.3, s=1)
plt.xlabel('True Values [SolarMass]')
plt.ylabel('Predictions [SolarMass]')
plt.plot([0, 3.5], [0, 0])
plt.xlim(0,4)
plt.ylim(-2.5,2.5)
plt.show()

# astronn compare