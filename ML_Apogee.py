import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('DR17FULL_PARAM_BinCut.csv')


data = data[(abs(data['Parallax_error']) < 0.15)]


train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

train_dataset.to_csv('trainsample_withcarbon_param.csv')
test_dataset.to_csv('testsample_withcarbon_param.csv')

# train_dataset = pd.read_csv('trainsample_withClustersadded8112020x3_nitroFixed.csv')
# test_dataset = pd.read_csv('testsample_withcarbon.csv')
# test_dataset.pop('Unnamed: 0')


id = train_dataset.pop('APOGEE_ID')
Apparent = train_dataset.pop('Apparent')
Extinction = train_dataset.pop('Extinction')
parallax_error = train_dataset.pop('Parallax_error')
distance = train_dataset.pop('distance')
clust_member = train_dataset.pop('clusterID')
# g = train_dataset.pop('G')
# G_mag_train = g - 5 * np.log10((1/(parallax/1000)) / 10)
# BP = train_dataset.pop('BP')
# RP = train_dataset.pop('RP')
# vscatter_train = train_dataset.pop('vscatter')

id_test = test_dataset.pop('APOGEE_ID')
Apparent_test = test_dataset.pop('Apparent')
Extinction_test = test_dataset.pop('Extinction')
parallax_error_test = test_dataset.pop('Parallax_error')
distance_test = test_dataset.pop('distance')
clust_member_test = test_dataset.pop('clusterID')
# g_test = test_dataset.pop('G')
# G_mag_test = g_test - 5 * np.log10((1/(parallax_test/1000)) / 10)
# BP_test = test_dataset.pop('BP')
# RP_test = test_dataset.pop('RP')
# vscatter_test = test_dataset.pop('vscatter')


# sns.pairplot(train_dataset[["Abs_MAG", "TEFF", "Grav", "Metal"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("Abs_MAG")


train_stats = train_stats.transpose()

train_stats.to_csv('trainstatsDR17_param_cut.csv',  index=True)

train_labels = train_dataset.pop('Abs_MAG')
test_labels = test_dataset.pop('Abs_MAG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']



normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
# carbon = normed_train_data.pop('carbon')
# nitrogen = normed_train_data.pop('nitrogen')
#
# carbon2 = normed_test_data.pop('carbon')
# nitrogen2 = normed_test_data.pop('nitrogen')

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


model.save('DR17_param.h5')




test_predictions = model.predict(normed_test_data).flatten()
perc_errors = (test_labels - test_predictions) / test_predictions

# distance_test = (1/(parallax_test/1000))

distance_prediction = 10*10**((test_predictions+Extinction_test-Apparent_test)/-5)
regression = (distance_test - distance_prediction) / distance_test

def plot_predict():
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions, alpha=0.3, s = 1)
    plt.xlabel('True Values [Abs_Mag]')
    plt.ylabel('Predictions [Abs_Mag]')
    lims = [-10, 15]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)


def plot_regression(lims=[-15, 15], alpha=0.05):
    regression = (test_labels - test_predictions)
    plt.scatter(test_labels, regression, alpha=alpha)
    plt.xlabel('True Values [Abs_Mag]')
    plt.ylabel('regression [Abs_Mag]')
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, np.zeros(len(lims)))


def histogram(range=[-1, 1], bins=50):
    plt.figure()
    plt.hist(perc_errors, bins=bins, range=range)
    plt.show()

def plot_hr(Mag=test_labels):
    plt.figure()
    plt.scatter(test_dataset['TEFF'], Mag, c=test_dataset['Grav'], alpha=0.1, s = 0.1)
    plt.xlim(7500, 3000)
    plt.xscale('log')
    plt.ylim(10, -12)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram')
    plt.show()

def plot_error_hr():
    plt.figure()
    plt.scatter(test_dataset['TEFF'], test_labels, alpha=0.1)
    plt.scatter(test_dataset['TEFF'][abs(perc_errors) > 0.3], test_labels[abs(perc_errors) > 0.3], alpha=0.1)
    plt.xlim(7500, 3000)
    plt.xscale('log')
    plt.ylim(10, -12)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram (Where relative error > 0.3)')
    plt.show()

def plot_parallaxerror_hr():
    plt.figure()
    plt.scatter(test_dataset['TEFF'], test_predictions, c=parallax_error_test, alpha=0.1)
    plt.xlim(7500, 3000)
    plt.xscale('log')
    plt.ylim(10, -12)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram')
    plt.show()

def regression_pergrav():
    for i in range(1, 6):
        plt.figure()
        plt.scatter(distance_test[(data['Grav'] < i) & (data['Grav'] > i - 1)],
                    regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], s=0.01)
        plt.ylim([-1, 1])
        plt.xlim([-100, 4000])
        plt.title('regression plot for Grav:' + str(i) + ' -No flattening model')
        plt.show()


def plot_predict_dist():
    plt.figure()
    a = plt.axes(aspect='equal')
    plt.scatter(distance_test, distance_prediction, s=0.01, alpha=0.3)
    plt.xlabel('True Values [Abs_Mag]')
    plt.ylabel('Predictions [Abs_Mag]')
    lims = [0, 10000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()

def gaia_hr():
    plt.figure()
    plt.scatter((BP - RP), G_mag_test, s=1, alpha=0.1)
    plt.xlim(0, 5)
    plt.ylim(13, -2)
    plt.xlabel('Gaia BP-RP colour')
    plt.ylabel('Absolute Magnitude Gaia G')
    plt.title('Gaia HR diagram')
    plt.show()

def plot_regression_hist():
    for i in range(1, 6):
        plt.figure()
        plt.hist(regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], bins=np.arange(-1,1,0.05))
        plt.title('regression histogram for Grav:' + str(i) + ' -No flattening model')
        plt.xlim(-1, 1)
        plt.show()


