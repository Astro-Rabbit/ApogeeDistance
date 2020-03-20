import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


data = pd.read_csv('ALL_withapparent_DATA_r13.csv')


data = data[(data['Parallax_error'] < 0.15)]




train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

flatten = False

if flatten == True:
    bins = np.arange(0,5,0.25)
    new_train = pd.DataFrame()
    for i ,bin in enumerate(bins):
        g = train_dataset[(train_dataset['Grav']> bin) & (train_dataset['Grav']< (bin+0.25))]
        if len(g) > 20000:
            frac = 20000/len(g)
            g = g.sample(frac = frac, random_state = 0)
        new_train = new_train.append(g, ignore_index=True)
    train_dataset = new_train

Apparent = train_dataset.pop('Apparent')
Extinction = train_dataset.pop('Extinction')
parallax_error = train_dataset.pop('Parallax_error')
parallax = train_dataset.pop('parallax')


Apparent_test = test_dataset.pop('Apparent')
Extinction_test = test_dataset.pop('Extinction')
parallax_error_test = test_dataset.pop('Parallax_error')
parallax_test = test_dataset.pop('parallax')


# sns.pairplot(train_dataset[["Abs_MAG", "TEFF", "Grav", "Metal"]], diag_kind="kde")

train_stats = data.describe()
train_stats.pop("Abs_MAG")
train_stats.pop('Apparent')
train_stats.pop('Extinction')
train_stats.pop('Parallax_error')
train_stats.pop('parallax')
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Abs_MAG')
test_labels = test_dataset.pop('Abs_MAG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(4, activation='relu'),
        tf.keras.layers.Dense(2, activation='relu'),
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
    callbacks=[early_stop, tfdocs.modeling.EpochDots()])


model.save('supersmallNet_15perc_noflattening.h5')




test_predictions = model.predict(normed_test_data).flatten()
perc_errors = (test_labels - test_predictions) / test_predictions

distance_test = (1/(parallax_test/1000))

distance_prediction = 10*10**((test_predictions+Extinction_test-Apparent_test)/-5)
regression = (distance_test - distance_prediction) / distance_test

def plot_predict():
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions, alpha=0.3)
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
    plt.scatter(test_dataset['TEFF'], Mag, c=test_dataset['Grav'], alpha=0.1)
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
                    regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], s=1)
        plt.ylim([-1, 1])
        plt.xlim([-100, 4000])
        plt.show()