import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

data1 = pd.read_csv('ALL_withapparent_DATA_r12.csv')
data2 = pd.read_csv('ALL_withapparent_DATA_r13.csv')

data1 = data1[(data1['Parallax_error'] < 0.15)]
data2 = data2[(data2['Parallax_error'] < 0.15)]

data = data1.append(data2, ignore_index = True)


Apparent_test = data.pop('Apparent')
Extinction_test = data.pop('Extinction')
parallax_error_test = data.pop('Parallax_error')
parallax_test = data.pop('parallax')


# sns.pairplot(train_dataset[["Abs_MAG", "TEFF", "Grav", "Metal"]], diag_kind="kde")

train_stats = data.describe()
train_stats.pop("Abs_MAG")
train_stats = train_stats.transpose()

test_labels = data.pop('Abs_MAG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_test_data = norm(data)

model = tf.keras.models.load_model('largeNet_15perc_noflattening.h5')

test_predictions = model.predict(normed_test_data).flatten()
perc_errors = (test_labels - test_predictions) / test_predictions

distance_test = (1/(parallax_test/1000))

distance_prediction = 10*10**((test_predictions+Extinction_test-Apparent_test)/-5)
regression = (distance_test - distance_prediction) / distance_test

title = 'largeNet_15perc_noflattening.h5'

def plot_regression(lims=[-15, 15], alpha=0.05):
    plt.scatter(parallax_test, regression, alpha=alpha)
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
    plt.scatter(data['TEFF'], Mag, c=data['Grav'], alpha=0.1)
    plt.xlim(7500, 3000)
    plt.xscale('log')
    plt.ylim(10, -12)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram')
    plt.show()

def plot_error_hr():
    plt.figure()
    plt.scatter(data['TEFF'], test_labels, alpha=0.1)
    plt.scatter(data['TEFF'][abs(regression) > 0.3], test_labels[abs(regression) > 0.3], alpha=0.01)
    plt.xlim(7500, 3000)
    plt.xscale('log')
    plt.ylim(10, -12)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram (Where relative error > 0.3)')
    plt.show()

def plot_parallaxerror_hr():
    plt.figure()
    plt.scatter(data['TEFF'], test_predictions, c=regression, alpha=0.1)
    plt.xlim(7500, 3000)
    plt.xscale('log')
    plt.ylim(10, -12)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram')
    plt.show()

def plot_hist_regression():
    for i in range(1, 6):
        plt.figure()
        name = 'distance regression for Grav < ' + str(i) + ' & Grav > ' + str(i - 1)
        plt.hist(regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], bins=50, range=[-1, 1])
        plt.title(name)
        plt.xlabel('Distance regression')
        plt.ylabel('Number of stars')
        plt.show()
        plt.savefig(name)

def regression_pergrav():
    for i in range(1, 6):
        plt.figure()
        plt.title(title + 'Grav: ' + str(i))
        plt.scatter(distance_test[(data['Grav'] < i) & (data['Grav'] > i - 1)],
                    regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], s=1)
        plt.ylim([-1, 1])
        plt.xlim([-100, 4000])
        plt.show()