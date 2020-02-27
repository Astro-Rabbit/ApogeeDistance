import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('05_withapparent_DATA.csv')

normalizing = pd.read_csv('15_DATA.csv')


stats = normalizing.describe()
stats.pop("Abs_MAG")
# stats.pop('Apparent')
# stats.pop('Extinction')
#
# stats.pop('Parallax_error')
stats = stats.transpose()

data_labels = data.pop('Abs_MAG')

Apparent = data.pop('Apparent')
Extinction = data.pop('Extinction')
parallax_error = data.pop('Parallax_error')

def norm(x):
    return (x - stats['mean']) / stats['std']


normed_data = norm(data)

model = tf.keras.models.load_model('15_ApogeeModel.h5')

test_predictions = model.predict(normed_data).flatten()
perc_errors = (data_labels - test_predictions)

parallaxs = 10*10**((test_predictions+Extinction-Apparent)/-5)
parallaxs_label = 10*10**((data_labels+Extinction-Apparent)/-5)
regression = (parallaxs_label - parallaxs) / parallaxs_label


def plot_regression(lims=[-15, 15], alpha=0.05):
    plt.scatter(parallaxs_label, regression, alpha=alpha)
    plt.xlabel('True Values [Abs_Mag]')
    plt.ylabel('regression [Abs_Mag]')
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, np.zeros(len(lims)))


def histogram(range=[-1, 1], bins=50):
    plt.figure()
    plt.hist(perc_errors, bins=bins, range=range)
    plt.show()

def plot_hr(Mag=data_labels):
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
    plt.scatter(data['TEFF'], data_labels, alpha=0.1)
    plt.scatter(data['TEFF'][abs(regression) > 0.3], data_labels[abs(regression) > 0.3], alpha=0.01)
    plt.xlim(7500, 3000)
    plt.xscale('log')
    plt.ylim(10, -12)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram (Where relative error > 0.3)')
    plt.show()

def plot_parallaxerror_hr():
    plt.figure()
    plt.scatter(data['TEFF'], test_predictions, c=parallax_error, alpha=0.1)
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
        plt.scatter(parallaxs_label[(data['Grav'] < i) & (data['Grav'] > i - 1)],
                    regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], s=1)
        plt.ylim([-1, 1])
        plt.xlim([-100, 4000])
        plt.show()