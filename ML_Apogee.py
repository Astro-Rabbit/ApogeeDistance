import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('15_DATA.csv')



train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)

# sns.pairplot(train_dataset[["Abs_MAG", "TEFF", "Grav", "Metal"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("Abs_MAG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Abs_MAG')
test_labels = test_dataset.pop('Abs_MAG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=[len(train_dataset.keys())]),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.01),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
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

plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)

def plot_loss():
    plotter.plot({'Basic': history}, metric="mse")
    plt.ylim([0, 0.6])
    plt.ylabel('MAE [Abs_Mag]')

test_predictions = model.predict(normed_test_data).flatten()

def plot_predict():
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions, alpha=0.3)
    plt.xlabel('True Values [Abs_Mag]')
    plt.ylabel('Predictions [Abs_Mag]')
    lims = [-10, 15]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)

def plot_hr():
    plt.figure()
    plt.xlim(7500, 3000)
    plt.xscale('log')
    plt.ylim(10, -12)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram')
    plt.scatter(test_dataset['TEFF'], test_labels, alpha=0.1)
    plt.scatter(test_dataset['TEFF'], test_predictions, alpha=0.1)
    plt.savefig('HR diagram')
    plt.show()


def plot_regression(lims = [-15, 15], alpha=0.05):
    regression = (test_labels - test_predictions)
    plt.scatter(test_labels, regression, alpha=alpha)
    plt.xlabel('True Values [Abs_Mag]')
    plt.ylabel('regression [Abs_Mag]')
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, np.zeros(len(lims)))

def histogram(range = [-1, 1],  bins= 50 ):
    perc_errors = (test_labels - test_predictions)/test_predictions
    plt.hist(perc_errors, bins=bins, range=range)

def random():
    plt.scatter(test_dataset['TEFF'], test_labels, c=test_dataset['Grav'], alpha=0.1)
    i = np.where(test_dataset.iloc[:, 1] < 3)[0]