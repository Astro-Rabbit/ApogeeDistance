import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling
import numpy as np
import seaborn as sns

data = pd.read_csv('DATA.csv')

data = data[(data['parallax']>0) & (data['TEFF']>0) & (data['Grav']>0) &(data['Metal']>-9000)]

train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)

#sns.pairplot(train_dataset[["parallax", "TEFF", "Grav", "Metal"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("parallax")
train_stats = train_stats.transpose()
train_stats


train_labels = train_dataset.pop('parallax')
test_labels = test_dataset.pop('parallax')


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

EPOCHS = 500

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[early_stop, tfdocs.modeling.EpochDots()])