import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

data = pd.read_csv('APOKASC_combinedCalModel.csv')

train_dataset = data.sample(frac=0.8, random_state=0)
test_dataset = data.drop(train_dataset.index)
#
# id = train_dataset.pop('SDSS_ids')
# id2 = train_dataset.pop('APOKASC_ids')
#
# id_test = test_dataset.pop('SDSS_ids')
# id_test2 = test_dataset.pop('APOKASC_ids')
#
# Mass = train_dataset.pop('Mass')
# Mass_test = test_dataset.pop('Mass')

calibrated_grav = train_dataset.pop('Grav')
calibrated_grav_test = test_dataset.pop('Grav')

calibrated_TEFF = train_dataset.pop('TEFF')
calibrated_TEFF_test = test_dataset.pop('TEFF')

# Raw_TEFF = train_dataset.pop('Raw_TEFF')
# Raw_TEFF_test = test_dataset.pop('Raw_TEFF')

# carbon = train_dataset.pop('carbon')
# carbon_test = test_dataset.pop('carbon')
#
# nitrogen = train_dataset.pop('nitrogen')
# nitrogen_test = test_dataset.pop('nitrogen')
#
# evState = train_dataset.pop('evstate')
# evState_test = test_dataset.pop('evstate')

train_stats = train_dataset.describe()
train_stats.pop("APOK_Grav")

train_stats = train_stats.transpose()

train_stats.to_csv('trainstatsLogG_CalModel1.csv',  index=True)

# val_dataset = train_dataset.sample(frac=0.2, random_state=0)
# val_label = val_dataset.pop('APOK_Grav')

train_labels = train_dataset.pop('APOK_Grav')
test_labels = test_dataset.pop('APOK_Grav')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# normed_val_data = norm(val_dataset)






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

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)

EPOCHS = 750

history = model.fit(
    normed_train_data,(train_labels - train_dataset['Raw_Grav']),
    epochs=EPOCHS, validation_split = 0.2, verbose=0,
    callbacks=[early_stop])

plt.figure()
plt.plot(history.history['val_loss'])
plt.xlabel('Val loss')
plt.ylabel('Loss')
plt.show()

plt.figure()
plt.plot(history.history['loss'])
plt.xlabel('Train loss')
plt.ylabel('Loss')
plt.show()

corrections = model.predict(normed_test_data).flatten()
test_predictions = corrections + test_dataset['Raw_Grav']

perc_errors = (test_labels - test_predictions) / test_predictions

residual=test_predictions-test_labels
calibrated_residuals = calibrated_grav_test-test_labels
raw_residuals = test_dataset['Raw_Grav'] - test_labels


# plt.figure()
# plt.scatter(test_labels, test_predictions-test_labels, alpha=0.3, s=1)
# plt.xlabel('APOKASC LogG')
# plt.ylabel('Residual for Predicted calibrated LogG')
# plt.plot([0, 3.5], [0, 0])
# plt.show()
#
#
# plt.figure()
# plt.scatter(test_labels, corrections, alpha=0.3, s=1)
# plt.xlabel('APOKASC LogG')
# plt.ylabel('Correction vs LogG')
# plt.plot([0, 3.5], [0, 0])
# plt.show()

# plt.figure()
# plt.scatter(calibrated_TEFF_test, test_predictions, s=1)
# plt.xlim(7500, 3000)
# plt.ylim(5.5, 0)
# plt.xscale('log')
# plt.xlabel('Calibrated Temp (k)')
# plt.ylabel('NN Model Grav')
# plt.title('HR diagram NN model calibration')
# plt.show()
#
# plt.figure()
# plt.scatter(calibrated_TEFF_test, test_labels, s=1)
# plt.xlim(7500, 3000)
# plt.ylim(5.5, 0)
# plt.xscale('log')
# plt.xlabel('Calibrated Temp (k)')
# plt.ylabel('APOKASC Grav')
# plt.title('HR diagram APOKASC')
# plt.show()
#
# plt.figure()
# plt.scatter(calibrated_TEFF_test, calibrated_grav_test, s=1)
# plt.xlim(7500, 3000)
# plt.ylim(5.5, 0)
# plt.xscale('log')
# plt.xlabel('Calibrated Temp (k)')
# plt.ylabel('cal Grav')
# plt.title('HR diagram calibrated grav')
# plt.show()
#
# plt.figure()
# plt.scatter(calibrated_TEFF_test, test_dataset['Raw_Grav'], s=1)
# plt.xlim(7500, 3000)
# plt.ylim(5.5, 0)
# plt.xscale('log')
# plt.xlabel('Calibrated Temp (k)')
# plt.ylabel('Raw Grav')
# plt.title('HR diagram raw grav')
# plt.show()


# fig2 = plt.figure(constrained_layout=True, figsize=(14, 8))
# spec2 = gridspec.GridSpec(ncols=4, nrows=3, figure=fig2)
# f2_ax1 = fig2.add_subplot(spec2[0, 0])
# f2_ax2 = fig2.add_subplot(spec2[0, 1])
# f2_ax3 = fig2.add_subplot(spec2[0, 2])
# f2_ax4 = fig2.add_subplot(spec2[0, 3])
# f2_ax5 = fig2.add_subplot(spec2[1, 0])
# f2_ax6 = fig2.add_subplot(spec2[1, 1])
# f2_ax7 = fig2.add_subplot(spec2[1, 2])
# f2_ax8 = fig2.add_subplot(spec2[1, 3])
# f2_ax9 = fig2.add_subplot(spec2[2, 0])
# f2_ax10 = fig2.add_subplot(spec2[2, 1])
# f2_ax11 = fig2.add_subplot(spec2[2, 2])
# f2_ax12 = fig2.add_subplot(spec2[2, 3])
#
#
# f2_ax1.scatter(test_labels, residual, s=1)
# f2_ax1.plot([0, 5], [0, 0])
# f2_ax1.set_ylabel('Residual for NN\n against APOKASC Log G')
# f2_ax5.scatter(test_labels, calibrated_residuals, s=1)
# f2_ax5.plot([0, 5], [0, 0])
# f2_ax5.set_ylabel('Residual for calibrated\n log G against APOKASC Log G')
# f2_ax9.scatter(test_labels, raw_residuals, s=1)
# f2_ax9.plot([0, 5], [0, 0])
# f2_ax9.set_ylabel('Residual for Spectroscopic\n Log G against APOKASC Log G')
# f2_ax9.set_xlabel('LogG')
#
# f2_ax2.scatter(test_dataset['Metal'], residual, s=1)
# f2_ax2.plot([-2, 0.5], [0, 0])
# f2_ax6.scatter(test_dataset['Metal'], calibrated_residuals, s=1)
# f2_ax6.plot([-2, 0.5], [0, 0])
# f2_ax10.scatter(test_dataset['Metal'], raw_residuals, s=1)
# f2_ax10.plot([-2, 0.5], [0, 0])
# f2_ax10.set_xlabel('Metallicity')
#
# f2_ax3.scatter(calibrated_TEFF_test, residual, s=1)
# f2_ax3.plot([3500, 7000], [0, 0])
# f2_ax7.scatter(calibrated_TEFF_test, calibrated_residuals, s=1)
# f2_ax7.plot([3500, 7000], [0, 0])
# f2_ax11.scatter(calibrated_TEFF_test, raw_residuals, s=1)
# f2_ax11.plot([3500, 7000], [0, 0])
# f2_ax11.set_xlabel('TEFF')
#
# f2_ax4.scatter(test_dataset['carbon']-test_dataset['nitrogen'], residual, s=1)
# f2_ax4.plot([-1.2, 0.5], [0, 0])
# f2_ax8.scatter(test_dataset['carbon']-test_dataset['nitrogen'], calibrated_residuals, s=1)
# f2_ax8.plot([-1.2, 0.5], [0, 0])
# f2_ax12.scatter(test_dataset['carbon']-test_dataset['nitrogen'], raw_residuals, s=1)
# f2_ax12.plot([-1.2, 0.5], [0, 0])
# f2_ax12.set_xlabel('C/N')
#
#
# plt.show()


fig2 = plt.figure(constrained_layout=True, figsize=(14, 4))
spec2 = gridspec.GridSpec(ncols=4, nrows=1, figure=fig2)
f2_ax1 = fig2.add_subplot(spec2[0, 0])
f2_ax2 = fig2.add_subplot(spec2[0, 1])
f2_ax3 = fig2.add_subplot(spec2[0, 2])
f2_ax4 = fig2.add_subplot(spec2[0, 3])



f2_ax1.scatter(test_labels, corrections, s=1)
f2_ax1.plot([0, 5], [0, 0])
f2_ax1.set_ylabel('corrections for NN\n')
f2_ax1.set_xlabel('LogG')

f2_ax2.scatter(test_dataset['Metal'], corrections, s=1)
f2_ax2.plot([-2, 0.5], [0, 0])
f2_ax2.set_xlabel('Metallicity')

f2_ax3.scatter(calibrated_TEFF_test, corrections, s=1)
f2_ax3.plot([3500, 7000], [0, 0])
f2_ax3.set_xlabel('TEFF')

f2_ax4.scatter(test_dataset['carbon']-test_dataset['nitrogen'], corrections, s=1)
f2_ax4.plot([-1.2, 0.5], [0, 0])
f2_ax4.set_xlabel('C/N')


plt.show()

# fig2 = plt.figure(constrained_layout=True, figsize=(14, 8))
# spec2 = gridspec.GridSpec(ncols=4, nrows=3, figure=fig2)
# f2_ax1 = fig2.add_subplot(spec2[0, 0])
# f2_ax2 = fig2.add_subplot(spec2[0, 1])
# f2_ax3 = fig2.add_subplot(spec2[0, 2])
# f2_ax4 = fig2.add_subplot(spec2[0, 3])
# f2_ax5 = fig2.add_subplot(spec2[1, 0])
# f2_ax6 = fig2.add_subplot(spec2[1, 1])
# f2_ax7 = fig2.add_subplot(spec2[1, 2])
# f2_ax8 = fig2.add_subplot(spec2[1, 3])
# f2_ax9 = fig2.add_subplot(spec2[2, 0])
# f2_ax10 = fig2.add_subplot(spec2[2, 1])
# f2_ax11 = fig2.add_subplot(spec2[2, 2])
# f2_ax12 = fig2.add_subplot(spec2[2, 3])
#
#
# f2_ax1.scatter(test_labels[evState_test==1], residual[evState_test==1], s=1, label='RGB')
# f2_ax1.scatter(test_labels[evState_test==2], residual[evState_test==2], s=1, label='RC')
# f2_ax1.scatter(test_labels[evState_test==-1], residual[evState_test==-1], s=1, label='N/A')
# f2_ax1.legend()
# f2_ax1.plot([0, 3.5], [0, 0])
# f2_ax1.set_ylabel('Residual for NN\n against APOKASC Log G')
# f2_ax5.scatter(test_labels, calibrated_residuals, s=1)
# f2_ax5.plot([0, 3.5], [0, 0])
# f2_ax5.set_ylabel('Residual for calibrated\n log G against APOKASC Log G')
# f2_ax9.scatter(test_labels, raw_residuals, s=1)
# f2_ax9.plot([0, 3.5], [0, 0])
# f2_ax9.set_ylabel('Residual for Spectroscopic\n Log G against APOKASC Log G')
# f2_ax9.set_xlabel('LogG')
#
# f2_ax2.scatter(test_dataset['Metal'][evState_test==1], residual[evState_test==1], s=1)
# f2_ax2.scatter(test_dataset['Metal'][evState_test==2], residual[evState_test==2], s=1)
# f2_ax2.scatter(test_dataset['Metal'][evState_test==-1], residual[evState_test==-1], s=1)
# f2_ax2.plot([-2, 0.5], [0, 0])
# f2_ax6.scatter(test_dataset['Metal'], calibrated_residuals, s=1)
# f2_ax6.plot([-2, 0.5], [0, 0])
# f2_ax10.scatter(test_dataset['Metal'], raw_residuals, s=1)
# f2_ax10.plot([-2, 0.5], [0, 0])
# f2_ax10.set_xlabel('Metallicity')
#
# f2_ax3.scatter(calibrated_TEFF_test[evState_test==1], residual[evState_test==1], s=1)
# f2_ax3.scatter(calibrated_TEFF_test[evState_test==2], residual[evState_test==2], s=1)
# f2_ax3.scatter(calibrated_TEFF_test[evState_test==-1], residual[evState_test==-1], s=1)
# f2_ax3.plot([3500, 5750], [0, 0])
# f2_ax7.scatter(calibrated_TEFF_test, calibrated_residuals, s=1)
# f2_ax7.plot([3500, 5750], [0, 0])
# f2_ax11.scatter(calibrated_TEFF_test, raw_residuals, s=1)
# f2_ax11.plot([3500, 5750], [0, 0])
# f2_ax11.set_xlabel('TEFF')
#
# f2_ax4.scatter((test_dataset['carbon']-test_dataset['nitrogen'])[evState_test==1], residual[evState_test==1], s=1)
# f2_ax4.scatter((test_dataset['carbon']-test_dataset['nitrogen'])[evState_test==2], residual[evState_test==2], s=1)
# f2_ax4.scatter((test_dataset['carbon']-test_dataset['nitrogen'])[evState_test==-1], residual[evState_test==-1], s=1)
# f2_ax4.plot([-1.2, 0.5], [0, 0])
# f2_ax8.scatter(test_dataset['carbon']-test_dataset['nitrogen'], calibrated_residuals, s=1)
# f2_ax8.plot([-1.2, 0.5], [0, 0])
# f2_ax12.scatter(test_dataset['carbon']-test_dataset['nitrogen'], raw_residuals, s=1)
# f2_ax12.plot([-1.2, 0.5], [0, 0])
# f2_ax12.set_xlabel('C/N')
#
# plt.savefig('APOKASCOnly.png')
# plt.show()


def mad_per():
    plt.figure()
    MAD = []
    MAD2 = []
    MAD3 = []
    x = np.arange(1, 5, 0.25)
    for i in x:
        temp = raw_residuals[(test_labels < i) & (test_labels > i - 0.25)]
        MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))

        temp2 = residual[(test_labels < i) & (test_labels > i - 0.25)]
        MAD2.append(1.4826 * np.median(abs(abs(temp2) - np.median(abs(temp2)))))

        temp3 = calibrated_residuals[(test_labels < i) & (test_labels > i - 0.25)]
        MAD3.append(1.4826 * np.median(abs(abs(temp3) - np.median(abs(temp3)))))

    plt.plot(x, MAD, label='Raw')
    plt.plot(x, MAD2, label='NN')
    plt.plot(x, MAD3, label='Calibrated')
    plt.xlabel('Log G')
    plt.ylabel('MAD of calibrated Log G')
    plt.title('MAD of predicted star distance per Log G')
    plt.legend()
    # plt.savefig('MAD_grav.png')
    plt.show()

from astropy.io import fits


SDSSdata = fits.open('allStarLite-r13-l33-58932beta.fits')
columns = ['TEFF', 'Grav', 'Metal', 'carbon', 'nitrogen', 'Raw_TEFF', 'Raw_Grav']
star = pd.DataFrame(
    data=[SDSSdata[1].data['TEFF'], SDSSdata[1].data['LOGG'], SDSSdata[1].data['M_H'], SDSSdata[1].data['C_FE'],
          SDSSdata[1].data['N_FE'], SDSSdata[1].data['TEFF_SPEC'], SDSSdata[1].data['LOGG_SPEC']], index=columns)
stars = star.T
stars = stars[(stars['TEFF'] > 0) & (stars['Grav'] > 0) & (stars['Metal'] > -9000) & (stars['carbon'] > -9000) & (
                stars['nitrogen'] > -9000)]

calibrated_grav_sdss = stars.pop('Grav')

calibrated_TEFF_sdss = stars.pop('TEFF')

org_data = stars.copy()

def clipping():
    stars['Metal'][stars['Metal'] > 0.5] = 0.5
    stars['Metal'][(stars['Metal'] < -1.5)& (stars['Raw_Grav'] < 4)] = -1.5
    stars['Metal'][(stars['Metal'] < -1) & (stars['Raw_Grav'] > 4)] = -1
    stars['Raw_TEFF'][(stars['Raw_TEFF'] > 6500)] = 6500
    stars['Raw_Grav'][stars['Raw_Grav'] > 5] = 5
    stars['Raw_Grav'][stars['Raw_Grav'] < 0.5] = 0.5
    # stars['Raw_Grav'][(stars['Raw_TEFF'] < 4800)&(stars['Raw_Grav'] < 3.5)] = 4

clipping()

train_stats = pd.read_csv('trainstatsLogG_CalModel.csv', index_col=0)

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_sdss_data = norm(stars)
SDSS_corr = model.predict(normed_sdss_data).flatten()
SDSS_predictions = SDSS_corr + org_data['Raw_Grav']

# met = [0.5,0,-0.5,-1,-1.5]
# for m in met:
#     temp = stars
#     temp['Metal'] = m
#     normed_sdss_data = norm(temp)
#     SDSS_corr = model.predict(normed_sdss_data).flatten()
#     SDSS_predictions = SDSS_corr + stars['Raw_Grav']
#
#     plt.figure()
#     plt.scatter(calibrated_grav_sdss, SDSS_corr, alpha=0.3, s=1)
#     plt.xlabel('APOKASC LogG')
#     plt.ylabel('Correction')
#     plt.plot([0, 3.5], [0, 0])
#     plt.title('Correction vs LogG'+str(m)+' Metal')
#     plt.show()
#
# plt.figure()
# plt.2dhist(calibrated_TEFF_sdss, calibrated_grav_sdss, bins  = 100)
# plt.xlim(7500, 3000)
# plt.ylim(5.5, 0)
# plt.xscale('log')
# plt.xlabel('Calibrated Temp (k)')
# plt.ylabel('calibrated Grav')
# plt.title('HR diagram calibrated grav on SDSS')
# plt.show()
#
# plt.figure()
# plt.2dhist(calibrated_TEFF_sdss, SDSS_predictions, bins  = 100)
# plt.xlim(7500, 3000)
# plt.ylim(5.5, 0)
# plt.xscale('log')
# plt.xlabel('Calibrated Temp (k)')
# plt.ylabel('NN Model Grav')
# plt.title('HR diagram NN model calibration on SDSS')
# plt.show()



# fig2 = plt.figure(constrained_layout=True, figsize=(14, 4))
# spec2 = gridspec.GridSpec(ncols=4, nrows=1, figure=fig2)
# f2_ax1 = fig2.add_subplot(spec2[0, 0])
# f2_ax2 = fig2.add_subplot(spec2[0, 1])
# f2_ax3 = fig2.add_subplot(spec2[0, 2])
# f2_ax4 = fig2.add_subplot(spec2[0, 3])
#
# f2_ax1.scatter(calibrated_grav_sdss, SDSS_corr, s=0.1)
# f2_ax1.plot([0, 5], [0, 0])
# f2_ax1.set_ylabel('corrections for NN\n')
# f2_ax1.set_xlabel('LogG')
# f2_ax1.set_ylim([-1,1])
# f2_ax1.set_xlim([0,5.1])
#
#
# f2_ax2.scatter(stars['Metal'], SDSS_corr, s=0.1)
# f2_ax2.plot([-2, 0.5], [0, 0])
# f2_ax2.set_xlabel('Metallicity')
# f2_ax2.set_ylim([-1,1])
# f2_ax2.set_xlim([-2,0.6])
#
#
# f2_ax3.scatter(calibrated_TEFF_sdss, SDSS_corr, s=0.1)
# f2_ax3.plot([3500, 7000], [0, 0])
# f2_ax3.set_xlabel('TEFF')
# f2_ax3.set_ylim([-1,1])
# f2_ax3.set_xlim([3000,7050])
#
#
# f2_ax4.scatter(stars['carbon']-stars['nitrogen'], SDSS_corr, s=0.1)
# f2_ax4.plot([-1.2, 0.5], [0, 0])
# f2_ax4.set_xlabel('C/N')
# f2_ax4.set_ylim([-1,1])
# f2_ax4.set_xlim([-1.2,0.5])
#
#
#
# plt.show()