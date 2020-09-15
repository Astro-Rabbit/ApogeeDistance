import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('testsample_withcarbon.csv')
data.pop('Unnamed: 0')

train = pd.read_csv('trainsample_withcarbon.csv')

id = data.pop('APOGEE_ID')
Apparent = data.pop('Apparent')
Extinction = data.pop('Extinction')
parallax_error = data.pop('Parallax_error')
parallax = data.pop('parallax')
g = data.pop('G')
G_mag_train = g - 5 * np.log10((1 / (parallax / 1000)) / 10)
BP = data.pop('BP')
RP = data.pop('RP')
vscatter_train = data.pop('vscatter')

# sns.pairplot(train_dataset[["Abs_MAG", "TEFF", "Grav", "Metal"]], diag_kind="kde")


train_stats = pd.read_csv('trainstats_withcarbon_cluster.csv', index_col=0)
test_labels = data.pop('Abs_MAG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_test_data = norm(data)

model = tf.keras.models.load_model('8112020WithclustersnolmcorsmcX2_withNitro.h5')
model2 = tf.keras.models.load_model('8182020WithclustersnolmcorsmcX3_NONitro.h5')

test_predictions = model.predict(normed_test_data).flatten()
perc_errors = (test_labels - test_predictions) / test_predictions
distance_test = (1 / (parallax / 1000))
distance_prediction = 10 * 10 ** ((test_predictions + Extinction - Apparent) / -5)
regression = (distance_test - distance_prediction) / distance_test

carbon = normed_test_data.pop('carbon')
nitrogen = normed_test_data.pop('nitrogen')
test_predictions2 = model2.predict(normed_test_data).flatten()
perc_errors2 = (test_labels - test_predictions2) / test_predictions2
distance_prediction2 = 10 * 10 ** ((test_predictions2 + Extinction - Apparent) / -5)
regression2 = (distance_test - distance_prediction2) / distance_test


def plot_regression(lims=[-15, 15], alpha=0.05):
    plt.figure()
    a = plt.axes(aspect='equal')
    plt.scatter(distance_test, distance_prediction, c=parallax_error, alpha=0.3, s=0.1)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.colorbar()
    lims = [0, 5000]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


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
                    regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], s=0.1)
        plt.ylim([-1, 1])
        plt.xlim([-100, 4000])
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
        plt.hist(regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], bins=np.arange(-1, 1, 0.05))
        plt.title('regression histogram for Grav:' + str(i) + ' -No flattening model')
        plt.xlim(-1, 1)
        plt.show()


def plot_regression_extinction():
    plt.scatter(Extinction, regression)
    plt.ylim([-1, 1])
    plt.show()


def mad_per():
    plt.figure()
    MAD = []
    model2_err_un = []
    x = np.arange(1, 5, 0.25)
    for i in x:
        temp = regression[(data['Grav'] < i) & (data['Grav'] > i - 0.25)]
        MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
        model2_err = regression2[(data['Grav'] < i) & (data['Grav'] > i - 0.5)]
        model2_err_un.append(1.4826 * np.median(abs(abs(model2_err) - np.median(abs(model2_err)))))
    plt.plot(x, MAD)
    plt.plot(x, model2_err_un)
    plt.xlabel('Log G')
    plt.ylabel('MAD of predicted star distance')
    plt.title('MAD of predicted star distance per Log G')
    # plt.savefig('MAD_grav.png')
    plt.show()


def mad_perGaia():
    plt.figure()
    MAD = []
    x = np.arange(0.01, .16, 0.02)
    for i in x:
        temp = regression[(parallax_error < i) & (parallax_error > i - 0.02)]
        MAD.append(np.median(abs(temp)))
    plt.plot(x, MAD)
    plt.xlabel('Gaia relative uncertainty')
    plt.ylabel('MAD of predicted star distance')
    plt.savefig('StarMAD_GaiaUncertainty.png')
    plt.show()


# fig = plt.figure()
# fig.set_figheight(10)
# fig.set_figwidth(20)
# fig.subplots_adjust(hspace=0.6, wspace=0.2)
#
# # fig2 = plt.figure()
# # fig2.set_figheight(10)
# # fig2.set_figwidth(20)
# # fig2.subplots_adjust(hspace=0.6, wspace=0.2)
#
# grav = ((1, 3), (3, 5))
# te = ((3000, 4000), (4000, 5000), (5000, 20000))
# Metal = np.arange(-2.2, 1.5, 0.05)
# plot = 0
# for i, g in enumerate(grav):
#     for l, t in enumerate(te):
#         Regression = []
#         plot += 1
#         for m in Metal:
#             temp = regression2[
#                 (data['Metal'] < m) & (data['Metal'] > m - 0.25) & (data['Grav'] > g[0]) & (data['Grav'] < g[1]) & (
#                         data['TEFF'] > t[0]) & (data['TEFF'] < t[1]) & (abs(parallax_error)<0.15)]
#             Regression.append(np.median(temp))
#         ax = fig.add_subplot(2, 3, plot)
#         # ax2 = fig2.add_subplot(2, 3, plot)
#         ax.plot(Metal, Regression)
#         ax.set_ylabel('Relative error medians nitrogen with cluster training')
#         ax.set_xlabel('Metallicity')
#         ax.set_title(('log g ' + str(g) + ' and temp ' + str(t)).format())
#         num_train = train[(train['Grav'] > g[0]) & (train['Grav'] < g[1]) & (
#                 train['TEFF'] > t[0]) & (train['TEFF'] < t[1])].shape[0]
#         ax.text(-2, 0.4, 'Number of training samples: ' + str(num_train))
#         ax.set_ylim([-1, 0.5])
#
#         # ax2.hist(train['Metal'][(train['Grav'] > g[0]) & (train['Grav'] < g[1]) & (
#         #         train['TEFF'] > t[0]) & (train['TEFF'] < t[1])])
#         # ax2.set_ylim([0, 1000])
#
# plt.show()