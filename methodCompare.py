import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
from astropy.io import fits



data = pd.read_csv('testsample_withcarbon.csv')
data = data[(abs(data['Parallax_error']) < 0.9)]
data.pop('Unnamed: 0')

data2 = fits.open('apogee_astroNN-r13-l33-58932beta.fits')
AstroNN_ids = data2[1].data['APOGEE_ID']
AstroNN_dist = data2[1].data['DIST']
# Astro_m = data2[1].data['M_H']
# Astro_grav = data2[1].data['LOGG']
# Astro_T = data2[1].data['TEFF']
err  = data2[1].data['DIST_ERROR']
NN_err = err/AstroNN_dist

dataS = fits.open('apogee_starhorse-DR16.fits')
starhorseID = dataS[1].data['APOGEE_ID']
starhorseDist = 1000*dataS[1].data['dist50']

id = data.pop('APOGEE_ID')
ids = id.to_numpy()

AstroNN_ids = AstroNN_ids[(AstroNN_dist>0)&(NN_err<.2)]
AstroNN_dist = AstroNN_dist[(AstroNN_dist>0)&(NN_err<.2)]

NNxy, NNx_ind, NNy_ind = np.intersect1d(ids, AstroNN_ids, return_indices=True)
NN_match = AstroNN_dist[NNy_ind]

starxy,starx_ind,stary_ind = np.intersect1d(ids, starhorseID, return_indices=True)
star_match = starhorseDist[stary_ind]
star_grav_match = data['Grav'][starx_ind]


Apparent = data.pop('Apparent')
Extinction = data.pop('Extinction')
parallax_error = data.pop('Parallax_error')
parallax = data.pop('parallax')
g = data.pop('G')
G_mag_train = g - 5 * np.log10((1/(parallax/1000)) / 10)
BP = data.pop('BP')
RP = data.pop('RP')
vscatter_train = data.pop('vscatter')

train_stats = pd.read_csv('trainstats_withcarbon_cluster.csv', index_col=0)
test_labels = data.pop('Abs_MAG')

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_test_data = norm(data)

model = tf.keras.models.load_model('71620WithclustersX3_withNitro.h5')

test_predictions = model.predict(normed_test_data).flatten()

distance_prediction = 10*10**((test_predictions+Extinction-Apparent)/-5)
distance_true = 1/(parallax/1000)
NN_distance_true = distance_true[NNx_ind]
star_distance_true = distance_true[starx_ind]

NNgrav_match = data['Grav'][NNx_ind]

NNpredictions = distance_prediction[NNx_ind]
starpredictions = distance_prediction[starx_ind]

regression = (NN_distance_true - NN_match)/NN_distance_true
regression2 = (NN_distance_true - NNpredictions) / NN_distance_true

star_regression = (star_distance_true - star_match)/star_distance_true
Star_regression2 = (star_distance_true - starpredictions) / star_distance_true




# fig = plt.figure()
# fig.set_figheight(10)
# fig.set_figwidth(20)
# fig.subplots_adjust(hspace=0.6, wspace=0.2)
# grav = ((1, 3), (3, 5))
# te = ((3000, 4000), (4000, 5000), (5000, 20000))
# Metal = np.arange(-2.2, 1.5, 0.05)
# plot = 0
# for i, g in enumerate(grav):
#     for l, t in enumerate(te):
#         Regression = []
#         plot += 1
#         for m in Metal:
#             temp = regression[
#                 (data['Metal'][NNx_ind] < m) & (data['Metal'][NNx_ind] > m - 0.25) & (NNgrav_match > g[0]) & (NNgrav_match < g[1]) & (
#                         data['TEFF'][NNx_ind] > t[0]) & (data['TEFF'][NNx_ind] < t[1])]
#             Regression.append(temp.mean())
#         ax = fig.add_subplot(2, 3, plot)
#
#         ax.plot(Metal, Regression)
#         ax.set_ylabel('Relative error')
#         ax.set_xlabel('Metallicity')
#         ax.set_title(('AstroNN Full log g ' + str(g) + ' and temp ' + str(t)).format())
#
#         ax.set_ylim([-1, 0.5])
#
#
#
# plt.show()

# plt.figure()
# a = plt.axes(aspect='equal')
# plt.scatter(NNpredictions, NN_distance_true, c = NNgrav_match, alpha=0.3, s=0.1)
# plt.xlabel('Model predictions [pc]')
# plt.ylabel('Gaia [pc]')
# lims = [0, 5000]
# plt.xlim(lims)
# plt.ylim(lims)
# plt.title('Model vs Gaia')
# cbar = plt.colorbar()
# cbar.set_label('Log G')
# _ = plt.plot(lims, lims)
# plt.show()
#
#
# plt.figure()
# a = plt.axes(aspect='equal')
# plt.scatter(NN_match, NN_distance_true, c = NNgrav_match, alpha=0.3, s=0.1)
# plt.xlabel('AstroNN [pc]')
# plt.ylabel('Gaia [pc]')
# lims = [0, 5000]
# plt.xlim(lims)
# plt.ylim(lims)
# plt.title('AstroNN vs Gaia')
# cbar = plt.colorbar()
# cbar.set_label('Log G')
# _ = plt.plot(lims, lims)
# plt.savefig('AstroNNVGaia.png')
# plt.show()
#
#
plt.figure()
MAD = []
MAD2 = []
Gaia_un = []
x = np.arange(1, 5, 0.25)
for i in x:
    temp = regression[(NNgrav_match < i) & (NNgrav_match > i - 0.25)]
    MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
    temp2 = regression2[(NNgrav_match < i) & (NNgrav_match > i - 0.25)]
    MAD2.append(1.4826 * np.median(abs(abs(temp2) - np.median(abs(temp2)))))
    Gaia_err = parallax_error[NNx_ind][(NNgrav_match < i) & (NNgrav_match > i - 0.25)]
    Gaia_un.append(np.median(abs(Gaia_err)))
plt.plot(x, MAD, label = 'AstroNN')
plt.plot(x, MAD2, label = 'Model')
plt.plot(x, Gaia_un, linestyle = ':' ,label = 'Gaia_Uncertainties')
plt.xlabel('Log G')
plt.ylabel('MAD of predicted star distance')
plt.title('σMAD of predicted star distance per Log G')
plt.legend()
plt.savefig('MAD.png')
plt.show()
#
#
#
# plt.figure()
# MAD = []
# MAD2 = []
# Gaia_un = []
# x = np.arange(1, 5, 0.25)
# for i in x:
#     temp = star_regression[(star_grav_match < i) & (star_grav_match > i - 0.25)]
#     MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
#     temp2 = Star_regression2[(star_grav_match < i) & (star_grav_match > i - 0.25)]
#     MAD2.append(1.4826 * np.median(abs(abs(temp2) - np.median(abs(temp2)))))
#     Gaia_err = parallax_error[starx_ind][(star_grav_match < i) & (star_grav_match > i - 0.25)]
#     Gaia_un.append(np.median(abs(Gaia_err)))
# plt.plot(x, MAD, label = 'StarHorse')
# plt.plot(x, MAD2, label = 'Model')
# plt.plot(x, Gaia_un, linestyle = ':' ,label = 'Gaia_Uncertainties')
# plt.xlabel('Log G')
# plt.ylabel('MAD of predicted star distance')
# plt.title('σMAD of predicted star distance per Log G')
# plt.legend()
# plt.show()