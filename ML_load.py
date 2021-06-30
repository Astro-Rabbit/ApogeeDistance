import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import statistics
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import interpolate

data = pd.read_csv('DR17FULL_PARAMarray.csv')
data['trained'] = np.zeros(len(data))

train_data = pd.read_csv('trainsample_withcarbon_param.csv')
index = train_data.pop('Unnamed: 0')

data['trained'].iloc[index] = 1

id = data.pop('APOGEE_ID')
Apparent = data.pop('Apparent')
Extinction = data.pop('Extinction')
parallax_error = data.pop('Parallax_error')
distance = data.pop('distance')
clust_member = data.pop('clusterID')
trained = data.pop('trained')

train_stats = pd.read_csv('trainstatsDR17_param_cut.csv', index_col=0)
labels = data.pop('Abs_MAG')


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_data = norm(data)

model = tf.keras.models.load_model('DR17_param.h5')

prediction = model.predict(normed_data).flatten()
perc_errors = (labels - prediction) / labels

distance_prediction = 10 * 10 ** ((prediction + Extinction - Apparent) / -5)
regression = (distance - distance_prediction) / distance

clusters = clust_member.dropna().drop_duplicates()

# for i, row in clusters.iteritems():
#     if (clust_member == row).value_counts()[1] >= 5:
#         plt.figure()
#         plt.hist(regression[clust_member== row],bins = 20)
#         plt.title(row +' '+str(distance[clust_member== row].iloc[0])+' Parsecs')
#         plt.xlabel('Percent error')
#         plt.ylabel('Number of stars')
#         plt.xlim([-1,1])
#         plt.show()
#         print()

# plt.figure()
# plt.scatter(distance, regression, s=1)
# plt.xlabel('True Values')
# plt.ylabel('regression')
# plt.ylim([-1,1])
# plt.show()
#
# plt.figure()
# plt.scatter(data['TEFF'], prediction, c=data['Grav'], alpha=0.1, s=0.1)
# plt.xlim(7500, 3000)
# plt.xscale('log')
# plt.ylim(10, -12)
# plt.xlabel('Temp (k)')
# plt.ylabel('Absolute Magnitude (K-band)')
# plt.title('HR diagram')
# plt.show()
#
# for i in range(1, 6):
#     plt.figure()
#     plt.scatter(distance[(data['Grav'] < i) & (data['Grav'] > i - 1)],
#                 regression[(data['Grav'] < i) & (data['Grav'] > i - 1)], s=0.01)
#     plt.ylim([-1, 1])
#     plt.xlim([-100, 30000])
#     plt.title('regression plot for Grav:' + str(i) + ' -No flattening model')
#     plt.show()
#
#     print()


# plt.figure()
# MAD = []
# model2_err_un = []
# x = np.arange(1, 5, 0.25)
# for i in x:
#     temp = perc_errors[(data['Grav'] < i) & (data['Grav'] > i - 0.25)].dropna()
#     MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
# plt.plot(x, MAD)
# plt.xlabel('Log G')
# plt.ylabel('MAD of predicted magnitudes')
# plt.title('MAD of predicted magnitudes per Log G')
# # plt.savefig('MAD_grav.png')
# plt.show()

test_data = pd.read_csv('testsample_withcarbon_param.csv')
test_data.pop('APOGEE_ID')
test_Apparent = test_data.pop('Apparent')
test_Extinction = test_data.pop('Extinction')
test_data.pop('Parallax_error')
test_distance = test_data.pop('distance')
test_data.pop('clusterID')
test_data.pop('Unnamed: 0')

test_labels = test_data.pop('Abs_MAG')

n = norm(test_data)
t = model.predict(n).flatten()
p = (test_labels - t) / test_labels

dp = 10 * 10 ** ((t + test_Extinction - test_Apparent) / -5)
regression = (test_distance - dp) / test_distance

plt.figure()
MAD = []
model2_err_un = []
x = np.arange(1, 5, 0.25)
for i in x:
    temp = regression[(test_data['Grav'] < i) & (test_data['Grav'] > i - 0.25)]
    MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
plt.plot(x, MAD)
plt.xlabel('Log G')
plt.ylabel('MAD of predicted star distance')
plt.title('MAD of predicted star distance per Log G')
# plt.savefig('MAD_grav.png')
plt.show()

[M, B] = np.polyfit(x, MAD, 1)

uncertaintyDist = (data['Grav'] * M + B) * distance_prediction

uncertaintyABS_mag = (np.ones(len(data)) * .1) * labels

#
trainmass_stats = pd.read_csv('trainstatsMassModel.csv', index_col=0)


def norm_Mass(x):
    return (x - trainmass_stats['mean']) / trainmass_stats['std']


normed_massData = norm_Mass(data)
Mass_model = tf.keras.models.load_model('Masses.h5')

Mass_predictions = Mass_model.predict(normed_massData).flatten()

M_mass = -0.03476860319362355
B_mass = 0.18722328796094467

uncertaintyMass = ((data['Grav'] * M_mass) + B_mass) * Mass_predictions

data['bounded'] = np.zeros(len(data))
data['bounded'][((data['TEFF'] < 4500) & (data['Grav'] > 3)) | (data['Grav'] > 4.9) | (data['TEFF'] > 6800) | (
            (data['TEFF'] > 5500) & (data['Grav'] < 3.1))|((data['Grav'] < 1.1))|(data['Grav'] < (0.000906*data['TEFF']-2.525))|(
        (data['Grav'] < 2)&(data['TEFF'] > 4800)&(data['TEFF'] < 5500))|((data['Grav']>(0.002*data['TEFF']-6.2))&(data['Grav']<4.1))|((data['Metal']<-0.5))] = 2

bound = data.pop('bounded')

fe = [0.25, 0, -0.25, -0.50, -0.75, -1.00, -1.25, -1.50, -1.75, -2.00, -2.50, -3.00, -3.50, -4.00]
t = []
mass = []
feh_m = []
for f in fe:
    if f < 0:
        sign = 'm'
        f_0 = f * -1
    else:
        sign = 'p'
        f_0 = f
    feh = np.loadtxt("MIST_v1.2_vvcrit0.4_basic_isos\MIST_v1.2_vvcrit0.4_basic_isos\MIST_v1.2_vvcrit0.4_basic_isos"
                     "/MIST_v1.2_feh_" + sign + "{:.2f}".format(f_0) + "_afe_p0.0_vvcrit0.4_basic.iso")
    split = np.where(np.ediff1d(feh[:, 0]) < 1)
    splitAge = np.split(feh, split[0] + 1, axis=0)

    for age in splitAge:
        a = age[age[:, 24] > 4]
        mass.append(a[:, 2].mean())
        t.append(a[:, 1].mean())
        feh_m.append(f)

t = np.asarray(t)
mass = np.asarray(mass)
feh_m = np.asarray(feh_m)

t_final = t[np.logical_not(np.isnan(t))]
mass_final = mass[np.logical_not(np.isnan(mass))]
feh_m_final = feh_m[np.logical_not(np.isnan(mass))]

test_function = interpolate.interp2d(feh_m_final, mass_final, t_final, kind='cubic')

age_prediction = []
for i, M in enumerate(Mass_predictions):
    if data['Grav'].iloc[i] < 3.5:
        age_prediction.append(test_function(data['Metal'].iloc[i], M)[0])
    else:
        age_prediction.append(None)
age_prediction = np.array(age_prediction)

ages = 10 ** age_prediction.astype('float')

bitmask = trained.astype('int') | bound.astype('int')

nmsu = fits.open('nmsudist.fits')
diso = nmsu[1].data['diso']




train_mass1 = pd.read_csv('APOKASC_Massmodel_withAddonData.csv')

xy1, x_ind1, y_ind1 = np.intersect1d(id, train_mass1['SDSS_ids'],return_indices=True)

data['train_mass'] = np.zeros(len(data))
data['train_mass'][x_ind1] = train_mass1['Mass'][y_ind1]
trained_mass = data.pop('train_mass')
trained_mass.replace(0, np.nan, inplace=True)

data['source'] = np.zeros(len(data))+7
data['source'][x_ind1] = train_mass1['source'][y_ind1]
s = data.pop('source')
s = s.replace(0,4)
s = s.replace(1,8)
s = s.replace(2,16)
s.replace(7, 0, inplace=True)
s = s.astype('int')

bitmask = bitmask|s


columns = ['ID', 'TEFF', 'LOGG', 'M_H', 'C_FE', 'N_FE', 'GAIAEDR3_DIST', 'EXTINCTION', 'MAG', 'ABS_MAG'
    , "ABS_MAG_ERR", 'DISTANCE', 'DISTANCE_ERR', 'MASS', 'MASS_ERR', 'TRAIN_MASS','AGE', 'BITMASK']
dataModel = pd.DataFrame(
    data=[id, data['TEFF'], data['Grav'], data['Metal'], data['carbon'], data['nitrogen'], distance, Extinction,
          Apparent, prediction, uncertaintyABS_mag, distance_prediction,
          uncertaintyDist, Mass_predictions, uncertaintyMass,trained_mass, ages, bitmask], index=columns).T

tab = Table.from_pandas(dataModel)
for name in columns:
    if name != 'ID':
        d = tab[name].data
        newD = Table.Column(d, dtype='float')
        tab.replace_column(name, newD)

d = tab['BITMASK'].data
newD = Table.Column(d, dtype='int')
tab.replace_column('BITMASK', newD)

tab['NMSU_DIST'] = diso

# tab.write('APOGEE_DistMass-DR17.fits', format='fits')

#  bitwise and:  &
#  bitwise or :   |


# astroNN

# nn = fits.open('apogee_astroNN-DR17.fits')[1]
#
# plt.figure()
# plt.scatter(nn.data['DIST'], distance_prediction, s=1)
# plt.xlabel('True Values')
# plt.ylabel('regression')
# plt.ylim([-1,1])
# plt.show()


# plt.figure()
# plt.scatter(data['TEFF'][ages!=None], data['Grav'][ages!=None], s= 1, c = Mass_predictions[ages!=None])
# plt.xlabel('TEFF')
# plt.ylabel('Grav')
# plt.colorbar()
# plt.ylim([5.5,-.5])
# plt.xlim([8000,3000])
# plt.show()
#
#
# plt.figure()
# plt.scatter(Mass_predictions[ages!=None], ages, s= 1, c = data['Metal'][ages!=None])
# plt.xlabel('Mass')
# plt.ylabel('Age')
# plt.colorbar()
# plt.show()

gaiaError = abs(distance_prediction-distance)/distance

plt.figure()
plt.scatter(data['TEFF'][trained==1], data['Grav'][trained==1], s= 0.01)
plt.xlabel('TEFF')
plt.ylabel('Grav')
plt.ylim([5.5,-.5])
plt.xlim([7000,3000])
plt.title('HR distance training set')
plt.show()

plt.figure()
plt.scatter(data['TEFF'], data['Grav'], s= 0.01, c = gaiaError)
plt.xlabel('TEFF')
plt.ylabel('Grav')
plt.ylim([5.5,-.5])
plt.xlim([7000,3000])
plt.colorbar()
plt.clim(0,2)
plt.title('HR distance gaia error')
plt.show()

plt.figure()
plt.scatter(data['TEFF'], data['Grav'], s= 0.1, c = abs(Mass_predictions-trained_mass)/trained_mass)
plt.xlabel('TEFF')
plt.ylabel('Grav')
plt.ylim([5.5,-.5])
plt.xlim([7000,3000])
plt.colorbar()
plt.clim(0,1)
plt.title('HR mass error')
plt.show()

plt.figure()
plt.scatter(data['TEFF'][bound==0], data['Grav'][bound==0], s= 0.1,c = age_prediction[bound==0])
plt.xlabel('TEFF')
plt.ylabel('Grav')
plt.ylim([5.5,-.5])
plt.xlim([7000,3000])
plt.colorbar()
plt.title('HR ages for in training bound stars')
plt.show()


plt.figure()
plt.scatter(data['TEFF'][trained==1], data['Metal'][trained==1], s= 0.01)
plt.xlabel('TEFF')
plt.ylabel('Metal')
plt.xlim([7000,3000])
plt.title('Temp v Metal: distance training set')
plt.show()

plt.figure()
plt.scatter(data['TEFF'], data['Metal'], s= 0.01, c = gaiaError)
plt.xlabel('TEFF')
plt.ylabel('Metal')
plt.xlim([7000,3000])
plt.colorbar()
plt.clim(0,2)
plt.title('Temp v Metal: distance gaia error')
plt.show()

plt.figure()
plt.scatter(data['TEFF'], data['Metal'], s= 0.1, c = abs(Mass_predictions-trained_mass)/trained_mass)
plt.xlabel('TEFF')
plt.ylabel('Metal')
plt.xlim([7000,3000])
plt.colorbar()
plt.clim(0,1)
plt.title('Temp v Metal: mass error')
plt.show()

plt.figure()
plt.scatter(data['TEFF'][bound==0], data['Metal'][bound==0], s= 0.1,c = age_prediction[bound==0])
plt.xlabel('TEFF')
plt.ylabel('Metal')
plt.xlim([7000,3000])
plt.colorbar()
plt.title('Temp v Metal: ages for in training bound stars')
plt.show()



plt.figure()
plt.scatter(data['Grav'][trained==1], data['Metal'][trained==1], s= 0.01)
plt.xlabel('Log G')
plt.ylabel('Metal')
plt.title('Log G v Metal: distance training set')
plt.show()

plt.figure()
plt.scatter(data['Grav'], data['Metal'], s= 0.01, c = gaiaError)
plt.xlabel('Log G')
plt.ylabel('Metal')
plt.colorbar()
plt.clim(0,2)
plt.title('Log G v Metal: distance gaia error')
plt.show()

plt.figure()
plt.scatter(data['Grav'], data['Metal'], s= 0.1, c = abs(Mass_predictions-trained_mass)/trained_mass)
plt.xlabel('Log G')
plt.ylabel('Metal')
plt.colorbar()
plt.clim(0,1)
plt.title('Log G v Metal: mass error')
plt.show()

plt.figure()
plt.scatter(data['Grav'][bound==0], data['Metal'][bound==0], s= 0.1,c = age_prediction[bound==0])
plt.xlabel('Log G')
plt.ylabel('Metal')
plt.colorbar()
plt.title('Log G v Metal: ages for in training bound stars')
plt.show()


plt.figure()
MAD = []
x = np.arange(1, 5, 0.25)
for i in x:
    temp = gaiaError[(data['Grav'] < i) & (data['Grav'] > i - 0.25) & (trained==0)].dropna()
    MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
plt.plot(x, MAD)
plt.xlabel('Log G')
plt.ylabel('MAD of predicted star distance')
plt.title('MAD of predicted star distance per Log G for Non-training set stars')
# plt.savefig('MAD_grav.png')
plt.show()

plt.figure()
MAD = []
x = np.arange(3000, 7000, 100)
for i in x:
    temp = gaiaError[(data['TEFF'] < i) & (data['TEFF'] > i - 0.25) & (trained==0)].dropna()
    MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
plt.plot(x, MAD)
plt.xlabel('Temp')
plt.ylabel('MAD of predicted star distance')
plt.title('MAD of predicted star distance per temp for Non-training set stars')
# plt.savefig('MAD_grav.png')
plt.show()




massERR = abs(Mass_predictions-trained_mass)/trained_mass

plt.figure()
MAD = []
x = np.arange(1, 5, 0.25)
for i in x:
    temp = massERR[(data['Grav'] < i) & (data['Grav'] > i - 0.25)].dropna()
    MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
plt.plot(x, MAD)
plt.xlabel('Log G')
plt.ylabel('MAD of predicted star distance')
plt.title('MAD of predicted star Mass per Log G for Non-training set stars')
# plt.savefig('MAD_grav.png')
plt.show()


plt.figure()
plt.scatter(distance, (distance_prediction-distance)/distance, s=1)
plt.xlabel('True Values')
plt.ylabel('residual')
plt.ylim([-1,1])
plt.show()