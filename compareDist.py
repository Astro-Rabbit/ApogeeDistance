import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.io import fits

data1 = fits.open('allStarLite-r13-l33-58932beta.fits')
full_ids = data1[1].data['APOGEE_ID']
Gaia_dist = 1/(data1[1].data['GAIA_PARALLAX']/1000)
parallax_err = data1[1].data['GAIA_PARALLAX_ERROR']/data1[1].data['GAIA_PARALLAX']
Grav = data1[1].data['LOGG']

Grav = Grav[(abs(parallax_err)<.15)& (Gaia_dist>0)]
full_ids = full_ids[(abs(parallax_err)<.15)& (Gaia_dist>0)]
Gaia_dist = Gaia_dist[(abs(parallax_err)<.15)& (Gaia_dist>0)]


data = fits.open('apogee_astroNN-DR16.fits')
AstroNN_ids = data[1].data['APOGEE_ID']
AstroNN_dist = data[1].data['DIST']
err  = data[1].data['DIST_ERROR']

NN_err = err/AstroNN_dist


AstroNN_ids = AstroNN_ids[(AstroNN_dist>0)&(NN_err<0.50)]
AstroNN_dist = AstroNN_dist[(AstroNN_dist>0)&(NN_err<0.50)]



xy, x_ind, y_ind = np.intersect1d(full_ids, AstroNN_ids,return_indices=True)

NN_match = AstroNN_dist[y_ind]
Gaia_match = Gaia_dist[x_ind]
grav_match = Grav[x_ind]


regression = (Gaia_match - NN_match)/Gaia_match

plt.figure()
MAD = []
x = np.arange(1, 5, 0.5)
for i in x:
    temp = regression[(grav_match < i) & (grav_match > i - 0.5)]
    MAD.append(1.4826 * np.median(abs(abs(temp) - np.median(abs(temp)))))
plt.plot(x, MAD)
plt.xlabel('Log G')
plt.ylabel('MAD of predicted star distance')
plt.title('MAD of predicted star distance per Log G')
plt.savefig('MAD_grav.png')
plt.show()

data = pd.read_csv('testsample_fixederror.csv')
data.pop('Unnamed: 0')
