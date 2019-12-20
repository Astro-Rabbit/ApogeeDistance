from astropy.io import fits
import pandas as pd
import numpy as np

data = fits.open('allStarLite-r12-l33.fits')

ids = data[1].data['APOGEE_ID']
TEFF = data[1].data['TEFF']
Grav = data[1].data['LOGG']
Metal = data[1].data['M_H']
parallax = data[1].data['GAIA_PARALLAX']
parallax_err = data[1].data['GAIA_PARALLAX_ERROR']
Kmag = data[1].data['K']
extinction = data[1].data['AK_WISE']

stars = pd.DataFrame()
columns = ['Abs_MAG', 'TEFF', 'Grav', 'Metal']

for i, id in enumerate(ids):

    parallax_error_perc = parallax_err[i] / parallax[i]
    if parallax_error_perc < 0.5:
        Abs_mag = Kmag[i] - 5 * np.log10((1 / parallax[i]) / 10) - extinction[i]
        star = pd.DataFrame([[Abs_mag, TEFF[i], Grav[i], Metal[i]]], columns=columns)
        stars = stars.append(star, ignore_index=True)
        print(i)

stars.to_csv('Full_DATA.csv', index=False)
