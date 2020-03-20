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
columns = ['Abs_MAG', 'TEFF', 'Grav', 'Metal', 'Apparent', 'Extinction', 'Parallax_error', 'parallax']


for i, id in enumerate(ids):

    parallax_error_perc = parallax_err[i] / parallax[i]

    Abs_mag = Kmag[i] - 5 * np.log10((1 / (parallax[i]/1000)) / 10) - extinction[i]
    star = pd.DataFrame([[Abs_mag, TEFF[i], Grav[i], Metal[i], Kmag[i], extinction[i], parallax_error_perc, parallax[i]]], columns=columns)
    stars = stars.append(star, ignore_index=True)
    print(i)

stars = stars[(stars['Abs_MAG'] < 100) & (stars['TEFF'] > 0) & (stars['Grav'] > 0) & (stars['Metal'] > -9000) & (
        stars['Abs_MAG'] > -40)]

stars.to_csv('ALL_withapparent_DATA_r12.csv', index=False)
