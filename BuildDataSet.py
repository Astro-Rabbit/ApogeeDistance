from astropy.io import fits
import pandas as pd

data = fits.open('allStarLite-r12-l33.fits')

ids = data[1].data['APOGEE_ID']
TEFF = data[1].data['TEFF']
Grav = data[1].data['LOGG']
Metal = data[1].data['M_H']
parallax = data[1].data['GAIA_PARALLAX']
parallax_err = data[1].data['GAIA_PARALLAX_ERROR']

stars = pd.DataFrame()
columns = ['parallax', 'TEFF', 'Grav', 'Metal']

for i, id in enumerate(ids):

    parallax_error_perc = parallax_err[i]/parallax[i]
    if parallax_error_perc < 0.05:
        star = pd.DataFrame([[1/parallax[i], TEFF[i], Grav[i], Metal[i]]], columns=columns)
        stars = stars.append(star, ignore_index=True)
        print(i)


stars.to_csv('DATA.csv', index = False)