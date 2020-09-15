from astropy.io import fits
import pandas as pd
import numpy as np

data = fits.open('allStarLite-r13-l33-58932beta.fits')

ids = data[1].data['APOGEE_ID']
TEFF = data[1].data['TEFF']
Grav = data[1].data['LOGG']
Metal = data[1].data['M_H']
parallax = data[1].data['GAIA_PARALLAX']
parallax_err = data[1].data['GAIA_PARALLAX_ERROR']
Kmag = data[1].data['K']
carbon = data[1].data['C_FE']
nitrogen = data[1].data['N_FE']


gaia_G = data[1].data['GAIA_PHOT_G_MEAN_MAG']
gaia_BP = data[1].data['GAIA_PHOT_BP_MEAN_MAG']
gaia_RP = data[1].data['GAIA_PHOT_RP_MEAN_MAG']
Vscatter = data[1].data['VSCATTER']

stars = pd.DataFrame()
columns = ['APOGEE_ID','Abs_MAG', 'TEFF', 'Grav', 'Metal','carbon', 'nitrogen', 'Apparent', 'Extinction', 'Parallax_error', 'parallax', 'G', 'BP', 'RP',
           'vscatter']


def extiction():
    # array input/output
    ak= data[1].data['AK_WISE']
    gd=np.where(np.core.defchararray.find( data[1].data['AK_TARG_METHOD'],'IRAC') >= 0)[0]
    ak[gd]= data[1].data['AK_TARG'][gd]
    gd=np.where((abs( data[1].data['GLAT']) > 16) & ((0.302* data[1].data['SFD_EBV'] < 1.2*ak) | (ak<0)) )[0]
    ak[gd]=0.302 *  data[1].data['SFD_EBV'][gd]
    return ak

if True:
    extinction = extiction()
else:
    extinction = data[1].data['AK_WISE']

offset = 0.052 # enter gaia parallax offset here


parallax_error_perc = parallax_err / parallax

Abs_mag = Kmag - 5 * np.log10((1 / ((parallax+offset) / 1000)) / 10) - extinction
star = pd.DataFrame(data=[ids, Abs_mag, TEFF, Grav, Metal, carbon, nitrogen, Kmag, extinction, parallax_error_perc,
                          (parallax + offset),gaia_G, gaia_BP, gaia_RP, Vscatter], index=columns)
stars = star.T

stars = stars[(stars['Abs_MAG'] < 100) & (stars['TEFF'] > 0) & (stars['Grav'] > 0) & (stars['Metal'] > -9000) & (
        stars['Abs_MAG'] > -40) & (stars['Extinction'] > 0) & (stars['parallax']>0)]

stars.to_csv('r13_withCN.csv', index=False) # saves modified data to csv file
