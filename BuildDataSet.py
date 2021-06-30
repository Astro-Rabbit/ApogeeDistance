from astropy.io import fits
import pandas as pd
import numpy as np

data = fits.open('allStar-dr17-synspec.fits')

clust = pd.read_csv('distances/clusters', delim_whitespace=True, header  = None)

ids = data[1].data['APOGEE_ID']
# TEFF = data[1].data['TEFF']
# Grav = data[1].data['LOGG']
# Metal = data[1].data['M_H']
TEFF = data[1].data['PARAM'][:,0]
Grav = data[1].data['PARAM'][:,1]
Metal = data[1].data['PARAM'][:,3]
parallax2 = data[1].data['GAIAEDR3_PARALLAX']
distance = data[1].data['GAIAEDR3_R_MED_PHOTOGEO']
parallax_err = data[1].data['GAIAEDR3_PARALLAX_ERROR']
Kmag = data[1].data['K']
# carbon = data[1].data['C_FE']
# nitrogen = data[1].data['N_FE']
carbon = data[1].data['PARAM'][:,4]
nitrogen = data[1].data['PARAM'][:,5]
Clust_member = data[1].data['MEMBER']



# gaia_G = data[1].data['GAIA_PHOT_G_MEAN_MAG']
# gaia_BP = data[1].data['GAIA_PHOT_BP_MEAN_MAG']
# gaia_RP = data[1].data['GAIA_PHOT_RP_MEAN_MAG']
# Vscatter = data[1].data['VSCATTER']

stars = pd.DataFrame()
columns = ['APOGEE_ID','Abs_MAG', 'TEFF', 'Grav', 'Metal','carbon', 'nitrogen', 'Apparent', 'Extinction', 'Parallax_error', 'distance', 'clusterID']


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

for i,cluster in enumerate(clust[0]):
    index  =np.where(Clust_member == cluster)[0]
    dist = clust[1][i]*1000
    distance[index] = dist
    parallax_err[index] = 0

parallax_error_perc = parallax_err / parallax2

Abs_mag = Kmag - 5 * np.log10((distance) / 10) - extinction
star = pd.DataFrame(data=[ids, Abs_mag, TEFF, Grav, Metal, carbon, nitrogen, Kmag, extinction, parallax_error_perc,
                          (distance), Clust_member], index=columns)
stars = star.T

# stars = stars[(stars['Abs_MAG'] < 100) & (stars['TEFF'] > 0) & (stars['Grav'] > 0) & (stars['Metal'] > -9000) & (
#         stars['Abs_MAG'] > -40) & (stars['Extinction'] > 0) & (stars['distance']>0)]

# stars = stars.dropna()

stars.to_csv('DR17FULL_PARAMarray.csv', index=False) # saves modified data to csv file
