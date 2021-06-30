from astropy.io import fits
import pandas as pd
import numpy as np

data = fits.open('APOKASC_cat_v6.6.5.fits')

SDSSdata = fits.open('allStar-dr17-synspec.fits')

# Isochronedata = fits.open('isologg-r12-l33-58932beta.fits')
# #
# # Isochrone_ids = Isochronedata[1].data['APOGEE_ID']

extra_data = fits.open('mass_cal.fits')

# addondata = pd.read_csv('addonAPOKASC/APOKASC3_mrseisg (1).out',delim_whitespace=True)
# addonIDs = pd.read_csv("addonAPOKASC/id_xmatch_apokasc (1).csv")
# addonid = addonIDs['2MASS_ID']
# addonTemp = addondata['TEFF']
# addonMetal = addondata['FEH']
# addonGrav = addondata['LOGG_S_SS']
# addonCarbon = addondata['C_H']
# addonNitro = addondata['N_H']
# addonMass = addondata['M_S_SS']
# addonerr = addondata['SIG_M_S_SS']
# perc1 = addonerr/addonMass
#
# addonTemp = addonTemp[perc1<0.15]
# addonGrav = addonGrav[perc1<0.15]
# addonMetal = addonMetal[perc1<0.15]
# addonCarbon = addonCarbon[perc1<0.15]
# addonNitro = addonNitro[perc1<0.15]
# addonMass = addonMass[perc1<0.15]
# addonid = addonid[perc1<0.15]
#
#
massCalIDs  = extra_data[1].data['APOGEE_ID']
#
# APOKASC_ids = data[1].data['2MASS_ID']
SDSS_ids = SDSSdata[1].data['APOGEE_ID']
# xy, x_ind, y_ind = np.intersect1d(SDSS_ids, APOKASC_ids,return_indices=True)
xy1, x_ind1, y_ind1 = np.intersect1d(SDSS_ids, massCalIDs,return_indices=True)
# # xyIso, x_indIso, y_indIso = np.intersect1d(Isochrone_ids, SDSS_ids,return_indices=True)
#
#
#
# TEFF = SDSSdata[1].data['TEFF'][x_ind]
# Grav = SDSSdata[1].data['LOGG'][x_ind]
# Raw_TEFF = SDSSdata[1].data['TEFF_SPEC'][x_ind]
# Raw_Grav = SDSSdata[1].data['LOGG_SPEC'][x_ind]
#
# APOK_Grav = data[1].data['APOKASC3P_LOGG'][y_ind]
# DWLogG = data[1].data['LOGG_DW'][y_ind]
# APOK_evstate = data[1].data['APOKASC3_CONS_EVSTATES'][y_ind]
#
# Metal = SDSSdata[1].data['M_H'][x_ind]
# Mass = data[1].data['APOKASC2_MASS'][y_ind]
#
# carbon = SDSSdata[1].data['C_FE'][x_ind]
# nitrogen = SDSSdata[1].data['N_FE'][x_ind]
#
# APOK_Grav[APOK_Grav<0] = 0
# DWLogG[DWLogG<0] = 0
# train_grav = APOK_Grav+DWLogG
# train_grav[train_grav==0] = -9999


extra_TEFF = SDSSdata[1].data['TEFF'][x_ind1]
extra_Grav = SDSSdata[1].data['LOGG'][x_ind1]
extra_Metal = SDSSdata[1].data['M_H'][x_ind1]
extra_carbon = SDSSdata[1].data['C_FE'][x_ind1]
extra_nitrogen = SDSSdata[1].data['N_FE'][x_ind1]
extra_mass  = extra_data[1].data['MASS'][y_ind1]
extra_err = extra_data[1].data['MASS_ERR'][y_ind1]
perc = extra_err/extra_mass
massCalIDs = massCalIDs[y_ind1]
s = extra_data[1].data['source'][y_ind1]

extra_TEFF = extra_TEFF[perc<0.1]
extra_Grav = extra_Grav[perc<0.1]
extra_Metal = extra_Metal[perc<0.1]
extra_carbon = extra_carbon[perc<0.1]
extra_nitrogen = extra_nitrogen[perc<0.1]
extra_mass = extra_mass[perc<0.1]
massCalIDs = massCalIDs[perc<0.1]
s = s[perc<0.1]
perc = perc[perc<0.1]


# Iso_TEFF = SDSSdata[1].data['TEFF'][y_indIso]
# Iso_Grav = SDSSdata[1].data['LOGG'][y_indIso]
# Iso_Raw_TEFF = SDSSdata[1].data['TEFF_SPEC'][y_indIso]
# Iso_Raw_Grav = SDSSdata[1].data['LOGG_SPEC'][y_indIso]
# Iso_carbon = SDSSdata[1].data['C_FE'][y_indIso]
# Iso_nitrogen = SDSSdata[1].data['N_FE'][y_indIso]
# Iso_Metal = SDSSdata[1].data['M_H'][y_indIso]
#
# Iso_LogG = Isochronedata[1].data['ISOLOGG'][x_indIso]
#
#
#
# TEFF_comb = np.concatenate((TEFF,Iso_TEFF))
# grav_comb = np.concatenate((Grav,Iso_Grav))
# rawT_comb = np.concatenate((Raw_TEFF,Iso_Raw_TEFF))
# rawG_comb = np.concatenate((Raw_Grav,Iso_Raw_Grav))
# metal_comb = np.concatenate((Metal,Iso_Metal))
# C_comb = np.concatenate((carbon,Iso_carbon))
# N_comb = np.concatenate((nitrogen,Iso_nitrogen))
# train_grav = np.concatenate((train_grav,Iso_LogG))

# stars = pd.DataFrame()
# # columns = ['SDSS_ids', 'APOKASC_ids','TEFF', 'Grav', 'Metal','Mass', 'carbon', 'nitrogen', 'Raw_TEFF', 'Raw_Grav', 'APOK_Grav', 'evstate']
# columns = ['TEFF', 'Grav', 'Metal', 'carbon', 'nitrogen', 'Raw_TEFF', 'Raw_Grav', 'APOK_Grav']
#
# star = pd.DataFrame(data=[TEFF_comb, grav_comb, metal_comb, C_comb, N_comb, rawT_comb, rawG_comb, train_grav], index=columns)
#
# stars = star.T
#
# stars = stars[(stars['TEFF'] > 0) & (stars['Grav'] > 0) & (stars['APOK_Grav'] > 0) & (stars['Metal'] > -9000)  & (stars['carbon'] > -9000)& (stars['nitrogen'] > -9000)]
#
# # stars.to_csv('APOKASC_Massmodel.csv', index=False) # saves modified data to csv file
# stars.to_csv('APOKASC_combinedCalModel.csv', index=False) # saves modified data to csv file

# addonComb_TEFF = np.concatenate((TEFF,addonTemp,extra_TEFF))
# addonComb_Grav = np.concatenate((train_grav,addonGrav,extra_Grav))
# addonComb_C = np.concatenate((carbon,addonCarbon, extra_carbon))
# addonComb_N = np.concatenate((nitrogen,addonNitro, extra_nitrogen))
# addonComb_Metal = np.concatenate((Metal,addonMetal, extra_Metal))
# addonComb_Mass = np.concatenate((Mass,addonMass, extra_mass))
# id_comb = np.concatenate((SDSS_ids[x_ind],addonid, massCalIDs))

addonComb_TEFF = extra_TEFF
addonComb_Grav = extra_Grav
addonComb_C =extra_carbon
addonComb_N = extra_nitrogen
addonComb_Metal = extra_Metal
addonComb_Mass = extra_mass
id_comb = massCalIDs


columns2 = ['SDSS_ids','TEFF', 'Grav', 'Metal','Mass', 'carbon', 'nitrogen', 'Mass_err','source']
massVersion =  pd.DataFrame(data=[id_comb, addonComb_TEFF, addonComb_Grav, addonComb_Metal, addonComb_Mass, addonComb_C, addonComb_N,perc,s], index=columns2)
massVersion = massVersion.T
massVersion = massVersion[(massVersion['TEFF'] > 0) & (massVersion['Mass'] > 0)& (massVersion['Grav'] > 0)  & (massVersion['Metal'] > -9000)  & (massVersion['carbon'] > -9000)& (massVersion['nitrogen'] > -9000)]
massVersion = massVersion[(massVersion['Mass']>0.8) | ((massVersion['Mass']<0.8) & (massVersion['Grav']>3.2))]
massVersion.to_csv('APOKASC_Massmodel_withAddonData.csv', index=False) # saves modified data to csv file
