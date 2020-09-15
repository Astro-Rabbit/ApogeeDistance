import pandas as pd
import tensorflow as tf
import tensorflow_docs as tfdocs
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

training = pd.read_csv('trainsample.csv')
test = pd.read_csv('testsample.csv')

# plot Teff

plt.figure()
plt.hist(training['TEFF'], bins=50, range = [3000,10000])
plt.hist(test['TEFF'], bins=50, range = [3000,10000])
plt.xlabel('Effective Temperature [K]')
plt.ylabel('Star count')
plt.title('Effective Temperature (140 K bins)')
plt.savefig('Effective_Temperature.png')
plt.show()

# plot Log G

plt.figure()
plt.hist(training['Grav'], bins=50, range = [0,6])
plt.hist(test['Grav'], bins=50, range = [0,6])
plt.xlabel('Log(G)')
plt.ylabel('Star count')
plt.title('Log Gravity (0.12 Log(G) bins)')
plt.savefig('Log_Gravity.png')
plt.show()

# plot Metaliccity

plt.figure()
plt.hist(training['Metal'], bins=50, range = [-2.5,0.5])
plt.hist(test['Metal'], bins=50, range = [-2.5,0.5])
plt.xlabel('Metallicity')
plt.ylabel('Star count')
plt.title('Metallicity (0.06 [M/H] bins)')
plt.savefig('Metallicity.png')
plt.show()

# plot parallax

plt.figure()
plt.hist(training['parallax'], bins=50, range =[0,10])
plt.hist(test['parallax'], bins=50,range =[0,10])
plt.xlabel('parallax')
plt.ylabel('Star count')
plt.title('Gaia parallax (0.2 mas bins')
plt.savefig('Gaia_parallax.png')
plt.show()

plt.figure()
plt.scatter(training['TEFF'], training['Abs_MAG'],  alpha=0.1, s=0.01)
plt.xlim(7500, 3000)
plt.xscale('log')
plt.ylim(10, -12)
plt.xlabel('Temp (k)')
plt.ylabel('Absolute Magnitude (K-band)')
plt.title('HR diagram for training dataset')
plt.show()

plt.figure()
plt.scatter(test['TEFF'], test['Abs_MAG'],  alpha=0.1, s=0.1)
plt.xlim(7500, 3000)
plt.xscale('log')
plt.ylim(10, -12)
plt.xlabel('Temp (k)')
plt.ylabel('Absolute Magnitude (K-band)')
plt.title('HR diagram for test dataset')
plt.show()

plt.figure()
plt.scatter(training['TEFF'], training['Grav'],  alpha=0.1, s=0.1)
plt.xlim(7500, 3000)
plt.xscale('log')
plt.ylim(5.5, 0)
plt.xlabel('Temp (k)')
plt.ylabel('Log G')
plt.title('HR diagram for training dataset')
plt.savefig('HRgravtrain.png')
plt.show()

plt.figure()
plt.scatter(test['TEFF'], test['Grav'],  alpha=0.1, s=0.1)
plt.xlim(7500, 3000)
plt.xscale('log')
plt.ylim(5.5, 0)
plt.xlabel('Temp (k)')
plt.ylabel('Log G')
plt.title('HR diagram for test dataset')
plt.savefig('HRgravtest.png')
plt.show()

