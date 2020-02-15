import pandas as pd
import matplotlib.pyplot as plt

names = ['05', '10', '15', '20']

for row in names:
    data = pd.read_csv(row+'_DATA.csv')

    Temp = data['TEFF']
    abs_mag = data['Abs_MAG']

    plt.figure()
    plt.xlim(10000, 3000)
    plt.xscale('log')
    plt.ylim(24, 5)
    plt.xlabel('Temp (k)')
    plt.ylabel('Absolute Magnitude (K-band)')
    plt.title('HR diagram of stars in Apogee at '+row+' percent distance error' )
    plt.scatter(Temp, abs_mag, alpha=0.1)
    plt.savefig('HR diagram for ' +row + ' percent distance')
    plt.show()