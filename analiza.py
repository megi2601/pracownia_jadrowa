import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import scipy as sp
from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline




import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'


filenames = [f'{p}{n}.lst.lst' for p in ["Al", 'Cu', 'Pb'] for n in range(6)]
filenames.append('Cu6.lst.lst')

upperchannel = 1200
lowerchannel = 800


def line(x, a, b):
    return a*x +b

def fit(k, a, b, c, d, N):
    return N/np.sqrt(d*d)*np.exp(-(k-c)**2/2/d/d) + a*k + b

cols = ["N", "niepewnosc", 'ln', 'dln']
tables = {"Al":pd.DataFrame(columns=cols), "Pb":pd.DataFrame(columns=cols), "Cu":pd.DataFrame(columns=cols)}


# fig, ax = plt.subplots(figsize=(8, 4))
# for name in filenames:
#     data = np.loadtxt("dane/"+name)
#     x=data.T[0][lowerchannel:upperchannel]
#     y=data.T[1][lowerchannel:upperchannel]
#     popt, pcov = curve_fit(fit, x, y,[0.01, 10, 990, 60, 1e4])
#     p=name[:2]
#     tables[p].loc[len(tables[p].index)] = [popt[-1], np.sqrt(pcov[-1][-1]), np.log(popt[-1]), np.sqrt(pcov[-1][-1])/popt[-1]] 
#     if p=='Pb' and name[2] in ['0', '5']:
#         label = 'Bez osłony' if name[2] == '0' else '15 płytek'
#         plt.plot(data.T[0], data.T[1], label=label)
#         #plt.plot(x, fit(x, *popt))
# plt.legend()
# plt.tight_layout(pad=2)
# plt.xlabel('Numer kanału')
# plt.ylabel('Liczba zliczeń')
# plt.savefig("Pb"+'.png', dpi=350)
# plt.clf()


for name in filenames:
    data = np.loadtxt("dane/"+name)
    x=data.T[0][lowerchannel:upperchannel]
    y=data.T[1][lowerchannel:upperchannel]
    popt, pcov = curve_fit(fit, x, y,[0.01, 10, 990, 60, 1e4])
    p=name[:2]
    tables[p].loc[len(tables[p].index)] = [popt[-1], np.sqrt(pcov[-1][-1]), np.log(popt[-1]), np.sqrt(pcov[-1][-1])/popt[-1]] 
    plt.plot(x, y, label='Widmo')
    plt.plot(x, fit(x, *popt), label='Dopasowanie', color='red', linewidth=2)
    plt.legend()
    plt.tight_layout(pad=2)
    plt.xlabel('Numer kanału')
    plt.ylabel('Liczba zliczeń')
    plt.savefig(name[:3]+'.png', dpi=350)
    plt.clf()


blaszki = np.loadtxt('blaszki.csv', delimiter=',')


tables['Pb']['x'] = np.append(0, blaszki[:5, 0])
tables['Cu']['x']= np.append(0, blaszki[5:11, 0])
tables['Al']['x'] = np.append(0, blaszki[11:, 0])
tables['Pb']['dx'] = np.append(0, blaszki[:5, 1])
tables['Cu']['dx']= np.append(0, blaszki[5:11, 1])
tables['Al']['dx'] = np.append(0, blaszki[11:, 1])


res = {}

fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))
#for element in tables.keys():
for i in range(3):
    element = list(tables.keys())[i]
    x = tables[element]['x'][1:]
    y = tables[element]['ln'][1:]
    popt, pcov = curve_fit(line, x, y, sigma=tables[element]['dln'][1:])
    res[element]=[ -popt[0], np.sqrt(pcov[0][0])]
    ax[i].plot(tables[element]['x'], line(tables[element]['x'], *popt), color='gray', linewidth=1, label="Dopasowanie")
    # ax[i].errorbar(tables[element]['x'][1:], tables[element]['ln'][1:], label='Dane eksperymentalne', color='orange', yerr=tables[element]['dln'][1:], xerr=tables[element]['dx'][1:], marker='.')
    # ax[i].errorbar([tables[element]['x'][0]], [tables[element]['ln'][0]], marker='*', color='red', label='Pomiar bez osłony', yerr=[tables[element]['dln'][0]],xerr=[tables[element]['dx'][0]])
    ax[i].scatter(tables[element]['x'][1:], tables[element]['ln'][1:], label='Dane eksperymentalne', color='orange')
    ax[i].scatter([tables[element]['x'][0]], [tables[element]['ln'][0]], marker='*', color='red', label='Pomiar bez osłony')
    ax[i].set_xlabel(r"$x_N$ [mm]")
    ax[i].set_ylabel(r"$\ln{(n)}$")
    ax[i].set_title(element)
    ax[i].legend()

plt.tight_layout(pad=1.5)
plt.savefig('dop.png', dpi=350)
plt.clf()

print(res)


wsp = pd.read_csv('wsp.csv')

for element in tables.keys():   
#     plt.scatter(wsp['E'], wsp['Al'], label=element)
    spl = CubicSpline(wsp['E'], wsp[element])   
    print(element, ' ', res[element][0], ' ', spl(0.66166))

# plt.legend()
# plt.show()



