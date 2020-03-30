import sys
import pandas as pd
import matplotlib.pyplot as plt
from spectral_fun import find_spectral_fun
import json
import numpy as np

try:
    with open('context.json') as f:
        context = json.load(f)
    RESULT_FP = context['results_filepath']
except:
    print('REQUIRED! context.json file with results_filepath entry')

if len(sys.argv) < 6:
    print('Usage: python rg_hpc.py [L] [N] [G] [steps] [states]')
    print('L: number of levels')
    print('N: total number of electrons (half up, half down)')
    print('G: desired coupling')
    print('steps: number of steps taken when integrating')
    print('states: number of eigenstates to find via Lanczos (or -999 for complete)')


L = int(sys.argv[1])
N = int(sys.argv[2])
G = float(sys.argv[3])
steps = int(sys.argv[4])
states = int(sys.argv[5])

ks = np.arange(1, L+1)*np.pi/L


aks1, aks2, omegas, ns = find_spectral_fun(L, N, G, ks, k=None, n_states=states,
                                           steps=steps)

df = pd.DataFrame({})
df['a_k_omega_1'] = aks1
df['a_k_omega_2'] = aks2
df['omega'] = omegas

df2 = pd.DataFrame({})
df2['n_k'] = ns
df2['k'] = np.concatenate((-1*ks[::-1], ks))

print('Done! Putting things in a CSV')
df.to_csv(RESULT_FP + 'spectral_functions_{}_{}_{}.csv'.format(L, N, np.round(G, 3)))
df2.to_csv(RESULT_FP + 'occupations_{}_{}_{}'.format(L, N, np.round(G, 3)))

plt.scatter(omegas, aks1)
plt.savefig(RESULT_FP + 'figs/spectral_function_{}_{}_{}.png'.format(L, N, np.round(G, 3)))
