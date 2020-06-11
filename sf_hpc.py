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

Nup = N//2
Ndown = N//2

ks = np.arange(1, L+1)*np.pi/L
k = L + Nup//2

postfix = '_L{}_N{}_G{}_k{}'.format(L, N, np.round(G, 3), k)

ap, am, omegas, ns = find_spectral_fun(L, N, G, ks, k=None, n_states=states,
                                           steps=steps,
                                           savefile=RESULT_FP+'spectral_funs/matrix_elts'+postfix)

df = pd.DataFrame({})
df['a_k_omega_plus'] = ap
df['a_k_omega_minus'] = am
df['omega'] = omegas

df2 = pd.DataFrame({})
df2['n_k'] = ns
df2['k'] = np.concatenate((-1*ks[::-1], ks))

print('Integrals:')
print(np.trapz(ap, omegas))
print(np.trapz(am, omegas))
print(np.trapz(ap+am, omegas))

print('Done! Putting things in a CSV')
df.to_csv(RESULT_FP+'/spectral_funs/sf' + postfix + '.csv')
# df2.to_csv(RESULT_FP+'/spectral_funs/occupations'+postfix)

plt.scatter(omegas, ap+am)
plt.savefig(RESULT_FP + 'figs/sf' + postfix + '.png')
