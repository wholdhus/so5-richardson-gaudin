import sys
import pandas as pd
import matplotlib.pyplot as plt
from solve_rg_eqs import np, solve_rgEqs_2, G_to_g
import json

try:
    with open('context.json') as f:
        context = json.load(f)
    RESULT_FP = context['results_filepath']
except:
    print('REQUIRED! context.json file with results_filepath entry')

if len(sys.argv) < 4:
    print('Usage: python rg_hpc.py [L] [N] [G]')
    print('L: number of levels')
    print('N: total number of electrons (half up, half down)')
    print('G: desired final coupling')

L = int(sys.argv[1])
N = int(sys.argv[2])
Gf = float(sys.argv[3])
Ne = N//2
Nw = N//2

dg = 0.04/L
g0 = .01*dg
imk = dg
imv = .01*g0

ks = np.arange(1, 2*L+1, 2)*0.5*np.pi/L

gf = G_to_g(Gf, ks)

print('')
print('Parameters:')
print('Length')
print(L)
print('Fermions')
print(N)
print('Final coupling (physical)')
print(Gf)
print('Final coupling (numerical)')
print(gf)
print('Spectrum')
print(ks)
print('Imaginary part of guesses')
print(imv)
print('')

dims = (L, Ne, Nw)

vars_df = solve_rgEqs_2(dims, Gf, ks, dg=dg, g0=g0, imscale_k=imk,
                        imscale_v=imv, skip=L)

print('Done! Putting things in a CSV')
vars_df.to_csv(RESULT_FP + 'antiperiodic/solutions_full_{}_{}_{}.csv'.format(L, N, np.round(Gf, 3)))

energies = vars_df['energy']

plt.scatter(vars_df['G'], energies)
plt.savefig(RESULT_FP + 'figs/energies_full_{}_{}_{}.png'.format(L, N, np.round(Gf, 3)))
