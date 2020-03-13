import sys
import pandas as pd
import matplotlib.pyplot as plt
from solve_rg_eqs import np, solve_rgEqs_1, solve_rgEqs_2, G_to_g
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

if len(sys.argv) == 5:
    dg = float(sys.argv[4])
    g0 = 0.001*dg
else:
    dg = 0.005/L
    g0 = .1*dg/L
imk = dg
imv = .1*g0/N

ks = (1.0*np.arange(L) + 1.0)/L

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

if L > Ne + Nw:
    print('Below half filling!')
    es, ws, vars_df = solve_rgEqs_1(dims, gf, ks, dg=dg, g0=g0, imscale_k=imk,
                                    imscale_v=imv)
else:
    print('Above half filling!')
    vars_df = solve_rgEqs_2(dims, gf, ks, dg=dg, g0=g0, imscale_k=imk,
                                    imscale_v=imv, skip=4)

print('Done! Putting things in a CSV')
vars_df.to_csv(RESULT_FP + 'solutions_full_{}_{}_{}.csv'.format(L, N, gf))

energies = vars_df['energy']

plt.scatter(vars_df['G'], energies)
plt.savefig(RESULT_FP + 'figs/energies_full_{}_{}_{}.png'.format(L, N, np.round(Gf, 3)))
