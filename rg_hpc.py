import sys
import pandas as pd
from solve_rg_eqs import np, solve_rgEqs, solve_rgEqs_2, G_to_g
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

dg = 0.01/L
g0 = .1*dg
imk = dg
# imv = g0
imv = g0/L

# dg = 0.01/N
# if Gf < 0:
#     dg *= 10 # works better this way?

# g0 = .1*dg
# imk = g0
# imv = .1*g0

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
print('dg')
print(dg)
print('g0')
print(g0)
print('')


dims = (L, Ne, Nw)

vars_df = solve_rgEqs_2(dims, Gf, ks, dg=dg, g0=g0, imscale_k=imk,
                        imscale_v=imv, skip=20*L)

print('Done! Putting things in a CSV')
vars_df.to_csv(RESULT_FP + 'antiperiodic/solutions_full_{}_{}_{}.csv'.format(L, N, np.round(Gf, 3)))
