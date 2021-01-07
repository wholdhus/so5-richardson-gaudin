import sys
import json
import os
import pickle
from solve_rg_eqs import np, bootstrap_g0, solve_Gs_list, solve_Gs_list_repulsive
import pandas as pd

# dg = 0.01/N
# if Gf < 0:
#     dg *= 10 # works better this way?

# g0 = .1*dg
# imk = g0
# imv = .1*g0

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

ks = np.arange(1, 2*L+1, 2)*0.5*np.pi/L
kc = np.concatenate((ks, imk*(-1)**np.arange(L)))
Gc = 1/np.sum(k)
Grs = np.arange(0.2, Gf+0.2, 0.2)
Gs = Grs*Gc

print('')
print('Parameters:')
print('Length')
print(L)
print('Fermions')
print(N)
print('Relative couplings')
print(Grs)
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

print('Looking for presolved solvs')
N = Ne+Nw
Ni = N
keep_going = True
sol0 = None
pf_1 = 'sols_l{}N{}.p'.format(l, N)
if os.path.exists(pf):
    print('Solution exists!')
    sol0 = pickle.load(open(pf, 'rb'))
else: # Checking in multiple N solutions
    while keep_going:
        pf = "sols_l{}_N_{}-{}.p".format(l, 2, Ni)
        if os.path.exists(pf):
            sd = pickle.load(open(pf, 'rb'))
            # These may be different, important to get them right!
            g0 = sd['g0']
            kc = sd['kc']
            sol0 = sd['sol_{}'.format(N)]
            print('Found a solution! Wow')
            keep_going=False
        if Ni >= 4*L:
            keep_going=False
if sol0 is None:
    print('Fine!!! Doing it myself!!! Ugh!!!')
    sol0 = bootstrap_g0(dims, g0, kc, imscale_v=imv)
    pickle.dump(open(pf_1, 'wb'))
print('Starting for real now!!')
print('')
vars_df = solve_Gs_list(dims, g0, kc, Gs, sol0, dg=dg,
                        imscale_v=imv)

print('Done! Putting things in a CSV')
vars_df.to_csv(RESULT_FP + 'antiperiodic/solutions_list_{}_{}_{}.csv'.format(L, N, np.round(Gf, 3)))
