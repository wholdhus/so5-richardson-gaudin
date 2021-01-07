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

l = int(sys.argv[1])
N = int(sys.argv[2])
Gf = float(sys.argv[3])
Ne = N//2
Nw = N//2

dg = 0.01/l
g0 = .1*dg*np.sign(Gf)
imk = dg
imv = g0/l

k = np.arange(1, 2*l+1, 2)*0.5*np.pi/l
kc = np.concatenate((k, imk*(-1)**np.arange(l)))
Gc = 1/np.sum(k)
Grs = np.sign(Gf)*np.arange(0.2, np.abs(Gf)+0.2, 0.2)
Gs = Grs*Gc

print('')
print('Parameters:')
print('Length/levels')
print(2*l)
print(l)
print('Fermions')
print(N)
print('Relative couplings')
print(Grs)
print('Spectrum')
print(k)
print('Imaginary part of guesses')
print(imv)
print('dg')
print(dg)
print('g0')
print(g0)
print('')


dims = (l, Ne, Nw)

print('Looking for presolved solvs')
N = Ne+Nw
Ni = N
keep_going = True
sol0 = None
pf_1 = 'pickles/sols_l{}N{}.p'.format(l, N)
try:
    sol0 = pickle.load(open(pf_1, 'rb'))
except:
    print('N=N solution lacking woops')
else: # Checking in multiple N solutions
    while keep_going:
        pf = "pickles/sols_l{}_N_{}-{}.p".format(l, 2, Ni)
        if os.path.exists(pf):
            sd = pickle.load(open(pf, 'rb'))
            # These may be different, important to get them right!
            g0 = sd['g0']
            kc = sd['kc']
            sol0 = sd['sol_{}'.format(N)]
            print('Found a solution! Wow')
            keep_going=False
        if Ni >= 4*l:
            keep_going=False
        Ni += 2
if sol0 is None:
    print('Fine!!! Doing it myself!!! Ugh!!!')
    sol0 = bootstrap_g0(dims, g0, kc, imscale_v=imv)
    pickle.dump(sol0, open(pf_1, 'wb'))
print('Starting for real now!!')
print('')
if Gf > 0:
    vars_df = solve_Gs_list(dims, g0, kc, Gs, sol0, dg=dg,
                            imscale_v=imv)
else:
    vars_df = solve_Gs_list_repulsive(dims, g0, kc, Gs, sol0, dg=dg,
                                      imscale_v=imv)

print('Done! Putting things in a CSV')
vars_df.to_csv(RESULT_FP + 'antiperiodic/solutions_list_{}_{}_{}.csv'.format(l, N, np.round(Gf, 3)))
