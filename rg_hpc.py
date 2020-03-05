import sys
import pandas as pd
from solve_rg_eqs import numpy, solve_rgEqs

if len(sys.argv) < 4:
    print('Usage: python rg_hpc.py [L] [N] [G]')
    print('L: number of levels')
    print('N: total number of electrons (half up, half down)')
    print('G: desired final coupling')

L = int(sys.argv[1])
N = int(sys.argv[2])
gf = float(sys.argv[3])
Ne = N//2
Nw = N//2


dg = 0.001*8/L
g0 = dg/N
imk = g0
imv = g0

ks = (1.0*numpy.arange(L) + 1.0)/L

VERBOSE=True
FORCE_GS=True
TOL=10**-10
TOL2=10**-7 # there are plenty of spurious minima around 10**-5
MAXIT=0 # let's use the default value
FACTOR=100
JOBS = 20

dims = (L, Ne, Nw)
es, ws, vars_df, vars = solve_rgEqs(dims, gf, ks, dg=dg, g0=g0, imscale_k=imk,
                                    imscale_v=imv)
vars_df.to_csv('rg_solutions_{}_{}_{}.csv'.format(L, N, gf))