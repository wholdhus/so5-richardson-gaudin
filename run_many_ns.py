from solve_rg_eqs import *
import numpy as np
import pandas as pd
import json

try:
    with open('context.json') as f:
        context = json.load(f)
    RESULT_FP = context['results_filepath']
except:
    print('REQUIRED! context.json file with results_filepath entry')


ls = [4]

for l in ls:

    L = l
    dg = 0.04/L
    g0 = .001/L
    imk = dg
    imv = g0/L
    dg0 = dg
    k = np.arange(1, 2*l+1, 2)*0.5*np.pi/l
    kim = imk*(-1)**np.arange(L)
    kc = np.concatenate((k, kim))

    Gc = 1./np.sum(k)
    Gfs = np.array([0.25, 0.5, 0.9, 1.1, 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])*Gc
    print('Finding initial solutions up to N = {}'.format(4*L-2))
    sols, Ns = bootstrap_g0_multi(L, g0, kc, imv)

    for i, N in enumerate(Ns):
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('l = {}, N = {}'.format(l, N))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('')
        dims = (l, N//2, N//2)
        df = solve_Gs_list(dims, sols[i], Gfs, k, dg=dg, g0=g0, imscale_k=imk,
                           imscale_v=imv)
        df.to_csv(RESULT_FP + '/gap_results/solutions_full_{}_{}_manyN.csv'.format(l, N))
print('')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('Done!!!!!')
