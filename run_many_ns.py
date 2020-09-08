from solve_rg_eqs import *
import numpy as np
import pandas as pd

RESULT_FP = '/home/wholdhus/so5_results/'

ls = [32, 64, 96]

for l in ls:
    Ns = np.arange(l-4, l+4, 2)
    L = l
    dg = 0.04/L
    g0 = .01*dg
    imk = dg
    # imv = g0
    imv = g0/L
    dg0 = dg
    k = np.arange(1, 2*l+1, 2)*0.5*np.pi/l
    Gc = 1./np.sum(k)
    Gfs = np.array([0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.1, 1.25, 1.4, 1.5, 1.6, 1.75, 2.])*Gc
    
    for N in Ns:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('l = {}, N = {}'.format(l, N))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('')
        dims = (l, N//2, N//2)
        df = solve_Gs_list(dims, Gfs, k, dg=dg, g0=g0, imscale_k=imk,
                               imscale_v=imv)
        df.to_csv(RESULT_FP + '/gap_results/solutions_full_{}_{}_batch6.csv'.format(l, N))
print('')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('Done!!!!!')
