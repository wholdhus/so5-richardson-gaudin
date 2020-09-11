from solve_rg_eqs import *
import numpy as np
import pandas as pd

RESULT_FP = '/home/wholdhus/so5_results/'

ls = [12]

for l in ls:
    Ns = np.arange(2, 4*l, 2)
    print('Ns for l = {}'.format(l))
    print(Ns)
    L = l
    dg = 0.04/L
    g0 = .001/L
    imk = dg
    imv = g0/L
    dg0 = dg
    k = np.arange(1, 2*l+1, 2)*0.5*np.pi/l
    Gc = 1./np.sum(k)
    Gfs = np.array([0.25, 0.5, 0.9, 1.1, 1.5, 2., 2.5, 3., 3.5, 4., 4.5, 5.])*Gc
    
    for N in Ns:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('l = {}, N = {}'.format(l, N))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('')
        dims = (l, N//2, N//2)
        df = solve_Gs_list(dims, Gfs, k, dg=dg, g0=g0, imscale_k=imk,
                               imscale_v=imv)
        df.to_csv(RESULT_FP + '/gap_results/solutions_full_{}_{}_manyN.csv'.format(l, N))
print('')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('Done!!!!!')
