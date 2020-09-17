from solve_rg_eqs import *
import numpy as np
import pandas as pd
import json
import sys
import pickle
import glob
from os import path

try:
    with open('context.json') as f:
        context = json.load(f)
    RESULT_FP = context['results_filepath']
except:
    print('REQUIRED! context.json file with results_filepath entry')


l = int(sys.argv[1])
final_N = int(sys.argv[2])


L = l
dg = 0.04/L
g0 = .001/L
imk = dg
imv = g0/L
# imv =g0
dg0 = dg
k = np.arange(1, 2*l+1, 2)*0.5*np.pi/l
kim = imk*(-1)**np.arange(L)
kc = np.concatenate((k, kim))

Gc = 1./np.sum(k)
Gfs = np.array([0.25, 0.5, 0.9, 1.1, 1.5, 2., 2.5, 3., 3.5, 4.])*Gc

pf = "sols_l{}_N_{}-{}.p".format(l, Ns[0], Ns[-1])

if path.exists(pf):
    print('Already bootstrapped this system.')

else:
    pfs = glob.glob('sols_l{}_N_*''.format(l))
    if len(pfs) > 0:
        pf_Ns = np.array([int(f.split(-)[1].split('.')[0]) for f in pf_Ns])
        best_ind = np.argmax(pf_Ns)
        print('Found results up to N = {}'.format(np.max(pf_Ns)))
        partial_sd = pickle.load(open(pfs[largest_ind], 'rb'))
        if final_N > max(pf_Ns)
            print('Bootstrapping the rest of the way')
            sols, Ns = continue_bootstrap(partial_sd, imv, final_N)
        else:
            print('Trimming existing solutions')
            Ns = np.arange(2, final_N+2, 2)
            sols = [sd['sol_{}'.format(N)] for N in Ns]
    else:
        print('Finding initial solutions up to N = {}'.format(4*L-2))
        sols, Ns = bootstrap_g0_multi(L, g0, kc, imv, final_N = final_N)
    print('Final N from boostrap:')
    print(Ns[-1])

    print('Putting sols into dict for pickling')
    sols_dict = {}
    for i, N in enumerate(Ns):
        sols_dict['sol_{}'.format(N)] = sols[i]
    sols_dict['g0'] = g0
    sols_dict['kc'] = kc
    sols_dict['l'] = L
    sols_dict['Ns'] = Ns
    pf = "sols_l{}_N_{}-{}.p".format(l, Ns[0], Ns[-1])
    print('Pickling dict to {}'.format(pf))
    pickle.dump(sols_dict, open(pf, 'wb'))
print('Loading initial solutions from {}'.format(pf))
sd = pickle.load(open(pf, 'rb'))
# these could be different
g0 = sd['g0']
kc = sd['kc']
Ns = sd['Ns']


for i, N in enumerate(Ns):
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('l = {}, N = {}'.format(l, N))
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('')
    dims = (l, N//2, N//2)
    sol0 = sd['sol_{}'.format(N)]
    df = solve_Gs_list(dims, sol0, Gfs, k, dg=dg, g0=g0, imscale_k=imk,
                       imscale_v=imv)
    df.to_csv(RESULT_FP + '/gap_results/solutions_full_{}_{}_{}.csv'.format(l, N, int(Gfs[-1]/Gc)))
print('')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('Done!!!!!')
