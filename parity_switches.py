from solve_rg_eqs import bootstrap_g0_multi, solve_Gs_list
import numpy as np
import pandas as pd
import pickle
import sys
import os

ls = [int(sys.argv[1])]
# ls = [16, 24, 32, 48, 64]
# ls = [12]
for l in ls:
    Nf = l
    L = 2*l
    k_peri = np.pi*np.arange(2, 2*l, 2)/(2*l) # leaving out the k=0 bit
    k_anti = np.pi*np.arange(1, 2*l+1, 2)/(2*l)
    # eta_peri = np.sin(.5*k_peri)
    # eta_anti = np.sin(.5*k_anti)
    eta_peri = k_peri
    eta_anti = k_anti
    Gc = 1./np.sum(eta_anti)
    Gs = np.array([0.4, 0.7, 1.75, 2.])*Gc

    dg = 0.04/L # step size of g.
    g0 = dg/L # initial value of g
    imk = g0 # scale of the imaginary parts added to k
    imv = g0/L # scale of the imaginary parts used in the initial guess

    eta_anti_c = np.concatenate((eta_anti, (-1)**np.arange(l)*imk))
    eta_peri_c = np.concatenate((eta_peri, (-1)**np.arange(l-1)*imk))

    if os.path.exists('parity_switch_linear_data_l{}N{}.p'.format(l, Nf)):
        print('Existing data exists')
        print('Finding initial variables for periodic b.c.')
        sols0_peri = bootstrap_g0_multi(l-1, g0, eta_peri_c, imv, final_N=Nf+2)
        print('Solving for N-4={} fermions (topological!)'.format(2*(Nf//2-2)))
        dims = (l-1, Nf//2-2, Nf//2-2)
        peri_df_minus4 = solve_Gs_list(dims, sols0_peri[2*(Nf//2-2)], Gs, eta_peri,
                                       dg=dg, g0=g0, imscale_k=imk,
                                       imscale_v=imv)
        out = pickle.load(open('parity_switch_linear_data_l{}N{}.p'.format(l, Nf), 'rb'))
        out['peri_minus4'] = peri_df_minus4
    else:
        dims = (l-1, Nf//2+1, Nf//2+1)
        print('Solving for N+2={} fermions'.format(2*(Nf//2+1)))
        sols0_peri = bootstrap_g0_multi(l-1, g0, eta_peri_c, imv, final_N=Nf+2)
        peri_df_plus2 = solve_Gs_list(dims, sols0_peri[Nf+2], Gs, eta_peri,
                                     dg=dg, g0=g0, imscale_k=imk,
                                     imscale_v=imv)
        print('Solving for N={} fermions'.format(2*Nf//2))
        dims = (l-1, Nf//2, Nf//2)
        peri_df_N = solve_Gs_list(dims, sols0_peri[2*Nf//2], Gs, eta_peri,
                                  dg=dg, g0=g0, imscale_k=imk,
                                  imscale_v=imv)
        print('Solving for N-2={} fermions'.format(2*(Nf//2-1)))
        dims = (l-1, Nf//2-1, Nf//2-1)
        peri_df_minus2 = solve_Gs_list(dims, sols0_peri[Nf-2], Gs, eta_peri,
                                      dg=dg, g0=g0, imscale_k=imk,
                                      imscale_v=imv)
        print('Solving for N-4={} fermions (topological!)'.format(2*(Nf//2-2)))
        dims = (l-1, Nf//2-2, Nf//2-2)
        peri_df_minus4 = solve_Gs_list(dims, sols0_peri[2*(Nf//2-2)], Gs, eta_peri,
                                      dg=dg, g0=g0, imscale_k=imk,
                                      imscale_v=imv)

        print('Finding initial variables for antiperiodic b.c.')
        sols0_anti = bootstrap_g0_multi(l, g0, eta_anti_c, imv, final_N=Nf+2)
        dims = (l, Nf//2+1, Nf//2+1)
        print('Solving for N+2 fermions')
        anti_df_plus2 = solve_Gs_list(dims, sols0_anti[Nf+2], Gs, eta_anti,
                                     dg=dg, g0=g0, imscale_k=imk,
                                     imscale_v=imv)
        print('Solving for N fermions')
        dims = (l, Nf//2, Nf//2)
        anti_df_N = solve_Gs_list(dims, sols0_anti[Nf], Gs, eta_anti,
                                  dg=dg, g0=g0, imscale_k=imk,
                                  imscale_v=imv)
        print('Solving for N-2 fermions')
        dims = (l, Nf//2-1, Nf//2-1)
        anti_df_minus2 = solve_Gs_list(dims, sols0_anti[Nf-2], Gs, eta_anti,
                                      dg=dg, g0=g0, imscale_k=imk,
                                      imscale_v=imv)

        out = {'anti_minus2': anti_df_minus2,
               'anti_plus2': anti_df_plus2,
               'anti_N': anti_df_N,
               'peri_minus4': peri_df_minus4,
               'peri_minus2': peri_df_minus2,
               'peri_plus2': peri_df_plus2,
               'peri_N': peri_df_N}
    print('Storing')
    pickle.dump(out, open('parity_switch_linear_data_l{}N{}.p'.format(l, Nf), 'wb'))
    
