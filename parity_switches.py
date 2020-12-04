from solve_rg_eqs import bootstrap_g0_multi, solve_Gs_list
import numpy as np
import pandas as pd
import pickle



# ls = [32, 64]
ls = [6]
for l in ls:
    Nf = l
    L = 2*l
    k_peri = np.pi*np.arange(2, 2*l, 2)/(2*l) # leaving out the k=0 bit
    k_anti = np.pi*np.arange(1, 2*l+1, 2)/(2*l)
    Gc = 1./np.sum(k_anti)
    Gs = np.array([0.2, 0.5, 0.7, 1.25, 1.75, 2.])*Gc
    eta_peri = np.sin(.5*k_peri)
    eta_anti = np.sin(.5*k_anti)

    dg = 0.1/L # step size of g.
    g0 = dg/L # initial value of g
    imk = dg # scale of the imaginary parts added to k
    imv = g0/L # scale of the imaginary parts used in the initial guess

    eta_anti_c = np.concatenate((eta_anti, (-1)**np.arange(l)*imk))
    eta_peri_c = np.concatenate((eta_peri, (-1)**np.arange(l-1)*imk))

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

    print('Finding initial variables for periodic b.c.')
    sols0_peri = bootstrap_g0_multi(l-1, g0, eta_peri_c, imv, final_N=Nf+2)
    dims = (l-1, Nf//2+1, Nf//2+1)
    print('Solving for N+2 fermions')
    peri_df_plus2 = solve_Gs_list(dims, sols0_peri[Nf+2], Gs, eta_peri,
                                 dg=dg, g0=g0, imscale_k=imk,
                                 imscale_v=imv)
    print('Solving for N fermions')
    dims = (l-1, Nf//2, Nf//2)
    peri_df_N = solve_Gs_list(dims, sols0_anti[Nf], Gs, eta_peri,
                              dg=dg, g0=g0, imscale_k=imk,
                              imscale_v=imv)
    print('Solving for N-2 fermions')
    dims = (l-1, Nf//2-1, Nf//2-1)
    peri_df_minus2 = solve_Gs_list(dims, sols0_anti[Nf-2], Gs, eta_peri,
                                  dg=dg, g0=g0, imscale_k=imk,
                                  imscale_v=imv)
    print('Solving for N-4 fermions (topological!)')
    dims = (l-1, Nf//2-2, Nf//2-2)
    peri_df_minus4 = solve_Gs_list(dims, sols0_anti[Nf-4], Gs, eta_peri,
                                  dg=dg, g0=g0, imscale_k=imk,
                                  imscale_v=imv)
    print('Storing')
    out = {'anti_minus2': anti_df_minus2,
           'anti_plus2': anti_df_plus2,
           'anti_N': anti_df_N,
           'peri_minus2': peri_df_minus2,
           'peri_plus2': peri_df_plus2,
           'peri_N': peri_df_N}
    pickle.dump(out, open('parity_switch_data_l{}N{}.p'.format(l, Nf), 'wb'))
