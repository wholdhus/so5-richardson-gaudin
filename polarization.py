from solve_rg_eqs import solve_Gs_list, G_to_g, unpack_dims, bootstrap_g0
import numpy as np
import pickle
import sys

l = int(sys.argv[1])
N = int(sys.argv[2])
L = 2*l
Ne = N//2
if N <= 2*l: # below half-filling
    Nws = np.arange(0, Ne+1, 1) # Not going to check Nw > Ne because that's for Sz > 0
else:
    # above half filling
    # Can fit at most L spin down fermions, L spin up
    # Nw = Ne + Sz = Ne + .5(Nup - Ndown)
    # max(Ndown) = 2*L
    # min(Sz) = .5((N-L) - L) = .5*(N - 4*L)
    min_Nw = Ne + (N-4*l)//2
    Nws = np.arange(min_Nw, Ne+1, 1)
print('Values of N_w')
print(Nws)
k = np.arange(1, 2*l+1, 2)*0.5*np.pi/l
Grs = np.arange(.2, 2.2, .2)
Gps = Grs/(2*Grs-1)
Grs = np.concatenate((Grs, Gps))
Grs = np.sort(Grs)
Grs = Grs[Grs != 1]
Grs = Grs[Grs > 0]
print(Grs)

Gs = Grs/np.sum(k)
print('Values of G/Gc')
print(Gs*np.sum(k))

dg = 0.01/L # step size of g.
g0 = dg/L # initial value of g
imk = dg # scale of the imaginary parts added to k
imv = g0/L # scale of the imaginary parts used in the initial guess

skip=N # it's harder for larger N, so let's make it easy on us

ki = (-1)**np.arange(l)*imk
kc = np.concatenate((k, ki))
outputs = []
filen = 'pols/pol_results_l{}Ne{}.p'.format(l, Ne)
for Nw in Nws:
    print('')
    print('Running with N_w = {}'.format(Nw))
    print('')
    dims = (l, Ne, Nw)
    sol = bootstrap_g0(dims, g0, kc, imscale_v=imv)
    outputs += [solve_Gs_list(dims, sol, Gs, k, dg=dg, g0=g0, imscale_k=imk, imscale_v=imv)]
    # rewriting file each time in case this terminates early for some reason
    pickle.dump(outputs, open(filen, 'wb'))
print('Done forever!')
