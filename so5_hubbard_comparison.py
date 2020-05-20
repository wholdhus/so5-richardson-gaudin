from spectral_fun import spinful_fermion_basis_1d, lanczos, lanczos_coeffs
from spectral_fun import akboth, quantum_operator, quantum_LinearOperator
from spectral_fun import matrix_elts, find_nk, find_spectral_fun

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

try:
    with open('context.json') as f:
        context = json.load(f)
    RESULT_FP = context['results_filepath']
except Exception as e:
    print('REQUIRED! context.json file with results_filepath entry')
    raise

def hubbard_dict_1d(L, t, u):
    hop_lst = [[-t, i, (i+1)%L] for i in range(L)]
    hop_lst += [[-t, (i+1)%L, i] for i in range(L)]
    hops = [['+-|', hop_lst],
            ['|+-', hop_lst]]

    dd_lst = [[u, i, i] for i in range(L)]
    dds = [['n|n', dd_lst]]
    hub_lst = hops + dds
    hub_dict = {'static': hub_lst}
    return hub_dict

def hubbard_akw_1d(L, N, t, u, k, order=None, lanczos_steps=None, steps=1000,
                   eta=None):
    hd = hubbard_dict_1d(L, t, u)
    basis = spinful_fermion_basis_1d(L, Nf=(N//2, N//2))
    basisp = spinful_fermion_basis_1d(L, Nf=(N//2+1, N//2))
    basism = spinful_fermion_basis_1d(L, Nf=(N//2-1, N//2))
    basisf = spinful_fermion_basis_1d(L)

    h = quantum_operator(hd, basis=basis)
    hp = quantum_operator(hd, basis=basisp)
    hm = quantum_operator(hd, basis=basism)
    print('Hamiltonians formed!')
    print('Finding lowest-energy eigenpair')
    e0, v0 = h.eigsh(k=1, which='SA')
    print('Finding highest-energy eigenvalue')
    ef, _ = h.eigsh(k=1, which='LA')
    e0 = e0[0]
    ef = ef[0]

    # Determining mu/fermi energy
    ep, _ = hp.eigsh(k=1, which='SA')
    em, _ = hm.eigsh(k=1, which='SA')
    mu = (ep[0]-em[0])/2
    # mu = 0
    print('')
    print('Chemical potential:')
    print(mu)
    print('')

    print('Putting ground state in full basis')
    v0_full = basis.get_vec(v0[:, 0], sparse=False)
    cp_lst = []
    cm_lst = []
    for x in range(L):
        cp_lst += [[np.exp(1j*k*(x+1))/np.sqrt(L), x]]
        cm_lst += [[np.exp(-1j*k*(x+1))/np.sqrt(L), x]]
        # cp_lst += [[np.exp(1j*k*(x+1)), x]]
        # cm_lst += [[np.exp(-1j*k*(x+1)), x]]

    print('Creating creation operator')
    cp_lo = quantum_LinearOperator([['+|', cp_lst]], basis=basisf,
                                    check_herm=False)
    print('Creating annihilation operator')
    cm_lo = quantum_LinearOperator([['-|', cm_lst]], basis=basisf,
                                        check_herm=False)
    if lanczos_steps is None and order is None:
        print('&&&##&#&#&#&#&#')
        print('full diagonalization!!')
        e_plus, vp = hp.eigh()
        e_minus, vm = hm.eigh()

        coeffs_plus, coeffs_minus = matrix_elts(k, v0_full, vp, vm, basisp, basism, basisf,
                                                operators=(cp_lo, cm_lo))
    elif lanczos_steps is None:
        e_plus, vp = hp.eigsh(k=order, which='SA')
        e_minus, vm = hm.eigsh(k=order, which='SA')
        print('Eigenthings found!')
        coeffs_plus, coeffs_minus = matrix_elts(k, v0_full, vp, vm, basisp, basism, basisf,
                                                operators=(cp_lo, cm_lo))
    else:
        if order is None:
            order = 100
        print('')
        print('Performing Lanczos for c^+')
        if N//2 < L-1:

            coeffs_plus, e_plus = lanczos_coeffs(v0_full, hp, cp_lo, basisf, basisp,
                                            lanczos_steps, k=order)
            coeffs_plus = coeffs_plus[:order]
            e_plus = e_plus[:order]
        else:
            print('Actually not, since c^+ gives full band')
            coeffs_plus, e_plus = np.zeros(order), np.zeros(order)
        print('')
        print('Performing Lanczos for c^-')
        if N//2 > 1:
            coeffs_minus, e_minus = lanczos_coeffs(v0_full, hm, cm_lo, basisf, basism,
                                               lanczos_steps, k=order)
            coeffs_minus, e_minus = coeffs_minus[:order], e_minus[:order]
        else:
            print('Actually not, since c^- gives vacuum')
            coeffs_minus, e_minus = np.zeros(order), np.zeros(order)
    # steps = 1000

    relevant_es = []
    all_es = np.concatenate((e_plus, e_minus))

    lowmega = min((e0 - max(all_es),
                   min(all_es) - e0))
    highmega = max((max(all_es) - e0,
                    e0 - min(all_es)))

    omegas = np.linspace(lowmega, highmega, steps)
    # omegas = np.linspace(-100, 100, steps)
    if eta is None:
        eta = np.mean(np.diff(omegas))
    # epsilon = 0.1
    ak_plus = np.zeros(steps)
    ak_minus = np.zeros(steps)
    for i, o in enumerate(omegas):
        ap, am = akboth(o, coeffs_plus, coeffs_minus,
                        e0 - N*mu, e_plus - (N+1)*mu, e_minus - (N-1)*mu, epsilon=eta)
        ak_plus[i] = ap
        ak_minus[i] = am
    ns = find_nk(L//2, v0[:,0], basis)
    return ak_plus, ak_minus, omegas, ns


L = int(input('Length: '))
N = int(input('N: '))
u = float(input('Coupling strength: '))
eta = float(input('Infinitesimal: '))
steps = int(input('steps: '))
do_hub = int(input('Type 0 to also run Hubbard model: '))
# k_i = int(input('Which k should I plot? '))
t = 1.
ks = np.array([(2*i+1)*np.pi/(L) for i in range(L//2)]) # actually only the positive ones
kps = np.arange(1, L//2+1)*2*np.pi/L
G = 2*u/L
spectral_functions = pd.DataFrame({})
plt.figure(figsize=(12,8))

acolors = ['red','orange','magenta','pink']
hcolors = ['blue','green','cyan','olive']

max_w = 0
min_w = 0

for i, kp in enumerate(kps):
    ki = i + L//2
    if do_hub == 1:
        hap, ham, ho, h_nk = hubbard_akw_1d(L, N, t, u, kp, order=None,
                                          lanczos_steps=None,
                                          steps=steps,
                                          eta=eta)
        spectral_functions['hubbard_k{}_akw'.format(i)] = hap + ham
        spectral_functions['hubbard_k{}_o'.format(i)] = ho
        plt.plot(ho, hap+ham, label = 'Hubbard, k = {}'.format(np.round(kp, 2)), color=hcolors[i])

    sap, sam, so, s_nk = find_spectral_fun(L//2, N, G, ks, steps=steps,
                                                       k=ki,
                                                       eta=eta)

    spectral_functions['so5_k{}_akw'.format(i)] = sap + sam
    spectral_functions['so5_k{}_o'.format(i)] = so

    plt.plot(so, sap+sam, label = 'SO(5), k = {}'.format(np.round(ks[i], 2)), color=acolors[i])

    sos = so[(sap + sam) > 5*10**-3]
    try:
        max_w = max([max_w, max(sos)])
        min_w = min([min_w, min(sos)])
    except:
        pass
plt.legend()

plt.ylabel('A(k,omega)')
plt.xlabel('omega/t')
plt.title('L = {}, N = {}, u = {}, eta = {}t'.format(L, N, u, eta))
print('Min, max omegas:')
print(min_w)
print(max_w)
if min_w != 0 and max_w != 0:
    plt.xlim(min_w, max_w)
    # w = min((-1*min_w, max_w))
    # plt.xlim(-1*w, w)
plt.savefig(RESULT_FP + '/figs/sf_comparison_L{}N{}U{}.png'.format(L, N, np.round(u/t, 2)))
spectral_functions.to_csv(RESULT_FP + 'sf_comparison_L{}N{}U{}.csv'.format(L, N, (np.round(u/t, 2))))
# if input('Type yes for n_k') == 'yes':
#     plt.figure()
#     plt.plot(np.concatenate((-1*ks[::-1], ks)), s_nk, label='SO(5)')
#     # plt.plot(np.concatenate((-1*kps[::-1], kps)), h_nk, label ='Hubbard')
#     plt.xlabel('k')
#     plt.ylabel('<n_k>')
#     plt.title('L = {}, N = {}, u = {}'.format(L, N, u))
#     plt.legend()
#     plt.show()
