from spectral_fun import spinful_fermion_basis_1d, lanczos, lanczos_coeffs
from spectral_fun import akboth, quantum_operator, quantum_LinearOperator
from spectral_fun import matrix_elts
import numpy as np

"""
4x4 lattice:
0  1  2  3
4  5  6  7
8  9  10 11
12 13 14 15
"""
lattice = np.arange(16).reshape((4,4))

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

def hubbard_akw_1d(L, N, t, u, k, order=None, lanczos_steps=None):
    if order is None:
        order = 100
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
    print('')
    print('Chemical potential:')
    print(mu)
    print('')

    print('Putting ground state in full basis')
    v0 = basis.get_vec(v0[:, 0], sparse=False)
    cp_lst = []
    cm_lst = []
    for x in range(L):
        cp_lst += [[np.exp(1j*k*(x+1))/np.sqrt(L), x]]
        cm_lst += [[np.exp(-1j*k*(x+1))/np.sqrt(L), x]]
    if N//2 < L-1:
        print('Creating creation operator')
        cp_lo = quantum_LinearOperator([['+|', cp_lst]], basis=basisf,
                                check_herm=False)
    if N//2 > 1:
        print('Creating annihilation operator')
        cm_lo = quantum_LinearOperator([['-|', cm_lst]], basis=basisf,
                                        check_herm=False)
    if lanczos_steps is None:
        e_plus, vp = hp.eigh()
        e_minus, vm = hm.eigh()

        # lanczos_steps = int(3*order)
        coeffs_plus, coeffs_minus = matrix_elts(k, v0, vp, vm, basisp, basism, basisf,
                                                operators=(cp_lo, cm_lo))
    else:
        print('')
        print('Performing Lanczos for c^+')
        if N//2 < L-1:

            coeffs_plus, e_plus = lanczos_coeffs(v0, hp, cp_lo, basisf, basisp,
                                            lanczos_steps, k=order)
            coeffs_plus = coeffs_plus[:order]
            e_plus = e_plus[:order]
        else:
            print('Actually not, since c^+ gives full band')
            coeffs_plus, e_plus = np.zeros(order), np.zeros(order)
        print('')
        print('Performing Lanczos for c^-')
        if N//2 > 1:
            coeffs_minus, e_minus = lanczos_coeffs(v0, hm, cm_lo, basisf, basism,
                                               lanczos_steps, k=order)
            coeffs_minus, e_minus = coeffs_minus[:order], e_minus[:order]
        else:
            print('Actually not, since c^- gives vacuum')
            coeffs_minus, e_minus = np.zeros(order), np.zeros(order)
    steps = 100
    aks = np.zeros(steps)

    relevant_es = []
    for i, c in enumerate(coeffs_plus):
        if c > 10**-6:
            relevant_es += [e_plus[i]]
    for i, c in enumerate(coeffs_minus):
        if c > 10**-6:
            relevant_es += [e_minus[i]]
    lowmega = min((min(e0 - relevant_es), min(relevant_es - e0)))
    highmega = max((max(e0 - relevant_es), max(relevant_es - e0)))
    omegas = np.linspace(1.2*lowmega, 1.2*highmega, steps)

    epsilon = np.mean(np.diff(omegas))
    # epsilon = 0.01
    for i, o in enumerate(omegas):
        aks[i], _ = akboth(o, coeffs_plus/2, coeffs_minus/2,
                              e0 - N*mu, e_plus - (N+1)*mu, e_minus - (N-1)*mu, epsilon=epsilon)
    return aks, omegas


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.special import binom
    L = int(input('L: '))
    N = int(input('N: '))
    t = float(input('t: '))
    u = float(input('u: '))

    ks = np.arange(L//2-1, L//2+1)*2*np.pi/L
    ks_pos = np.arange(0, L//2+1)*2*np.pi/L
    print('Fermi momentum is around:')
    print(ks_pos[N//4])

    k = float(input('k: '))

    # k = 0
    # dim_h = binom(L**2, N//2)**2
    dim_h = binom(L, N//2)**2
    print(dim_h)
    if dim_h < 4000:
        a1, o = hubbard_akw_1d(L, N, t, u, k, order=None, lanczos_steps=None)
        print('Integrals of exact:')
        print(np.trapz(a1, o))
        plt.plot(o, a1, label="exact", color='m')
    orders = [10, 20, 40]
    for i, order in enumerate(orders):
        print('')
        try:
            print('Running with {} eigenvalues'.format(order))
            a1, o = hubbard_akw_1d(L, N, t, u, k, order, lanczos_steps=3*order)
            print('Integrals at {}th order:'.format(order))
            print(np.trapz(a1, o))
            plt.scatter(o, a1, label="order: {}".format(order))
        except Exception as e:
            print('Failed at {}th order'.format(order))
            print(e)
            # raise

    orders = [10, 20, 40]
    u *= 1.5
    for i, order in enumerate(orders):
        print('')
        try:
            print('Running with {} eigenvalues'.format(order))
            a1, o = hubbard_akw_1d(L, N, t, u, k, order, lanczos_steps=3*order)
            print('Integrals at {}th order:'.format(order))
            print(np.trapz(a1, o))
            plt.scatter(o, a1, label="order: {}".format(order))
        except Exception as e:
            print('Failed at {}th order'.format(order))
            print(e)
    plt.legend()
    # plt.savefig('hubbard_spectral_fun_{}_{}_{}.png'.format(L, N, int(u/t)))
    plt.show()
