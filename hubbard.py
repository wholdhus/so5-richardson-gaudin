from exact_qs_so5 import spinful_fermion_basis_1d, quantum_operator
from spectral_fun import lanczos, lanczos_coeffs, akboth
import numpy as np

"""
4x4 lattice:
0  1  2  3
4  5  6  7
8  9  10 11
12 13 14 15
"""
lattice = np.arange(16).reshape((4,4))

def hubbard_dict(L, t, u):
    V = L**2
    hop_lst = [[-t, i, (i+1)%V] for i in range(V)]
    hop_lst += [[-t, i, (i-1)%V] for i in range(V)]
    hop_lst += [[-t, i, (i+L)%V] for i in range(V)]
    hop_lst += [[-t, i, (i-L)%V] for i in range(V)]
    hops = [['+-|', hop_lst],
            ['-+|', hop_lst],
            ['|+-', hop_lst],
            ['|-+', hop_lst]]

    dd_lst = [[u, i, i] for i in range(V)]
    dds = [['n|n', dd_lst]]
    hub_lst = hops + dds
    hub_dict = {'static': hub_lst}
    return hub_dict


def hubbard_akw(L, N, t, u, kx, ky, order, lanczos_steps=None):
    if lanczos_steps is None:
        lanczos_steps = 2*order
    V = L**2
    hd = hubbard_dict(L, t, u)
    basis = spinful_fermion_basis_1d(V, Nf=(N//2, N//2))
    basisp = spinful_fermion_basis_1d(V, Nf=(N//2+1, N//2))
    basism = spinful_fermion_basis_1d(V, Nf=(N//2-1, N//2))
    basisf = spinful_fermion_basis_1d(V)

    h = quantum_operator(hd, basis=basis)
    hp = quantum_operator(hd, basis=basisp)
    hm = quantum_operator(hd, basis=basism)
    print('Hamiltonians formed!')
    print('Finding lowest-energy eigenpair')
    e0, v0 = h.eigsh(k=1, which='SA')
    ef, _ = h.eigsh(k=1, which='LA')
    steps = 10*order
    e0 = e0[0]
    ef = ef[0]
    v0 = basis.get_vec(v0[:, 0], sparse=False)

    c_lst = []
    for x in range(L):
        for y in range(L):
            c_lst += [[np.exp(1j*(kx*x + ky*y)), (x+1)*(y+1) -1]]

    cp = quantum_operator({'static': [['+|', c_lst]]}, basis=basisf,
                          check_herm=False)
    cm = quantum_operator({'static': [['-|', c_lst]]}, basis=basisf,
                         check_herm=False)
    cp_lo = cp.aslinearoperator()
    cm_lo = cm.aslinearoperator()

    print('')
    print('Performing Lanczos for c^+')
    coeffs_plus, e_plus = lanczos_coeffs(v0, hp, cp_lo, basisf, basisp,
                                         lanczos_steps, k=order)
    print('')
    print('Performing Lanczos for c^-')
    # coeffs_minus, e_minus = lanczos_coeffs(v0, hm, cm_lo, basisf, basism,
    #                                        lanczos_steps)
    coeffs_minus, e_minus = np.zeros(order), np.zeros(order)
    aks1 = np.zeros(steps)
    aks2 = np.zeros(steps)
    e_mag = np.max(np.abs((e0, ef)))
    omegas = np.linspace(-10*e_mag, 10*e_mag, steps)
    epsilon = np.mean(np.diff(omegas))
    for i, o in enumerate(omegas):
        aks1[i], aks2[i] = akboth(o, coeffs_plus[:order], coeffs_minus[:order],
                                  e0, e_plus[:order], e_minus[:order], epsilon=epsilon)
    return aks1, aks2, omegas


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from scipy.special import binom
    L = 4
    t = 1
    u = 10
    N = 2
    kx = np.pi
    ky = np.pi
    dim_h = binom(L**2, N//2)*binom(L**2, N//2)
    orders = np.arange(5, dim_h//2, 4, dtype=int)
    for order in orders:
        print('')
        try:
            print('Running with {} eigenvalues'.format(order))
            a1, a2, o = hubbard_akw(L, N, t, u, kx, ky, order)
            print('Integrals at {}th order:'.format(order))
            print(np.trapz(a1, o))
            print(np.trapz(a2, o))
            plt.plot(o, a1, label=order)
        except Exception as e:
            print('Failed at {}th order'.format(order))
            print(e)
            raise
    plt.legend()
    plt.savefig('hubbard_spectral_fun_{}_{}_{}.png'.format(L, N, int(u/t)))
