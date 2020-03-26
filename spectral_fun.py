from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator
from exact_qs_so5 import hamiltonian_dict, form_basis, find_min_ev, ham_op, find_nk
import numpy as np
from scipy.sparse.linalg import eigsh

from tqdm import tqdm

def akboth(omega, celts, delts, e0, ep, em, epsilon=10**-10):
    gps = celts*epsilon/((omega + e0 - ep)**2 + epsilon**2)
    gms = delts*epsilon/((omega - e0 + em)**2 + epsilon**2)

    gp2 = celts/(omega - ep + e0 + 1j*epsilon)
    gm2 = delts/(omega + em - e0 - 1j*epsilon)
    # print(np.max(np.abs(gp2)))
    # print(np.max(np.abs(gm2)))

    return (np.sum(gps) - np.sum(gms))/(np.pi), 1j*(np.sum(gp2) + np.sum(gm2))/np.pi

def matrix_elts(k, v0, vp, vm, bp, bm, bf):
    """
    Gets matrix elements between creation and annihilation
    at kth site with initial vector v0. v0 should already
    be transformed to the full basis.
    bp, bm are basis for N+1, N-1.
    bf is the basis for all N
    """
    print('Creating at {}th spot'.format(k))
    kl = [[1.0, k]]
    cl = [['+|', kl]]
    dl = [['-|', kl]]
    cd = {'static': cl}
    dd = {'static': dl}
    cp = quantum_operator(cd, basis=bf, check_symm=False,
                          check_herm=False)
    cm = quantum_operator(dd, basis=bf, check_symm=False,
                          check_herm=False)
    lc = len(vp[0, :])
    ld = len(vm[0, :])
    # lc = len(vp[:,0])
    # ld = len(vm[:,0])
    celts = np.zeros(lc, dtype=np.complex128)
    delts = np.zeros(ld, dtype=np.complex128)
    print('Finding creation matrix elts.')
    for i in tqdm(range(lc)):
        v = bp.get_vec(vp[:, i], sparse=False)
        celts[i] = cp.matrix_ele(v, v0)
    print('Finding annihilation matrix elts.')
    for j in tqdm(range(ld)):
        v = bm.get_vec(vm[:, j], sparse=False)
        delts[j] = cm.matrix_ele(v, v0)
    return np.abs(celts)**2, np.abs(delts)**2

def find_spectral_fun(L, N, G, ks, k=None, n_states=-999, steps=None):
    Nup = N//2
    Ndown = N//2
    if k is None:
        k = L + Nup//2
    basis = form_basis(2*L, Nup, Ndown)
    basism = form_basis(2*L, Nup-1, Ndown)
    basisp = form_basis(2*L, Nup+1, Ndown)
    basisf = spinful_fermion_basis_1d(2*L)
    h = ham_op(L, G, ks, basis, rescale_g=True)
    hp = ham_op(L, G, ks, basisp, rescale_g=True)
    hm = ham_op(L, G, ks, basism, rescale_g=True)
    if n_states == -999:
        e, v = h.eigh()
        ep, vp = hp.eigh()
        em, vm = hm.eigh()
    else:
        # e, v = find_min_ev(h, L, basis, n_states)
        # ep, vp = find_min_ev(hp, L, basisp, n_states)
        # em, vm = find_min_ev(hm, L, basism, n_states)
        e, v = eigsh(h.aslinearoperator(), k=n_states, which='SA')
        ep, vp = eigsh(hp.aslinearoperator(), k=n_states, which='SA')
        em, vm = eigsh(hm.aslinearoperator(), k=n_states, which='SA')
    if steps is None:
        steps = 10*len(e)

    e0 = e[0]
    v0 = basis.get_vec(v[:,0], sparse=False)

    celts, delts = matrix_elts(k, v0, vp, vm, basisp, basism, basisf)
    print('Largest matrix elements: Creation')
    # print(np.round(celts, 5))
    print(np.max(celts))
    print(np.argmax(celts))
    print('Annihilation')
    print(np.max(delts))
    print(np.argmax(delts))

    aks1 = np.zeros(steps)
    aks2 = np.zeros(steps)
    lim = max(np.abs(G), 10)
    omegas = 2*np.linspace(-1*lim*np.sum(ks[:N]), lim*np.sum(ks[:N]), steps)
    # omegas = np.linspace(np.min(e), np.max(e), steps)
    epsilon = np.abs(e[1]-e[0])
    for i, o in enumerate(omegas):
        aks1[i], aks2[i] = akboth(o, celts, delts, e0, ep, em, epsilon=epsilon)
    #     print(o)
    #     print(aks1[i])
    #     print(aks2[i])
    ns = find_nk(L, v[:,0], basis)
    return aks1, aks2, omegas, ns

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    L = int(input('Length: '))
    Nup = int(input('Spin up fermions: '))
    Ndown = int(input('Spin down fermions: '))
    G = float(input('G: '))
    ind = int(input('Which index? '))
    steps = int(input('Integration steps: '))
    n_states = int(input('How many states to find (-999 for complete): '))


    N = Nup + Ndown

    ks = np.arange(1, L+1)*np.pi/L

    aks1, aks2, os, ns = find_spectral_fun(L, N, G, ks, k=ind, n_states=n_states, steps=steps)

    ak01, ak02, o0s, n0s = find_spectral_fun(L, N, 0, ks, k=ind, n_states=n_states, steps=steps)

    aks1 = np.abs(aks1)
    aks2 = np.abs(aks2)
    print(np.sum(ns))
    print(np.sum(n0s))
    print('Integrals!')

    print(np.trapz(aks1, os))
    print(np.trapz(aks2, os))
    print(np.trapz(ak01, o0s))
    print(np.trapz(ak02, o0s))

    plt.figure(figsize=(12, 8))

    plt.subplot(2,1,1)
    plt.scatter(os, aks1, label='G={}, method 1'.format(G), s=10, marker='x')
    plt.scatter(os, aks2, label='G={}, method 2'.format(G), s=10, marker='+')

    plt.scatter(o0s, ak01, label='G=0, method 1', s=10, marker='x')
    plt.scatter(o0s, ak02, label='G=0, method 2', s=10, marker='+')
    plt.legend()

    plt.subplot(2,1,2)
    plt.scatter(np.concatenate((-1*ks[::-1], ks)), ns, label='G={}'.format(G))
    plt.scatter(np.concatenate((-1*ks[::-1], ks)), n0s, label='G=0')
    plt.legend()
    plt.show()
