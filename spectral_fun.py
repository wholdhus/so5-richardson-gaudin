from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator
# from quspin.tools.misc import matvec
from exact_qs_so5 import hamiltonian_dict, form_basis, find_min_ev, ham_op, find_nk
import numpy as np
from scipy.sparse.linalg import eigsh

from tqdm import tqdm


def akboth(omega, celts, delts, e0, ep, em, epsilon=10**-10):
    # gps = -2j*celts*epsilon/((omega - ep)**2 + epsilon**2)
    # gms = 2j*delts*epsilon/((omega + em)**2 + epsilon**2)
    gps = -2j*celts*epsilon/((omega + e0 - ep)**2 + epsilon**2)
    gms = 2j*delts*epsilon/((omega - e0 + em)**2 + epsilon**2)

    gp2 = celts/(omega - ep +  e0 + 1j*epsilon)
    gm2 = delts/(omega + em - e0 - 1j*epsilon)
    # gp2 = celts/(omega - ep +  1j*epsilon)
    # gm2 = delts/(omega + em - 1j*epsilon)
    return (np.sum(gps) - np.sum(gms)).imag/(-2*np.pi), (np.sum(gm2) - np.sum(gp2)).imag/np.pi


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

    cp_lo = cp.aslinearoperator()
    cm_lo = cm.aslinearoperator()
    lc = len(vp[0, :])
    ld = len(vm[0, :])
    # lc = len(vp[:,0])
    # ld = len(vm[:,0])
    celts = np.zeros(lc, dtype=np.complex128)
    delts = np.zeros(ld, dtype=np.complex128)
    print('Finding creation matrix elts.')
    cpv = cp_lo.dot(v0)
    cmv = cm_lo.dot(v0)
    for i in tqdm(range(lc)):
        v = bp.get_vec(vp[:, i], sparse=False)
        # cpv = matvec(cp_lo, v0)
        # cpv = cp_lo.dot(v0)
        celts[i] = np.vdot(v, cpv)
        # celts[i] = cp.matrix_ele(v, v0)
    print('Finding annihilation matrix elts.')
    for j in tqdm(range(ld)):
        v = bm.get_vec(vm[:, j], sparse=False)
        # cmv = matvec(cm_lo, v0)
        # cmv = cm_lo.dot(v0)
        delts[j] = np.vdot(v, cmv)
        # delts[j] = cm.matrix_ele(v, v0)
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
    omegas = np.linspace(-1*np.max(np.abs(e)), np.max(np.abs(e)), steps)
    epsilon = np.mean(np.diff(omegas))
    for i, o in enumerate(omegas):
        aks1[i], aks2[i] = akboth(o, celts, delts, e0, ep, em, epsilon=epsilon)
    #     print(o)
    #     print(aks1[i])
    #     print(aks2[i])
    ns = find_nk(L, v[:,0], basis)
    return aks1, aks2, omegas, ns


def check_nonint_spectral_fun(L, N, disp, steps=1000):
    ks = np.arange(L, 2*L)
    plt.figure(figsize=(12, 8))
    for k in ks:
        print('')
        a1, a2, om, ns = find_spectral_fun(L, N, 0, disp, k=k, steps=steps)
        peak_i = np.argmax(a1)
        peak_om = om[peak_i]
        print('Omega at peak value:')
        print(peak_om)
        print('This should be:')
        print(disp[k-L])
        print('omega - epsilon_k:')
        print(peak_om - disp[k-L])
        print('Integrals')
        print(np.trapz(a1, om))
        print(np.trapz(a2, om))
        plt.scatter(om, a2, label = '{}th level'.format(k))
    plt.xlim(0, 1.5*max(disp))
    plt.legend()
    plt.xlabel('Omega')
    plt.ylabel('A(k,omega)')
    plt.title('L = {}, N = {}'.format(L, 2*N))

    for d in disp:
        plt.axvline(d)

    plt.show()


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    L = 4
    N = 2
    ks = np.arange(1, L+1)*np.pi/L
    check_nonint_spectral_fun(L, N, ks, steps=10**4)
