from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator
from exact_qs_so5 import hamiltonian_dict, form_basis, find_min_ev
import numpy as np


def a(omega, e, v, eps=10**-8):
    l = len(e)
    v0 = v[:, 0]
    ips = np.array([np.vdot(v0, v[:, i]) for i in range(l)])
    Gf = -1*np.sum(ips/(omega - e + 1j*eps))
    a = Gf.imag/np.pi
    return a

"""
Single-particle spectral function
Methods that require working in the N particle basis and the N-1 particle basis
"""
def matrix_elts(k, v1, v2, basis1, basis2, fullbasis):
    print('Creating at {}th spot'.format(k))
    print('Finding matrix elements')
    kl = [[1.0, k]]
    ol = [['+|', kl]]
    od = {'static': ol}
    op = quantum_operator(od, basis=fullbasis, check_symm=False,
                          check_herm=False)
    print('Operator created!')
    l1 = len(v1[0,:])
    l2 = len(v2[0,:])
    elts = np.zeros((l1, l2), dtype=np.complex128)
    for i in range(l1):
        for j in range(l2):
            # print('Doing {} {}'.format(i, j))
            vec1 = v1[:, i]
            vec2 = v2[:, j]
            vf1 = basis1.get_vec(vec1, sparse=False)
            vf2 = basis2.get_vec(vec2, sparse=False)
            elts[i, j] = op.matrix_ele(vf1, vf2)
    return np.abs(elts)**2


def akw(omega, m_elts, e1s, e2s, eps=10**-10):
    l1 = len(e1s)
    l2 = len(e2s)
    e0 = np.min(e2s)
    Gs = np.zeros((l1, l2), dtype=np.complex128)
    gs = np.zeros(l1, dtype=np.complex128)
    for i, e1 in enumerate(e1s):
        Gs[i, :] = m_elts[i, :]/(omega+e1-e2s+1j*eps)
    gs = 2j*np.pi*m_elts[:, 0]*eps/((omega+e0-e1s)**2+eps**2)
    G = np.sum(Gs)
    g = np.sum(gs)
    return -1*G.imag/np.pi, -1j*g


def matrix_elts_both(k, v0, vp, vm, bp, bm, bf):
    """
    Gets matrix elements between creation and annihilation
    at kth site with initial vector v0. v0 should already
    be transformed to the full basis.
    bp, bm are basis for N+1, N-1.
    bf is the basis for all N
    """
    print('Creating at {}th spot'.format(k))
    print('Finding matrix elements')
    kl = [[1.0, k]]
    cl = [['+|', kl]]
    dl = [['-|', kl]]
    cd = {'static': cl}
    dd = {'static': dl}
    cp = quantum_operator(cd, basis=bf, check_symm=False,
                          check_herm=False)
    cm = quantum_operator(dd, basis=bf, check_symm=False,
                          check_herm=False)
    lc = len(vp[:,0])
    ld = len(vm[:,0])
    celts = np.zeros(lc, dtype=np.complex128)
    delts = np.zeros(ld, dtype=np.complex128)
    for i in range(lc):
        v = bp.get_vec(vp[i, :], sparse=False)
        celts[i] = cp.matrix_ele(v, v0)
    for j in range(ld):
        v = bm.get_vec(vm[j, :], sparse=False)
        delts[j] = cm.matrix_ele(v, v0)
    return np.abs(celts)**2, np.abs(delts)**2


def akboth(omega, celts, delts, e0, ep, em, epsilon=10**-10):
    gps = -2j*celts*epsilon/((omega + e0 - ep)**2 + epsilon**2)
    gms = 2j*delts*epsilon/((omega - e0 + em)**2 + epsilon**2)
    return 1j*(np.sum(gps) - np.sum(gms))


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    steps = int(input('Integration steps: '))
    L = int(input('Length: '))
    Nup = int(input('Spin up fermions: '))
    Ndown = int(input('Spin down fermions: '))
    G = float(input('G: '))
    ind = int(input('Which index? '))
    kf = L + ind
    if ind == -1:
        kf = L + Nup//2  # index of fermion near surface


    basis = form_basis(2*L, Nup, Ndown)
    basism = form_basis(2*L, Nup-1, Ndown)
    basisp = form_basis(2*L, Nup+1, Ndown)
    basisf = spinful_fermion_basis_1d(2*L)


    k = np.arange(L) + 1.0
    if G == -999:
        hd = hamiltonian_dict(L, 1.0, k, no_kin=True)
    else:
        hd = hamiltonian_dict(L, G, k, no_kin=False)
    h = quantum_operator(hd, basis=basis)
    hp = quantum_operator(hd, basis=basisp)
    hm = quantum_operator(hd, basis=basism)
    e, v = h.eigh()
    e0 = e[0]
    ep, vp = hp.eigh()
    em, vm = hm.eigh()
    v0 = basis.get_vec(v[:,0], sparse=False)
    celts, delts = matrix_elts_both(kf, v0, vp, vm,
                                    basisp, basism, basisf)
    aks = np.zeros(steps, dtype=np.complex128)
    if np.min(e) > 10**-10:
        omegas = np.linspace(0.5*np.min(e), 2*np.max(e), steps)
    elif np.min(e) < -10**-10:
        omegas = np.linspace(2*np.min(e), 2*np.max(e), steps)
    else:
        omegas = np.linspace(-2*np.max(e), 2*np.max(e), steps)

    epsilon = omegas[2]-omegas[1] # more or less dx


    for i, o in enumerate(omegas):
        aks[i] = akboth(o, celts, delts, e0, ep, em, epsilon=epsilon)
    aks = aks.real/(2*np.pi)
    plt.figure()
    plt.plot(omegas, aks)
    plt.show()
    print(np.trapz(aks, omegas))
    print(e)

    if input('Type 1 to continue: ') == '1':
        plt.figure()
        e1s, v1 = find_min_ev(h, L, basis, n=100)
        e2s, v2 = find_min_ev(hm, L, basism, n=100)
        m_elts = matrix_elts(kf, v1, v2, basis, basism, basisf)
        sdens = np.zeros(steps)
        sdens2 = np.zeros(steps)
        omegas = np.linspace(-10, 10, steps)
        print('Matrix elts')
        print(m_elts[0,:])
        for i, o in enumerate(omegas):
            sdens[i], sdens2[i] = akw(o, m_elts, e1s, e2s, eps=10**-8)
        plt.plot(omegas, sdens, label='Im(G)')
        plt.show()
        plt.plot(omegas, sdens2, label='i * g>')
        plt.show()
        print(np.trapz(sdens, omegas))
        print(np.trapz(sdens2, omegas))
    else:
        print('OK then')
