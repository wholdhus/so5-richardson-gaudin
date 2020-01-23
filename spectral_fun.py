from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator
from exact_qs_so5 import ham_op, form_basis, find_min_ev
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

def akw(omega, m_elts, e1s, e2s, eps=10**-8, good=True):
    l1 = len(e1s)
    l2 = len(e2s)
    Gs = np.zeros((l1, l2), dtype=np.complex128)
    for i, e1 in enumerate(e1s):
        Gs[i, :] = np.abs(m_elts[i, :])/(omega+e1-e2s+1j*eps)
    G = np.sum(Gs)
    return -1*G.imag/np.pi

"""
1-particle sf but using only the N particle basis (maybe?)
This isn't correct. Woops
"""
def matrix_elts_1(k, vs, basis):
    kl = [[1.0, k, k]]
    ol = [['+-|', kl]]
    od = {'static': ol}
    op = quantum_operator(od, basis=basis, check_symm=False,
                          check_herm=False)
    l = len(vs[0,:])
    elts = np.zeros((l, l), dtype=np.complex128)
    for i in range(l):
        for j in range(l):
            elts[i, j] = op.matrix_ele(vs[:,i], vs[:,j])
    return elts






if __name__ == '__main__':

    import matplotlib.pyplot as plt

    do_other = True

    L = 4
    Nup = 2
    Ndown = 2
    G = 0.5

    basis1 = form_basis(2*L, Nup, Ndown)
    basis2 = form_basis(2*L, Nup-1, Ndown)
    fullbasis = spinful_fermion_basis_1d(2*L)

    k = np.arange(L) + 1.0
    ho1 = ham_op(L, G, k, basis1)
    hnope = ham_op(L, 0, k, basis1)
    ho2 = ham_op(L, G, k, basis2)
    e1s, v1 = find_min_ev(ho1, L, basis1, n=100)
    kf = L + Nup//2  # index of fermion near surface


    m_elts1 = matrix_elts_1(kf, v1, basis1)
    sdens = np.zeros(300)
    omegas = np.linspace(e1s[0] - 10, e1s[0] + 10, len(sdens))
    for i, o in enumerate(omegas):
        sdens[i] = akw(o, m_elts1, e1s, e1s, eps=10**-6)

    plt.plot(omegas, sdens)
    plt.show()

    print('Sum of sdens:')
    domega = 20./len(sdens)
    print(np.sum(sdens*domega))

    e0, v0 = find_min_ev(hnope, L, basis1, n=1)
    print('Overlap with G=0 ground state')
    print(np.vdot(v0, v1[:, 0]))

    if do_other:
        print('Now trying the slower way!')
        e2s, v2 = find_min_ev(ho2, L, basis2, n=100)
        m_elts = matrix_elts(kf, v1, v2, basis1, basis2, fullbasis)
        sdens = np.zeros(300)
        omegas = np.linspace(-10, 10, 300)
        for i, o in enumerate(omegas):
            sdens[i] = akw(o, m_elts, e1s, e2s, eps=10**-6)
        plt.plot(omegas, sdens)
        plt.show()
        de = omegas[1]-omegas[0]
        print(np.sum(sdens)*de)
