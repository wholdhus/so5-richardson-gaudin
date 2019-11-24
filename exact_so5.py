import numpy as np
from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator

def form_basis(L, Nup, Ndown):
    basis = spinful_fermion_basis_1d(L, Nf=(Nup, Ndown))
    print(basis)
    return basis


def construct_hamiltonian(L, Nup, Ndown, G, k):
    # k should include positive and negative values
    L = 2*L
    basis = form_basis(L, Nup, Ndown)
    pvals = G*np.outer(k, k)
    dvals = np.abs(pvals)
    kevals = np.diag(k)
    ke = [] # kinetic enerbgy terms
    ppairing = [] # spin 1 pairing
    mpairing = [] # spin -1 pairing
    zpairing = [] # spin 0 pairing
    densdens = [] # n_k n_k' interaction
    spm = [] # spin spin interaction
    szsz = []
    for k1 in range(L//2):
        p_k1 = k1 + L//2 # index of +k, spin up fermion
        m_k1 = L//2 - k1 # index of -k, spin up fermion

        ke += [[kevals[k1], p_k1],
               [kevals[k1], p_k1],
               [kevals[k1], m_k1],
               [kevals[k1], m_k1]
               ]

        for k2 in range(L//2):
            p_k2 = k2 + L//2 # index of +k, spin up fermion
            m_k2 =  L//2 - k2 # index of -k, spin down fermion

            ppairing += [
                         [pvals[k1, k2], p_k1, m_k1, m_k2, p_k2]
                        ]
            mpairing += [
                         [pvals[k1, k2], p_k1, m_k1, m_k2, p_k2]
                         ]
            zpairing += [
                         [pvals[k1, k2]/np.sqrt(2), p_k1, m_k2, m_k1, p_k2],
                         [pvals[k1, k2]/np.sqrt(2), m_k1, p_k2, p_k1, m_k2]
                         ]
            densdens += [[dvals[k1, k2], p_k1, p_k2],
                         [dvals[k1, k2], p_k1, p_k2],
                         [dvals[k1, k2], p_k1, m_k2],
                         [dvals[k1, k2], p_k1, m_k2],
                         [dvals[k1, k2], m_k1, p_k2],
                         [dvals[k1, k2], m_k1, p_k2],
                         [dvals[k1, k2], m_k1, m_k2],
                         [dvals[k1, k2], m_k1, m_k2],
                         [dvals[k1, k2], p_k1, p_k2],
                         [dvals[k1, k2], p_k1, p_k2],
                         [dvals[k1, k2], p_k1, m_k2],
                         [dvals[k1, k2], p_k1, m_k2],
                         [dvals[k1, k2], m_k1, p_k2],
                         [dvals[k1, k2], m_k1, p_k2],
                         [dvals[k1, k2], m_k1, m_k2],
                         [dvals[k1, k2], m_k1, m_k2]
                        ]
            spm += [
                       # splus sminus
                        [dvals[k1, k2], p_k1, p_k1, p_k2, p_k2],
                        [dvals[k1, k2], p_k1, p_k1, m_k1, m_k1],
                        [dvals[k1, k2], m_k1, m_k1, p_k2, p_k2],
                        [dvals[k1, k2], m_k1, m_k1, m_k1, m_k1],
                       # smninus smplus
                        [dvals[k1, k2], p_k1, p_k1, p_k2, p_k2],
                        [dvals[k1, k2], p_k1, p_k1, m_k1, m_k1],
                        [dvals[k1, k2], m_k1, m_k1, p_k2, p_k2],
                        [dvals[k1, k2], m_k1, m_k1, m_k1, m_k1],
                    ]
            szsz += [[dvals[k1, k2], p_k1, p_k2],
                        [-dvals[k1, k2], p_k1, p_k2],
                        [dvals[k1, k2], p_k1, m_k2],
                        [-dvals[k1, k2], p_k1, m_k2],
                        [dvals[k1, k2], m_k1, p_k2],
                        [-dvals[k1, k2], m_k1, p_k2],
                        [dvals[k1, k2], m_k1, m_k2],
                        [-dvals[k1, k2], m_k1, m_k2],
                        [-dvals[k1, k2], p_k1, p_k2],
                        [dvals[k1, k2], p_k1, p_k2],
                        [-dvals[k1, k2], p_k1, m_k2],
                        [dvals[k1, k2], p_k1, m_k2],
                        [-dvals[k1, k2], m_k1, p_k2],
                        [dvals[k1, k2], m_k1, p_k2],
                        [-dvals[k1, k2], m_k1, m_k2],
                        [dvals[k1, k2], m_k1, m_k2]
                        ]
    static = [['++--|', ppairing], ['--++|', ppairing],
              ['+-|+-', zpairing], ['-+|-+', zpairing],
              ['|++--', mpairing], ['|--++', mpairing],
              ['+-|+-', spm], ['n|n', szsz],
              ['n|n', densdens]
             ]
    op_dict = {'static': static}
    return quantum_operator(op_dict, basis=basis)


if __name__ == '__main__':
    L = 4
    k = np.arange(4) + 1.0
    h = construct_hamiltonian(L, 2, 2, 1.0, k)
    e, v = h.eigh()
    print(e)
