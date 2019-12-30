from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator
import numpy as np


def form_basis(L, Nup, Ndown):
    basis = spinful_fermion_basis_1d(L, Nf=(Nup, Ndown))
    print(basis)
    return basis


def construct_hamiltonian_dict(L, Nup, Ndown, G, k):
    # k should include positive and negative values
    pvals = G*np.outer(k, k)
    dvals = np.abs(pvals)
    pos_k = [] # kinetic energy terms
    neg_k = []
    ppairing = [] # spin 1 pairing
    mpairing = [] # spin -1 pairing
    zpairing = [] # spin 0 pairing
    densdens = [] # n_k n_k' interaction
    spm = [] # spin spin interaction
    szsz = []
    for k1 in range(L):
        p_k1 = L + k1 # index of +k, spin up fermion
        m_k1 = L - (k1+1) # index of -k, spin up fermion

        pos_k += [[k[k1], p_k1]]
        neg_k += [[k[k1], m_k1]]

        for k2 in range(L):
            p_k2 = L + k2 # index of +k, spin up fermion
            m_k2 = L - (k2+1) # index of -k, spin down fermion

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
    static = [['n|', pos_k], ['|n', neg_k],
              ['++--|', ppairing], ['--++|', ppairing],
              ['+-|+-', zpairing], ['-+|-+', zpairing],
              ['|++--', mpairing], ['|--++', mpairing],
              ['+-|+-', spm], ['n|n', szsz],
              ['n|n', densdens]
             ]
    return {'static': static}
    return op_dict


def iom_dict(L, Nup, Ndown, G, k, ind=0):
    
    ppairing = [] # spin 1 pairing
    mpairing = [] # spin -1 pairing
    zpairing = [] # spin 0 pairing
    densdens = [] # combined n n, sz sz interactions
    dens = []
    spm = [] # s+/s-  interaction
    k1 = ind
    p_k1 = L + k1 # index of +k, spin up fermion
    m_k1 = L - (k1+1) # index of -k, spin up fermion
    # print('p_k1, m_k1 = {}, {}'.format(p_k1, m_k1))

    all_k = [[0.5, p_k1], [0.5, m_k1]]
    for k2 in np.arange(L)[np.arange(L) != k1]:
        Zkk = G*k[k1]*k[k2]/(k[k1]-k[k2])
        p_k2 = L + k2 # index of +k, spin up fermion
        m_k2 = L - (k2+1) # index of -k, spin down fermion
        # print('p_k2, m_k2  = {}, {}'.format(p_k2, m_k2))

        ppairing += [
                     [Zkk, p_k1, m_k1, m_k2, p_k2]
                    ]
        mpairing += [
                     [Zkk, p_k1, m_k1, m_k2, p_k2]
                     ]
        zpairing += [
                     [Zkk/np.sqrt(2), p_k1, m_k2, m_k1, p_k2],
                     [Zkk/np.sqrt(2), m_k1, p_k2, p_k1, m_k2]
                     ]
        densdens += [[Zkk/2, p_k1, p_k2],
                     [Zkk/2, p_k1, m_k2],
                     [Zkk/2, m_k1, p_k2],
                     [Zkk/2, m_k1, m_k2]
                    ]
        spm += [
                   # splus sminus
                    [Zkk, p_k1, p_k1, p_k2, p_k2],
                    [Zkk, p_k1, p_k1, m_k1, m_k1],
                    [Zkk, m_k1, m_k1, p_k2, p_k2],
                    [Zkk, m_k1, m_k1, m_k1, m_k1],
                   # smninus smplus
                    [Zkk, p_k1, p_k1, p_k2, p_k2],
                    [Zkk, p_k1, p_k1, m_k1, m_k1],
                    [Zkk, m_k1, m_k1, p_k2, p_k2],
                    [Zkk, m_k1, m_k1, m_k1, m_k1],
                ]
        dens += [[-0.5*Zkk, p_k1],
                 [-0.5*Zkk, m_k1],
                 [-0.5*Zkk, p_k2],
                 [-0.5*Zkk, m_k2]
                ]
    static = [['n|', all_k], ['|n', all_k],
              ['++--|', ppairing], ['--++|', ppairing],
              ['+-|+-', zpairing], ['-+|-+', zpairing],
              ['|++--', mpairing], ['|--++', mpairing],
              ['+-|+-', spm], # need to check these out woops
              ['nn|', densdens], # the up/down density density stuff cancels
              ['|nn', densdens],
              ['n|', dens],
              ['|n', dens]
             ]
    return {'static': static}
    return op_dict

def constant_op_dict(L, c):
    co = []

    for k1 in range(L):
        p_k1 = k1 + L # index of +k, spin up fermion
        m_k1 = L - (k1+1) # index of -k, spin up fermion

        co += [[c, p_k1, m_k1]]
        const = [['I|I', co]]
        return {'static': const}

def N_dict(L):
    pos_n = []
    neg_n = []
    for k in range(L):
        p_k = k + L
        m_k = L - (k+1)
        pos_n += [[1.0, p_k]]
        neg_n += [[1.0, p_k]]
    nk = [['n|', pos_n + neg_n],
          ['|n', pos_n + neg_n]]
    return {'static': nk}


def find_nk(L, state, basis):
    nks = np.zeros(2*L)
    for k in range(2*L):
        nval = [[1.0, k]]
        nt = [['n|', nval]]
        dic = {'static': nt}
        Nup = quantum_operator(dic, basis=basis)
        nt = [['|n', nval]]
        dic = {'static': nt}
        Ndown = quantum_operator(dic, basis=basis)
        nks[k] = Nup.matrix_ele(state, state) + Ndown.matrix_ele(state,state)
    return nks


def find_gs_observables(L, Nup, Ndown, g, k, basis, trysparse=True):
    hdict = construct_hamiltonian_dict(L, Nup, Ndown, g, k)

    h = quantum_operator(hdict, basis=basis)
    if trysparse:
        e, v = h.eigsh()
        print('Initial energies:')
        print(e)
        if e[0] > 0:
            cdict = constant_op_dict(L, e[0])
            cop = quantum_operator(cdict, basis=basis)
            e2, v = (cop - h).eigsh()
            e = -(e2-e[0])
    else:
        e, v = h.eigh()
        print('energies:')
        print(e)
    v0 = v[:,0]
    nks = find_nk(L, v0, basis)
    return e[0], nks



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    L = int(input('L: '))
    Nup = int(input('Nup: '))
    Ndown = Nup
    Gmax = int(input('Gmax: '))
    basis = form_basis(2*L, Nup, Ndown)

    k = np.array([(2*i+1)*np.pi/L for i in range(L)])
    all_k = np.concatenate((-1*k[::-1],k))
    gs = Gmax*np.linspace(0, 1, 7)[1:]
    es = np.zeros(6)
    plt.figure(figsize=(8,6))
    for i, g in enumerate(gs):
        print('g = {}'.format(g))
        es[i], nks = find_gs_observables(L, Nup, Ndown, g, k, basis)
        plt.scatter(k, nks[L:], label='G = {}'.format(np.round(g,2)))
    plt.title('L = {}, Nup = Ndown = {}'.format(L, Nup))
    plt.xlabel('k')
    plt.ylabel('n_k')
    plt.legend()

    if gs[-1] > 0:
        plt.savefig('L{}N{}Repulsive.png'.format(L, Nup+Ndown))
    else:
        plt.savefig('L{}N{}Repulsive.png'.format(L, Nup+Ndown))
    plt.show()

    print(all_k)
