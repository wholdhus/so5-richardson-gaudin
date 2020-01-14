from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator
import numpy as np
from scipy.optimize import root



def form_basis(L, Nup, Ndown):
    basis = spinful_fermion_basis_1d(L, Nf=(Nup, Ndown))
    print(basis)
    return basis

def casimir(L, k):
    p_k = L + k
    m_k = l - (k+1)


def hamiltonian_dict(L, Nup, Ndown, G, k):
    # k should include positive and negative values
    pvals = G*np.outer(k, k)
    dvals = np.abs(pvals)
    all_k = []
    ppairing = [] # spin 1 pairing
    zpairing = [] # spin 0 pairing
    samesame = [] # n_k n_k' interaction
    dens = []
    spm = [] # spin spin interaction
    for k1 in range(L):
        p_k1 = L + k1 # index of +k, spin up fermion
        m_k1 = L - (k1+1) # index of -k, spin up fermion

        all_k += [[0.5*k[k1], p_k1], [0.5*k[k1], m_k1]]

        for k2 in range(L):
            Zkk = G*k[k2]*k[k1]
            p_k2 = L + k2 # index of +k, spin up fermion
            m_k2 = L - (k2+1) # index of -k, spin down fermion
            ppairing += [
                         [Zkk, p_k1, m_k1, m_k2, p_k2]
                        ]
            zpairing += [
                         [0.5*Zkk, p_k1, p_k2, m_k1, m_k2],
                         [0.5*Zkk, m_k1, m_k2, p_k1, p_k2],
                         [-0.5*Zkk, p_k1, m_k2, m_k1, p_k2],
                         [-0.5*Zkk, m_k1, p_k2, p_k1, m_k2]
                        ]
            spm += [
                    [0.5*Zkk, p_k1, p_k2, p_k1, p_k2],
                    [0.5*Zkk, p_k1, m_k2, p_k1, m_k2],
                    [0.5*Zkk, m_k1, p_k2, m_k1, p_k2],
                    [0.5*Zkk, m_k1, m_k2, m_k1, m_k2]
                    ]
            samesame += [[0.5*Zkk, p_k1, p_k2],
                         [0.5*Zkk, p_k1, m_k2],
                         [0.5*Zkk, m_k1, p_k2],
                         [0.5*Zkk, m_k1, m_k2]
                        ]
            dens += [
                     [-0.5*Zkk, p_k1],
                     [-0.5*Zkk, m_k1],
                     [-0.5*Zkk, p_k2],
                     [-0.5*Zkk, m_k2]
                    ]
    static = [['n|', all_k], ['|n', all_k],
              ['++--|', ppairing], ['--++|', ppairing],
              ['+-|+-', zpairing], ['-+|-+', zpairing],
              ['|++--', ppairing], ['|--++', ppairing],
              ['+-|-+', spm], ['-+|+-', spm],
              ['nn|', samesame], # the up/down density density stuff cancels
              ['|nn', samesame],
              ['n|', dens],
              ['|n', dens]
             ]

    return {'static': static}
    return op_dict


def iom_dict(L, Nup, Ndown, G, k, k1=0):

    ppairing = [] # spin 1 pairing
    zpairing = [] # spin 0 pairing
    samesame = []
    dens = []
    spm = [] # s+/s-  interaction
    p_k1 = L + k1 # index of +k, spin up fermion
    m_k1 = L - (k1+1) # index of -k, spin up fermion
    print('p_k1, m_k1 = {}, {}'.format(p_k1, m_k1))

    all_k = [[0.5, p_k1], [0.5, m_k1]]
    for k2 in range(L):
        if k2 != k1:
            Zkk = G*k[k2]*k[k1]/(k[k2]-k[k1])
            p_k2 = L + k2 # index of +k fermions
            m_k2 = L - (k2+1) # index of -k fermions
            ppairing += [
                         [Zkk, p_k1, m_k1, m_k2, p_k2]
                        ]
            zpairing += [
                         [0.5*Zkk, p_k1, p_k2, m_k1, m_k2],
                         [0.5*Zkk, m_k1, m_k2, p_k1, p_k2],
                         [-0.5*Zkk, p_k1, m_k2, m_k1, p_k2],
                         [-0.5*Zkk, m_k1, p_k2, p_k1, m_k2]
                        ]
            samesame += [[0.5*Zkk, p_k1, p_k2],
                         [0.5*Zkk, p_k1, m_k2],
                         [0.5*Zkk, m_k1, p_k2],
                         [0.5*Zkk, m_k1, m_k2]
                        ]
            spm += [
                    [0.5*Zkk, p_k1, p_k2, p_k1, p_k2],
                    [0.5*Zkk, p_k1, m_k2, p_k1, m_k2],
                    [0.5*Zkk, m_k1, p_k2, m_k1, p_k2],
                    [0.5*Zkk, m_k1, m_k2, m_k1, m_k2]
                    ]
            dens += [
                     [-0.5*Zkk, p_k1],
                     [-0.5*Zkk, m_k1],
                     [-0.5*Zkk, p_k2],
                     [-0.5*Zkk, m_k2]
                    ]
    static = [['n|', all_k], ['|n', all_k],
              ['++--|', ppairing], ['--++|', ppairing],
              ['+-|+-', zpairing], ['-+|-+', zpairing],
              ['|++--', ppairing], ['|--++', ppairing],
              ['+-|-+', spm], ['-+|+-', spm],
              ['nn|', samesame], # the up/down density density stuff cancels
              ['|nn', samesame],
              ['n|', dens],
              ['|n', dens]
             ]
    return {'static': static}


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


def find_min_ev(operator, L):
    e, v = operator.eigsh()
    if e[0] > 0:
        print('Positive eigenvalue: finding a minimum')
        cdict = constant_op_dict(L, np.max(e))
        cop = quantum_operator(cdict, basis=basis)
        e2, v = (cop-operator).eigsh()
        e = -(e2-e[0])
    v0 = v[:, 0]
    return np.min(e), v0


def find_gs_observables(L, Nup, Ndown, g, k, basis, trysparse=True):
    hdict = hamiltonian_dict(L, Nup, Ndown, g, k)

    h = quantum_operator(hdict, basis=basis)
    if trysparse:
        e0, v0 = find_min_ev(h, L)
    else:
        e, v = h.eigh()
        print('energies:')
        print(e)
        e0 = np.min(e)
        v0 = v[:, 0]
    nks = find_nk(L, v0, basis)
    return e0, nks


def Z(x, y):
    return x*y/(x-y)


def pairon_eqs(es, L, Ne, G, ks, rs):
    ces = es[:Ne] + 1j*es[Ne:]
    pairon_eqs = np.zeros(L, dtype=np.complex128)
    for i, k in enumerate(ks):
        pairon_eqs[i] = G*np.sum(Z(k, ces)) - rs[i]
    # np.random.shuffle(pairon_eqs) # Does this help?
    some = pairon_eqs[:Ne]
    out = np.concatenate((some.real, some.imag))
    # out = np.concatenate((pairon_eqs.real, pairon_eqs.imag))
    # print(out)
    return out


def wairon_eqs(ws, ces, L, Nw):
    cws = ws[:Nw] + 1j*ws[Nw:]
    wairon_eqs = np.zeros(Nw, dtype=np.complex128)
    for i, w in enumerate(cws):
        otherws = cws[np.arange(Nw) != i]
        wairon_eqs[i] = Z(otherws, w).sum()
        wairon_eqs[i] += - Z(ces, w).sum()
    out = np.concatenate((wairon_eqs.real, wairon_eqs.imag))
    return out


def find_pairons(L, Nup, Ndown, G, ks, basis, trysparse=True):
    rs = np.zeros(L)
    for i in range(L):
        rd = iom_dict(L, Nup, Ndown, G, ks, k1=i)
        r = quantum_operator(rd, basis=basis)
        rs[i], _ = find_min_ev(r, L)
        # if rs[i] < 0:
        #     print('Woops this one is negativeroony!')
        #     rvs, _ = r.eigsh()
        #     rs[i] = np.min(rvs)
    # guess = np.array([ks[i//2] for i in range(Nup)], dtype=np.complex128)
    guess = np.array(np.random.rand(Nup), dtype=np.complex128)
    rguess = np.concatenate((guess.real, guess.imag))
    # padding = np.zeros(2*L-2*Nup) # so we have as many variables as equations
    # rguess = np.concatenate((rguess, padding))
    sol = root(pairon_eqs, rguess, args=(L, Nup, G, ks, rs))
    es = sol.x[:Nup] + 1j*sol.x[Nup:]
    er = pairon_eqs(sol.x, L, Nup, G, ks, rs)
    print('Pairons found with maximum error:')
    print(np.max(np.abs(er)))

    rguess2 = np.concatenate((ks[:Ndown], np.zeros(Ndown)))
    # rguess2 = np.concatenate((guess.real, guess.imag))
    sol2 = root(wairon_eqs, rguess2, args=(es, L, Ndown))
    ws = sol2.x[:Ndown] + 1j*sol2.x[Ndown:]
    er = wairon_eqs(sol.x, es, L, Ndown)
    print('Wairons found with maximum error:')
    print(np.max(np.abs(er)))

    return es, ws, rs


def make_plots():
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


if __name__ == '__main__':
    L = 4
    ks = np.arange(L) + 1.0
    G = 0
    Nup = 2
    Ndown = 2
    basis = form_basis(2*L, Nup, Ndown)

    ioms = np.zeros(L)
    rd = iom_dict(L, Nup, Ndown, G, ks, k1=L-1)
    r = quantum_operator(rd, basis=basis)
    _, v0 = find_min_ev(r, L)
    for i in range(L):
        iod = iom_dict(L, Nup, Ndown, G, ks, k1=i)
        io = quantum_operator(iod, basis=basis)
        # rs, vs = io.eigh()
        # ioms[i] = np.min(rs)
        ioms[i] = io.matrix_ele(v0, v0)

    e0 = np.sum(ioms*ks)

    hd = hamiltonian_dict(L, Nup, Ndown, G, ks)
    ho = quantum_operator(hd, basis=basis)
    e1, _ = find_min_ev(ho, L)
    print('Energy from sum of ioms:')
    print(e0)

    print('Energy from hamiltonian:')
    print(e1)


    # es, ws, rs = find_pairons(L, Nup, Ndown, G, ks, basis, trysparse=True)
    # print('wooh')
    # print(es)
    # print(ws)
