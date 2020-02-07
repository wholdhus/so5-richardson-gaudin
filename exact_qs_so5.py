from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator
import numpy as np
from scipy.optimize import root



def form_basis(L, Nup, Ndown):
    basis = spinful_fermion_basis_1d(L, Nf=(Nup, Ndown))
    return basis


def casimir_dict(L, k1):
    p_k1 = L + k1 # index of +k, spin up fermion
    m_k1 = L - (k1+1) # index of -k, spin up fermion
    ppairing = [
                 [1, p_k1, m_k1, m_k1, p_k1]
                ]
    zpairing = [
                 [0.5, p_k1, p_k1, m_k1, m_k1],
                 [0.5, m_k1, m_k1, p_k1, p_k1],
                 [-0.5, p_k1, m_k1, m_k1, p_k1],
                 [-0.5, m_k1, p_k1, p_k1, m_k1]
                ]
    samesame = [[0.5, p_k1, p_k1],
                 [0.5, p_k1, m_k1],
                 [0.5, m_k1, p_k1],
                 [0.5, m_k1, m_k1]
                ]
    spm = [
            [0.5, p_k1, p_k1, p_k1, p_k1],
            [0.5, p_k1, m_k1, p_k1, m_k1],
            [0.5, m_k1, p_k1, m_k1, p_k1],
            [0.5, m_k1, m_k1, m_k1, m_k1]
            ]
    dens = [
             [-0.5, p_k1],
             [-0.5, m_k1],
             [-0.5, p_k1],
             [-0.5, m_k1]
            ]
    static = [
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


def hamiltonian_dict(L, G, k, no_kin=False, trig=False):
    # k should include positive and negative values
    all_k = []
    ppairing = [] # spin 1 pairing
    zpairing = [] # spin 0 pairing
    samesame = [] # n_k n_k' interaction
    dens = []
    spm = [] # spin spin interaction
    smp = []
    for k1 in range(L):
        p_k1 = L + k1 # index of +k fermions
        m_k1 = L - (k1+1) # index of -k fermions

        all_k += [[0.5*k[k1], p_k1], [0.5*k[k1], m_k1]]

        for k2 in range(L):
            Zkk = G*k[k1]*k[k2]
            Xkk = Zkk
            Xskk = Zkk
            Xckk = Zkk
            if trig:
                Zkk = G*np.sin(k[k1]+k[k2])*np.cos(k[k1]-k[k2])
                Xkk = G*np.sin(k[k1]+k[k2])
                Xskk = G*Xkk*np.exp(1j*(k[k1]-k[k2]))
                Xckk = G*Xkk*np.exp(-1j*(k[k1]-k[k2]))
            p_k2 = L + k2
            m_k2 = L - (k2+1)
            ppairing += [
                         [Xkk, p_k1, m_k1, m_k2, p_k2]
                        ]
            zpairing += [
                         [0.5*Xkk, p_k1, p_k2, m_k1, m_k2],
                         [0.5*Xkk, m_k1, m_k2, p_k1, p_k2],
                         [-0.5*Xkk, p_k1, m_k2, m_k1, p_k2],
                         [-0.5*Xkk, m_k1, p_k2, p_k1, m_k2]
                        ]
            spm += [
                    [0.5*Xskk, p_k1, p_k2, p_k1, p_k2],
                    [0.5*Xskk, p_k1, m_k2, p_k1, m_k2],
                    [0.5*Xskk, m_k1, p_k2, m_k1, p_k2],
                    [0.5*Xskk, m_k1, m_k2, m_k1, m_k2]
                    ]
            smp += [
                    [0.5*Xckk, p_k1, p_k2, p_k1, p_k2],
                    [0.5*Xckk, p_k1, m_k2, p_k1, m_k2],
                    [0.5*Xckk, m_k1, p_k2, m_k1, p_k2],
                    [0.5*Xckk, m_k1, m_k2, m_k1, m_k2]
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
    if no_kin:
        static = [
                ['++--|', ppairing],
                ['+-|+-', zpairing],
                ['|++--', ppairing],
                ['+-|-+', spm]
                ]
    else:
        static = [['n|', all_k], ['|n', all_k],
                ['++--|', ppairing], ['--++|', ppairing],
                ['+-|+-', zpairing], ['-+|-+', zpairing],
                ['|++--', ppairing], ['|--++', ppairing],
                ['+-|-+', spm], ['-+|+-', smp],
                ['nn|', samesame], # the up/down density density stuff cancels
                ['|nn', samesame],
                ['n|n', samesame],
                ['n|', dens],
                ['|n', dens]
                ]

    return {'static': static}
    return op_dict


def iom_dict(L, G, k, k1=0, mult=1, kin=1):

    ppairing = [] # spin 1 pairing
    zpairing = [] # spin 0 pairing
    samesame = []
    dens = []
    spm = [] # s+/s-  interaction
    p_k1 = L + k1 # index of +k, spin up fermion
    m_k1 = L - (k1+1) # index of -k, spin up fermion
    print('p_k1, m_k1 = {}, {}'.format(p_k1, m_k1))

    all_k = [[0.5*mult, p_k1], [0.5*mult, m_k1]]
    for k2 in range(L):
        if k2 != k1:
            Zkk = mult*G*k[k2]*k[k1]/(k[k2]-k[k1])
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
        neg_n += [[1.0, m_k]]
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


def find_sz(L, state, basis):
    szs = np.zeros(2*L)
    for k in range(2*L):
        upval = [[0.5, k]]
        dwnval = [[-0.5, k]]
        sz = [['n|', upval], ['|n', dwnval]]
        dic = {'static': sz}
        Sz = quantum_operator(dic, basis=basis)
        szs[k] = Sz.matrix_ele(state, state)
    return szs


def find_min_ev(operator, L, basis, n=1):
    e, v = operator.eigsh(k=1)
    if e[0] > 0:
        print('Positive eigenvalue: finding a minimum')
        cdict = constant_op_dict(L, np.max(e))
        cop = quantum_operator(cdict, basis=basis)
        e2, v = (cop-operator).eigsh(k=n)
        e = -(e2-e[0])
    if n == 1:
        return e[0], v[:, 0]
    else:
        return e[:n], v[:, :n]


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


def ham_op(L, G, ks, basis, rescale_g=False):
    if rescale_g:
        g = G/(1+G*np.sum(ks))
    else:
        g = G
    for i in range(L):
        id = iom_dict(L, g, ks, k1=i, mult=ks[i], kin=1)
        if i == 0:
            h = quantum_operator(id, basis=basis)
        else:
            h += quantum_operator(id, basis=basis)
    return h




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    L = int(input('L: '))
    ks = np.array([(2*i+1)*np.pi/L for i in range(L)])
    # ks = np.arange(L) + 1.0
    G = float(input('G: '))
    Nup = int(input('Nup: '))
    Ndown = int(input('Ndown: '))
    basis = form_basis(2*L, Nup, Ndown)

    cd = casimir_dict(L, 1)
    co = quantum_operator(cd, basis=basis)
    print(G)
    h = ham_op(L, G, ks, basis)
    if L <= 4 and Nup < 3:
        es, vs = h.eigh()
    else:
        es, vs = find_min_ev(h, L, basis, 10)
    print('GS energy:')
    print(es)
    input('press enter to continue')
    nk0 = find_nk(L, vs[:,0], basis)
    nk1 = find_nk(L, vs[:,1], basis)

    plt.scatter(range(2*L), nk0)
    plt.scatter(range(2*L), nk1)
    plt.show()
    sz0 = find_sz(L, vs[:,0], basis)
    sz1 = find_sz(L, vs[:,1], basis)
    plt.scatter(range(2*L), sz0)
    plt.scatter(range(2*L), sz1)
    plt.show()


    input('Press enter to continue')

    gf = float(input('Final coupling: '))
    h = ham_op(L, gf, ks, basis)
    es, vs = find_min_ev(h, L, basis, 10)
    print('Lowest energies:')
    print(es)
    if gf > 0:
        gs = np.linspace(0, gf, 10)
    else:
        gs = np.linspace(gf, 0, 10)
    print(gs)
    e0s = np.zeros((10, 5))
    for i in range(10):
        print(i)
        h = ham_op(L, gs[i], ks, basis)
        if L < 5 and Nup < 3:
            e, v = h.eigh()
        else:
            e, v = find_min_ev(h, L, basis, 5)
        e0s[i] = e[:5]
    for i in range(5):
        plt.plot(gs, e0s[:, i])
    plt.show()

    if L < 4:
        for i, e in enumerate(es[:10]):
            v = vs[:, i]
            print('Casimir operator for {}th excited state'.format(i))
            print(co.matrix_ele(v, v))
            kfull = np.concatenate((ks[::-1], ks))
            v0 = vs[:,0]
            nks = find_nk(L, v0, basis)
            print('Nf: {}'.format(np.sum(nks)))
            plt.scatter(kfull, nks)
            plt.show()
