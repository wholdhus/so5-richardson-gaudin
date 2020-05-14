from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator, quantum_LinearOperator
import numpy as np
from tqdm import tqdm



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
    if G == -999:
        no_kin=True # easier to input this.
        G = 1
        print('Zero k.e. hamiltonian')
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
                         [2*Xkk, p_k1, m_k1, m_k2, p_k2]
                        ]
            zpairing += [
                         [Xkk, p_k1, p_k2, m_k1, m_k2],
                         [Xkk, m_k1, m_k2, p_k1, p_k2],
                         [-1*Xkk, p_k1, m_k2, m_k1, p_k2],
                         [-1*Xkk, m_k1, p_k2, p_k1, m_k2]
                        ]
            s_c = 0.5*Xskk
            # s_c = Xskk*g_spin
            spm += [
                    [s_c, p_k1, p_k2, p_k1, p_k2],
                    [s_c, p_k1, m_k2, p_k1, m_k2],
                    [s_c, m_k1, p_k2, m_k1, p_k2],
                    [s_c, m_k1, m_k2, m_k1, m_k2]
                    ]
            d_c = 0.5*Zkk
            if k1 != k2:
                samesame += [[0.5*d_c, p_k1, p_k2],
                             [0.5*d_c, p_k1, m_k2],
                             [0.5*d_c, m_k1, p_k2],
                             [0.5*d_c, m_k1, m_k2]
                            ]
                dens += [
                         [-d_c, p_k1],
                         [-d_c, m_k1],
                         [-d_c, p_k2],
                         [-d_c, m_k2]
                        ]
    if no_kin:
        static = [
                ['++--|', ppairing],
                ['+-|+-', zpairing],
                ['|++--', ppairing]
                # ['+-|-+', spm]
                ]
    else:
        static = [['n|', all_k], ['|n', all_k],
                ['++--|', ppairing], # ['--++|', ppairing],
                ['+-|+-', zpairing], # ['-+|-+', zpairing],
                ['|++--', ppairing], # ['|--++', ppairing],
                # ['+-|-+', spm], ['-+|+-', spm],
                # ['nn|', samesame], # the up/down density density stuff cancels
                # ['|nn', samesame]
                # ['n|', dens],
                # ['|n', dens]
                ]

    return {'static': static}
    return op_dict


def iom_dict(L, G, k, k1=0, mult=1, kin=1, g_spin=1, g_dens=1):

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
            d = 0.5*Zkk*g_dens
            samesame += [[d, p_k1, p_k2],
                         [d, p_k1, m_k2],
                         [d, m_k1, p_k2],
                         [d, m_k1, m_k2]
                        ]
            s = 0.5*Zkk*g_spin
            spm += [
                    [s, p_k1, p_k2, p_k1, p_k2],
                    [s, p_k1, m_k2, p_k1, m_k2],
                    [s, m_k1, p_k2, m_k1, p_k2],
                    [s, m_k1, m_k2, m_k1, m_k2]
                    ]
            d = -0.5*Zkk*g_dens
            dens += [
                     [d, p_k1],
                     [d, m_k1],
                     [d, p_k2],
                     [d, m_k2]
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

def find_skz(L, state, basis):
    sks = np.zeros(2*L)
    for k in range(2*L):
        nval = [[1.0, k]]
        nt = [['n|', nval]]
        dic = {'static': nt}
        Nup = quantum_operator(dic, basis=basis)
        nt = [['|n', nval]]
        dic = {'static': nt}
        Ndown = quantum_operator(dic, basis=basis)
        sks[k] = 0.5*(Nup.matrix_ele(state, state) - Ndown.matrix_ele(state,state))
    return sks


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


def ham_op(L, G, ks, basis, rescale_g=False, dtype=np.float64):
    factor = 1
    if rescale_g:
        g = G/(1+G*np.sum(ks))
        print('g = {}'.format(g))
        factor = 2/(1-g*np.sum(ks))
    else:
        g = G
    for i in range(L):
        id = iom_dict(L, g, ks, k1=i, mult=ks[i]*factor, kin=1)
        if i == 0:
            h = quantum_operator(id, basis=basis, dtype=dtype)
        else:
            h += quantum_operator(id, basis=basis, dtype=dtype)
    return h

def ham_op_2(L, G, ks, basis, rescale_g=True, no_kin=False):
    hd = hamiltonian_dict(L, G, ks, no_kin=no_kin)

    h = quantum_operator(hd, basis=basis, check_herm=False)
    return h


def pair_correlation(v, l1, l2, ks, basis, s1=0, s2=0):
    # assuming l1, l2 are one-indexed
    # calculates <n_{l1}n_{l2}>
    # if s1 (s2) >= 0, use n_{\up}, else n_{\down}
    L = len(ks)
    ka = np.concatenate((-1*ks[::-1], ks))
    l1_lst = []
    l2_lst = []
    for i, k1 in enumerate(ka):
        for j, k2 in enumerate(ka):
            l1_lst += [[np.exp(1j*(k1-k2)*(l1))/(2*L), i, j]]
            l2_lst += [[np.exp(1j*(k1-k2)*(l2))/(2*L), i, j]]

    op1 = '+-|'
    op2 = '+-|'
    if s1 < 0:
        op1 = '|-+'
    if s2 < 0:
        op2 = '|-+'
    l1_lo = quantum_operator({'static': [[op1, l1_lst]]}, basis=basis)

    l2_lo = quantum_operator({'static': [[op2, l2_lst]]}, basis=basis)

    n1 = l1_lo.matrix_ele(v,v)
    n2 = l2_lo.matrix_ele(v,v)
    if l1 == l2:
        return np.vdot(v, l1_lo.matvec(l2_lo.matvec(v))) - n1
    else:
        return np.vdot(v, l1_lo.matvec(l2_lo.matvec(v)))


def eta(x, ks):
    return -2j*np.sum(ks*np.sin(ks*x))
    # if x == 0:
    #     return 0
    # else:
    #     return 1./np.abs(x)

def pairing_correlation(vs, i, j, ks, basis):

    L = 2*len(ks)
    ls = np.arange(1, L+1)
    ka = np.concatenate((-1*ks[::-1], ks))

    pvs = [np.zeros(len(v), dtype=np.complex128) for v in vs]

    for m in tqdm(ls):
        for n in ls:
            pair1_lst = []
            pair2_lst = []
            pair3_lst = []
            pair4_lst = []
            # pairing function
            pf = eta(i-m, ks)*np.conjugate(eta(j-n, ks))
            for k1_ind, k1 in enumerate(ka):
                for k2_ind, k2 in enumerate(ka):
                    # c_i^+ c_j
                    pair1_lst += [[np.exp(1j*(k1*i - k2*j))/L, k1_ind, k2_ind]]
                    # c_m^+ c_n
                    pair2_lst += [[np.exp(1j*(k1*m - k2*n))/L, k1_ind, k2_ind]]
                    # c_i^+ c_n
                    pair3_lst += [[np.exp(1j*(k1*i - k2*n))/L, k1_ind, k2_ind]]
                    # c_m^+ c_j
                    pair4_lst += [[np.exp(1j*(k1*m - k2*j))/L, k1_ind, k2_ind]]
            if m == 1: # otherwise, we've already defined these
                ij_up = quantum_operator({'static': [['+-|', pair1_lst]]}, basis=basis,
                                check_herm=False, check_pcon=False, check_symm=False)
                ij_down = quantum_operator({'static': [['|+-', pair1_lst]]}, basis=basis,
                                check_herm=False, check_pcon=False, check_symm=False)
            if n == 1: # again, otherwise we already know this
                mj_up = quantum_operator({'static': [['+-|', pair4_lst]]}, basis=basis,
                                 check_herm=False, check_pcon=False, check_symm=False)
                mj_down = quantum_operator({'static': [['|+-', pair4_lst]]}, basis=basis,
                                 check_herm=False, check_pcon=False, check_symm=False)
            # depends on n, so need to do every time
            mn_up = quantum_operator({'static': [['+-|', pair2_lst]]}, basis=basis,
                             check_herm=False, check_pcon=False, check_symm=False)
            mn_down = quantum_operator({'static': [['|+-', pair2_lst]]}, basis=basis,
                             check_herm=False, check_pcon=False, check_symm=False)
            in_up = quantum_operator({'static': [['+-|', pair3_lst]]}, basis=basis,
                             check_herm=False, check_pcon=False, check_symm=False)
            in_down = quantum_operator({'static': [['|+-', pair3_lst]]}, basis=basis,
                             check_herm=False, check_pcon=False, check_symm=False)

            for vi, v in enumerate(vs):
                pvs[vi] += pf*(ij_up.dot(mn_down.dot(v)) + ij_down.dot(mn_up.dot(v))
                               -(in_up.dot(mj_down.dot(v)) + in_down.dot(mj_up.dot(v))))
                # pvs[vi] += np.conjugate(pvs[vi])
            # print('Done with {} {} term in double sum'.format(m, n))
    outs = np.zeros(len(vs), dtype=np.complex128)
    for i, v in enumerate(vs):
        outs[i] = np.vdot(v, pvs[i])/(2*L**2)
    return outs


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys


    # L = int(input('L: '))
    L = int(sys.argv[1])
    ks = np.array([(2*i+1)*np.pi/(2*L) for i in range(L)])
    print(ks)
    # ks = np.arange(L) + 1.0
    # Nup = int(input('Nup: '))
    # Ndown = int(input('Ndown: '))
    # sep = int(input('Pair separation: '))
    Nup = int(sys.argv[2])
    Ndown = int(sys.argv[3])
    basis = form_basis(2*L, Nup, Ndown)

    Gs = np.array([-0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5])/L
    vs = [None for G in Gs]
    for i, G in enumerate(Gs):
        hp = ham_op_2(L, G, ks, basis)
        ep, vp = hp.eigsh(k=1, which='SA')
        vs[i] = vp[:, 0]

    ls = np.arange(1, L+1)
    pcs = [np.zeros(L) for G in Gs]

    l1 = 1
    for i in range(L):
        print('')
        print('!!!!!!!!!!!!!!!!!!!!!!!')
        print(i)
        print('!!!!!!!!!!!!!!!!!!!!!!!')
        print('')
        # l2 = i%(2*L) + 1
        l2 = i+1
        outs = pairing_correlation(vs, l1, l2, ks, basis)
        print('L2: {}'.format(l2))
        print(outs)
        for j, o in enumerate(outs):
            pcs[j][i] = o


    dens = .25*(Nup+Ndown)/L
    for i, G in enumerate(Gs):
        plt.plot(np.abs(ls-l1), pcs[i]**2, label='G = {}'.format(G))

    plt.xlabel(r'$|a-b|$')
    plt.ylabel(r'$P_{ab}^2$')
    # plt.legend()
    plt.title(r'Pairing correlation, L = {}, N = {}'.format(
              2*L, Nup + Ndown))
    plt.legend()
    # plt.show()
    plt.savefig('pairing_L{}N{}.png'.format(2*L, Nup+Ndown))
