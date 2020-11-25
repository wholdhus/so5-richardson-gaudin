from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator, quantum_LinearOperator
import numpy as np
from tqdm import tqdm


def form_basis(L, Nup, Ndown):
    basis = spinful_fermion_basis_1d(L, Nf=(Nup, Ndown))
    return basis


def reduce_state(v, full_basis, target_basis, test=False):
    #  v *= 1./np.linalg.norm(v)
    fdim = len(v)
    v_out = np.zeros(target_basis.Ns, dtype=np.complex128)
    for i, s in enumerate(target_basis.states):
        # full_ind = np.where(full_basis.states == s)[0][0]
        v_out[i] = v[fdim - s - 1]
        # v_out[i] = v[full_ind]
    if test:
        vf = target_basis.get_vec(v_out, sparse=False)
        print('<vin|vout>')
        print(np.vdot(v, vf))
        print('| |vin> - |vout> |')
        print(np.linalg.norm(v - vf))
        print('Equal?')
        print((v == vf).all())
        print('Norms')
        print(np.linalg.norm(v))
        print(np.linalg.norm(v_out))
    return v_out/np.linalg.norm(v_out)


def casimir_dict(L, k1, factor):
    p_k1 = L + k1 # index of +k, spin up fermion
    m_k1 = L - (k1+1) # index of -k, spin up fermion
    ppairing = [
                 [1*factor, p_k1, m_k1, m_k1, p_k1]
                ]
    zpairing = [
                 [0.5*factor, p_k1, p_k1, m_k1, m_k1],
                 [0.5*factor, m_k1, m_k1, p_k1, p_k1],
                 [-0.5*factor, p_k1, m_k1, m_k1, p_k1],
                 [-0.5*factor, m_k1, p_k1, p_k1, m_k1]
                ]
    samesame = [[0.5*factor, p_k1, p_k1],
                 [0.5*factor, p_k1, m_k1],
                 [0.5*factor, m_k1, p_k1],
                 [0.5*factor, m_k1, m_k1]
                ]
    spm = [
            [0.5*factor, p_k1, p_k1, p_k1, p_k1],
            [0.5*factor, p_k1, m_k1, p_k1, m_k1],
            [0.5*factor, m_k1, p_k1, m_k1, p_k1],
            [0.5*factor, m_k1, m_k1, m_k1, m_k1]
            ]
    dens = [
             [-1*factor, p_k1],
             [-1*factor, m_k1]
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


def hamiltonian_dict(L, G, k, no_kin=False, trig=False, couplings=None,
                     exactly_solvable=True):
    if couplings is not None:
        GT, GS, GN = couplings
    else:
        GT, GS, GN = (1, 1, 1)
    if not exactly_solvable:
        GT *= 2
        GS *= 2
    if G == -999:
        no_kin=True # easier to input this.
        G = -1
        print('Zero k.e. hamiltonian')
    G *= -1 # woops i had some sign ambiguities!
    # k should include positive and negative values
    all_k = []
    ppairing = [] # spin 1 pairing
    zpairing = [] # spin 0 pairing
    spm = [] # spin spin interaction
    same_same = [] # n_ksigma n_ksigma
    same_diff = [] # n_ksigma n_ksigma'
    for k1 in range(L):
        p_k1 = L + k1 # index of +k fermions
        m_k1 = L - (k1+1) # index of -k fermions

        all_k += [[k[k1], p_k1], [k[k1], m_k1]]

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
                         [Xkk*GT, p_k1, m_k1, m_k2, p_k2]
                        ]
            zpairing += [
                         [.5*Xkk*GT, p_k1, p_k2, m_k1, m_k2],
                         [.5*Xkk*GT, m_k1, m_k2, p_k1, p_k2],
                         [-.5*Xkk*GT, p_k1, m_k2, m_k1, p_k2],
                         [-.5*Xkk*GT, m_k1, p_k2, p_k1, m_k2]
                        ]
            s_c = 0.5*Xskk*GS
            # s_c = Xskk*g_spin
            spm += [
                    [s_c, p_k1, p_k2, p_k1, p_k2],
                    [s_c, p_k1, m_k2, p_k1, m_k2],
                    [s_c, m_k1, p_k2, m_k1, p_k2],
                    [s_c, m_k1, m_k2, m_k1, m_k2]
                    ]
            sz_c = 0.25*Zkk*GS
            d_c = 0.25*Zkk*GN
            same_same += [[d_c + sz_c, p_k1, p_k2],
                          [d_c + sz_c, p_k1, m_k2],
                          [d_c + sz_c, m_k1, p_k2],
                          [d_c + sz_c, m_k1, m_k2]
                         ]
            same_diff += [[2*(d_c - sz_c), p_k1, p_k2],
                          [2*(d_c - sz_c), p_k1, m_k2],
                          [2*(d_c - sz_c), m_k1, p_k2],
                          [2*(d_c - sz_c), m_k1, m_k2]
                         ]
    if no_kin:
        # static = [
        #           ['++--|', ppairing],
        #           ['+-|+-', zpairing],
        #           ['|++--', ppairing],
        #           ['+-|-+', spm]
        #         ]
        static = [
                    ['++--|', ppairing], ['--++|', ppairing],
                    ['+-|+-', zpairing], ['-+|-+', zpairing],
                    ['|++--', ppairing], ['|--++', ppairing],
                    ['+-|-+', spm], ['-+|+-', spm],
                    ['nn|', same_same],
                    ['|nn', same_same],
                    ['n|n', same_diff]
                ]
    elif not exactly_solvable:
        static = [['n|', all_k], ['|n', all_k],
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
                ['+-|-+', spm], ['-+|+-', spm],
                ['nn|', same_same],
                ['|nn', same_same],
                ['n|n', same_diff]
                ]

    return {'static': static}
    return op_dict



def iom_dict(L, g, k, k1=0, mult=1, kin=1, g_spin=1, g_dens=1):

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
            Zkk = mult*g*k[k2]*k[k1]/(k[k2]-k[k1])
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
        Nup = quantum_operator(dic, basis=basis,
                               check_herm=False, check_pcon=False, check_symm=False)
        nt = [['|n', nval]]
        dic = {'static': nt}
        Ndown = quantum_operator(dic, basis=basis,
                                 check_herm=False, check_pcon=False, check_symm=False)
        nks[k] = np.real(Nup.matrix_ele(state, state) + Ndown.matrix_ele(state,state))
    return nks

def find_skz(L, state, basis):
    sks = np.zeros(2*L)
    for k in range(2*L):
        nval = [[1.0, k]]
        nt = [['n|', nval]]
        dic = {'static': nt}
        Nup = quantum_operator(dic, basis=basis,
                               check_herm=False, check_pcon=False,
                               check_symm=False)
        nt = [['|n', nval]]
        dic = {'static': nt}
        Ndown = quantum_operator(dic, basis=basis,
                                 check_herm=False, check_pcon=False,
                                 check_symm=False)
        sks[k] = np.real(0.5*(Nup.matrix_ele(state, state) - Ndown.matrix_ele(state,state)))
    return sks


def find_sz(L, state, basis):
    szs = np.zeros(2*L)
    for k in range(2*L):
        upval = [[0.5, k]]
        dwnval = [[-0.5, k]]
        sz = [['n|', upval], ['|n', dwnval]]
        dic = {'static': sz}
        Sz = quantum_operator(dic, basis=basis, check_herm=False, check_pcon=False,
                              check_symm=False)
        szs[k] = Sz.matrix_ele(state, state)
    return szs

def find_qk(L, state, basis):
    qks = np.zeros(L)
    for k in range(L):
        qd = casimir_dict(L, k, 1)
        qo = quantum_operator(qd, basis=basis, check_herm=False, check_pcon=False,
                              check_symm=False)
        qks[k] = qo.matrix_ele(state, state)
    return qks


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


def ham_op(L, G, ks, basis, dtype=np.float64,
           diagonal_terms=True):
    g = G/(1-G*np.sum(ks))
    factor = 2/(1+g*np.sum(ks))
    for i in range(L):
        id = iom_dict(L, g, ks, k1=i, mult=ks[i]*factor, kin=1)
        if i == 0:
            h = quantum_operator(id, basis=basis, dtype=dtype)
        else:
            h += quantum_operator(id, basis=basis, dtype=dtype)
        if diagonal_terms:
            # cd = casimir_dict(L, i, factor = G*ks[i]**2)
            cd = constant_op_dict(L, 3*G*ks[i]**2)
            co = quantum_operator(cd, basis=basis, dtype=dtype)
            h -= co

    return h

def ham_op_2(L, G, ks, basis, no_kin=False, couplings=None,
             exactly_solvable=True):
    hd = hamiltonian_dict(L, G, ks, no_kin=no_kin, couplings=couplings,
                          exactly_solvable=exactly_solvable)
    h = quantum_operator(hd, basis=basis,
                         check_herm=False, check_pcon=False, check_symm=False)
    return h


def periodic_ham(l, G, basis):
    # Arrangement: k = (Pi/L)[-L, -L+2, ..., -2, 0, 2, ..., L-2]
    k = np.pi*np.arange(-2*l, 2*l, 2)/(2*l)
    eta = np.abs(k)
    L = 2*l
    # k should include positive and negative values
    kin_e = [[eta[ki], ki] for ki in range(L)]
    ppairing = [] # spin 1 pairing
    zpairing = [] # spin 0 pairing
    spm = [] # spin spin interaction
    same_same = [] # n_ksigma n_ksigma
    same_diff = [] # n_ksigma n_ksigma'
    
    
    for k1 in range(L):
        for k2 in range(L):
            Vkk = -G*eta[k1]*eta[k2]
            spm += [[.5*Vkk, k1, k2, k1, k2]]
            same_same += [[.5*Vkk, k1, k2]
                         ]
            if k1 >= l+1 and k2 >= l+1: # leaving out k=-pi, k=0 modes
                mk1 = L-k1
                mk2 = L-k2
                if mk1 == 0 or mk2 == 0 or mk1 == l or mk2 == l:
                    print('Woah! -k1, -k2 = {}'.format(k[mk1], k[mk2]))
                # print('+k, +k\', -k\', -k')
                # print((k[k1], k[k2], k[mk2], k[mk1]))
                ppairing += [[Vkk, k1, mk1, mk2, k2]]
                zpairing += [
                             [.5*Vkk, k1, k2, mk1, mk2],
                             [.5*Vkk, mk1, mk2, k1, k2],
                             [-.5*Vkk, k1, mk2, mk1, k2],
                             [-.5*Vkk, mk1, k2, k1, mk2]
                            ]
            
    static = [['n|', kin_e], ['|n', kin_e],
                ['++--|', ppairing], ['--++|', ppairing],
                ['+-|+-', zpairing], ['-+|-+', zpairing],
                ['|++--', ppairing], ['|--++', ppairing],
                ['+-|-+', spm], ['-+|+-', spm],
                ['nn|', same_same],
                ['|nn', same_same],
                ]
    return quantum_operator({'static': static}, basis=basis,
                            check_herm=False, check_pcon=False, check_symm=False)


def antiperiodic_ham(l, G, basis):
    k  = np.pi*np.arange(-2*l+1, 2*l, 2)/(2*l)
    eta = np.abs(k)
    L = 2*l
    # k should include positive and negative values
    kin_e = [[eta[ki], ki] for ki in range(L)]
    ppairing = [] # spin 1 pairing
    zpairing = [] # spin 0 pairing
    spm = [] # spin spin interaction
    same_same = [] # n_ksigma n_ksigma
    same_diff = [] # n_ksigma n_ksigma'
    
    
    for k1 in range(L):
        for k2 in range(L):
            Vkk = -G*eta[k1]*eta[k2]
            spm += [[.5*Vkk, k1, k2, k1, k2]]
            same_same += [[.5*Vkk, k1, k2]
                         ]
            if k1 >= l and k2 >= l:
                mk1 = L-k1-1
                mk2 = L-k2-1
                ppairing += [[Vkk, k1, mk1, mk2, k2]]
                zpairing += [
                             [.5*Vkk, k1, k2, mk1, mk2],
                             [.5*Vkk, mk1, mk2, k1, k2],
                             [-.5*Vkk, k1, mk2, mk1, k2],
                             [-.5*Vkk, mk1, k2, k1, mk2]
                            ]
            
    static = [['n|', kin_e], ['|n', kin_e],
                ['++--|', ppairing], ['--++|', ppairing],
                ['+-|+-', zpairing], ['-+|-+', zpairing],
                ['|++--', ppairing], ['|--++', ppairing],
                ['+-|-+', spm], ['-+|+-', spm],
                ['nn|', same_same],
                ['|nn', same_same],
                ]
    return quantum_operator({'static': static}, basis=basis, check_herm=False, check_pcon=False,
                            check_symm=False)



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
            # print('Done with {} {} term in double sum'.format(m, n))
    outs = np.zeros(len(vs), dtype=np.complex128)
    for i, v in enumerate(vs):
        outs[i] = np.vdot(v, pvs[i])/(2*L**2)
    return outs


def quartet_wavefunction(L, N, basis, basisf):
    if N%4 != 0:
        print('Woah. You can only have quartets with multiples of 4 fermions!')
        return
    # vacuum state
    v0 = np.zeros(basisf.Ns, dtype=np.complex128)
    v0[-1] = 1
    creation_lst = []
    for i in range(L):
        for j in range(L):
            ki_p = L+i
            ki_m = L-i-1
            kj_p = L+j
            kj_m = L-j-1
            creation_lst += [[1, ki_p, ki_m, kj_p, kj_m], #T_1 T_-1
                             [1, kj_p, kj_m, ki_p, ki_m], #T_-1 T_1
                             [-1, ki_p, kj_p, ki_m, kj_m], # -1
                             [+1, ki_p, kj_m, ki_m, kj_p],
                             [-1, ki_m, kj_m, ki_p, kj_p], # -1
                             [+1, ki_m, kj_p, ki_p, kj_m]
                             ]
    creation_op = quantum_operator({'static': [['++|++', creation_lst]]}, basis=basisf,
                                   check_herm=False, check_symm=False)
    v = v0/np.linalg.norm(v0)
    nk = find_nk(L, v, basisf)
    print('Before doing anything')
    print('sum_k n_k')
    print(np.sum(nk))
    for i in range(N//4):
        v = creation_op.dot(v)
        v *= 1./np.linalg.norm(v)
        nk = find_nk(L, v, basisf)
        print('sum_k n_k')
        print(np.sum(nk))
    return reduce_state(v, basisf, basis)


def iso_wavefunction(L, N, basis, basisf):
    sum_t0_l = [['+|+', [[1/np.sqrt(2), L+l, L-l-1] for l in range(L)]],
                ['+|+', [[-1/np.sqrt(2), L-l-1, L+l] for l in range(L)]]]
    sum_t0_o = quantum_operator({'static': sum_t0_l}, basis=basisf,
                                check_herm=False, check_symm=False)
    v0 = np.zeros(basisf.Ns, dtype=np.complex128)
    v0[-1] = 1 # vacuum state!
    print('sum_k n_k in vacuum')
    nk = find_nk(L, v0, basisf)
    print(np.sum(nk))

    v = sum_t0_o.dot(v0)
    v *= 1./np.linalg.norm(v)
    nk = find_nk(L, v, basisf)
    print('sum_k n_k')
    print(np.sum(nk))
    for i in range(N//2 - 1):
        v = sum_t0_o.dot(v)
        v *= 1./np.linalg.norm(v)
        nk = find_nk(L, v, basisf)
        print('sum_k n_k')
        print(np.sum(nk))
    print('Now reducing to the N particle basis')
    return reduce_state(v, basisf, basis) # putting into the N particle basis


if __name__ == '__main__':
    L = 4
    ks = np.array([(2*i+1)*np.pi/(2*L) for i in range(L)])
    print(ks)
    # sep = int(input('Pair separation: '))
    Nup = 6
    Ndown = 6
    basis = form_basis(2*L, Nup, Ndown)
    basisf = spinful_fermion_basis_1d(2*L)

    Gc = 1./np.sum(ks)
    h = ham_op_2(L, Gc, ks, basis)
    # es, v = h.eigsh(k=10, which='SA')
    es, v = h.eigh()
    print('Ground state energy:')
    print(es[0])
    print('Constructing ?ground state?')
    v0 = iso_wavefunction(L, Nup+Ndown, basis, basisf)
    v0 *= 1./np.linalg.norm(v0)
    print('<vRG|H|vRG> - E0')
    print(h.matrix_ele(v0, v0) - es[0])
    print('Q_k')
    print(find_qk(L, v0, basis))

    print('Constructing quartet state')
    v4 = quartet_wavefunction(L, Nup+Ndown, basis, basisf)
    v4 *= 1./np.linalg.norm(v4)
    print('<v4|H|v4> - E0')
    print(h.matrix_ele(v4, v4) - es[0])
    print('Q_k')
    print(find_qk(L, v4, basis))

    print('Overlap between all the guys')
    print('|<vRG|v4>|')
    print(np.abs(np.vdot(v0, v4)))

    print('Degeneracy at this case')
    print(len(es[np.abs(es - es[0]) < 10**-12]))

    h_above = ham_op_2(L, 1.1*Gc, ks, basis)
    es, v = h_above.eigh()
    print('Degeneracy above:')
    print(len(es[np.abs(es - es[0]) < 10**-12]))

    h_below = ham_op_2(L, .9*Gc, ks, basis)
    es, v = h_below.eigh()
    print('Degeneracy below:')
    print(len(es[np.abs(es - es[0]) < 10**-12]))

    print('Above: is quartet eigenstate?')
    print(np.max(np.abs(h_above.dot(v4) - h_above.matrix_ele(v4, v4)*v4)))
    print('Below: is quartet eigenstate?')
    print(np.max(np.abs(h_below.dot(v4) - h_below.matrix_ele(v4, v4)*v4)))
