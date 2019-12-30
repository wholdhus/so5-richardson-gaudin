import numpy as np
from scipy import sparse as sp
from scipy.sparse import linalg

# Making sparse matrices for SO(5) operators
PM1 = sp.lil_matrix((5,5))
PM1[0,1] = 1.
PM1[3,4] = 1.

P0 = sp.lil_matrix((5,5))
P0[0,2] = 1.
P0[2,4] = -1.

P1 = sp.lil_matrix((5,5))
P1[0,3] = 1.
P1[1,4] = 1.

SM = sp.lil_matrix((5,5))
SM[1,2] = np.sqrt(2)
SM[2,3] = np.sqrt(2)

SZ = sp.dia_matrix(np.diag([0., -1., 0., 1., 0.]))
N = sp.dia_matrix(np.diag([0., 2., 2., 2., 4.]))

ID = sp.identity(5)

H = 0.5*N - ID

ZERO = sp.lil_matrix((5,5))

OPS = {'pm1': PM1,
       'p0': P0,
       'p1': P1,
       'pm1d': PM1.transpose(),
       'p0d': P0.transpose(),
       'p1d': P1.transpose(),
       'sm': SM,
       'sp': SM.transpose(),
       'sz': SZ,
       'h': H,
       'id': ID}

def big_op(op, L, i): # Takes kronecker products to put op in bigger thing
    before = sp.identity(5**i)
    after = sp.identity(5**(L-1-i)) # leftovers
    a = sp.kron(before, op) # if i == 0, before is just 1 so this is fine
    return sp.kron(a, after)


def construct_so5(L): # Builds L copies of the SO(5) algebra
    new_ops = {}
    for o in OPS:
        new_ops[o] = np.array([None for i in range(L)])
        op = OPS[o] # one copy of the matrix
        for i in range(L):
            new_ops[o][i] = big_op(op, L, i)
    return new_ops # dict of lists of operators

def make_Z(epsilon):
    L = len(epsilon)
    Z = np.zeros((L, L))
    for i in range(L):
        for j in range(i):
            Z[i, j] = (epsilon[i] * epsilon[j])/(epsilon[i] - epsilon[j])
            Z[j, i] = -1 * Z[i, j]
    return Z

def construct_ioms(epsilon, ops, g):
    L = len(epsilon)
    Z = make_Z(epsilon)
    ioms = [None for i in range(L)]
    for i in range(L):
        iom = ops['h'][i]
        for j in range(L):
            if i != j:
                iom += g*Z[j,i]*(ops['pm1d'][i].dot(ops['pm1'][j])
                                +ops['p0d'][i].dot(ops['p0'][j])
                                +ops['p1d'][i].dot(ops['p1'][j])
                                +ops['pm1'][i].dot(ops['pm1d'][j])
                                +ops['p0'][i].dot(ops['p0d'][j])
                                +ops['p1'][i].dot(ops['p1d'][j])
                                +0.5*ops['sp'][i].dot(ops['sm'][j])
                                +0.5*ops['sm'][i].dot(ops['sp'][j])
                                + ops['sz'][i].dot(ops['sz'][j])
                                + ops['h'][i].dot(ops['h'][j]))
        ioms[i] = iom
        # ioms[i] = iom + iom.transpose() # adding in hermitian transpose
    return ioms

def construct_ham(epsilon, ops, g):
    L = len(epsilon)
    H = np.sum(ops['h']*epsilon)
    for i in range(L):
        diags += -0.5*g*epsilon[i]**2*(
            ops['pm1d'][i].dot(ops['pm1'][i])
            +ops['p0d'][i].dot(ops['p0'][i])
            +ops['p1d'][i].dot(ops['p1'][i])
            +ops['pm1'][i].dot(ops['pm1d'][i])
            +ops['p0'][i].dot(ops['p0d'][i])
            +ops['p1'][i].dot(ops['p1d'][i])
            +0.5*ops['sp'][i].dot(ops['sm'][i])
            +0.5*ops['sm'][i].dot(ops['sp'][i])
            +ops['sz'][i].dot(ops['sz'][i])
            +ops['h'][i].dot(ops['h'][i]))
        for j in range(L):
            H += 0.5*g*epsilon[i]*epsilon[j]*(
                ops['pm1d'][i].dot(ops['pm1'][j])
                +ops['p0d'][i].dot(ops['p0'][j])
                +ops['p1d'][i].dot(ops['p1'][j])
                +ops['pm1'][i].dot(ops['pm1d'][j])
                +ops['p0'][i].dot(ops['p0d'][j])
                +ops['p1'][i].dot(ops['p1d'][j])
                +0.5*ops['sp'][i].dot(ops['sm'][j])
                +0.5*ops['sm'][i].dot(ops['sp'][j])
                +ops['sz'][i].dot(ops['sz'][j])
                +ops['h'][i].dot(ops['h'][j])
                )
    H += diags
    return H

def nk(L, ops, state):
    # Nks = 2*ops['h'] + 1*ops['id']
    Nks = ops['h']
    nks = np.zeros(L)
    for i in range(L):
        v = Nks[i].dot(state)
        nks[i] = state.dot(v)
    return nks



if __name__ == '__main__':
    L = 6
    g = -105
    import matplotlib.pyplot as plt
    list = construct_so5(L)
    print(list)
    print('Checking some commutators')
    print('[S_2^+, S_2^-] - 2 S_2^z')
    sz1 = list['sz'][1]
    sm1 = list['sm'][1]
    sp1 = list['sp'][1]
    com = sp1.dot(sm1) - sm1.dot(sp1) - 2*sz1
    print(com)
    print('[S_2^+, S_1^-]')
    sp2 = list['sp'][2]
    com2 = sp2.dot(sm1) - sm1.dot(sp2)
    print(np.max(com2))
    print(np.min(com2))

    epsilon = np.linspace(1, L, L)*np.pi/L
    ioms = construct_ioms(epsilon, list, g)
    com = ioms[0].dot(ioms[1])-ioms[1].dot(ioms[0])
    print('Max, min of [R_1, R_2]')
    print(np.max(com))
    print(np.min(com))

    H = construct_ham(epsilon, list, g)
    print('Hamiltonian!')
    print(H)

    e, v = linalg.eigsh(H, k=1) # Just getting one eigenvector/value
    print('First eigenvalue found:')
    print(e)
    if e[0] > 0:
        emax = e[0]
        e, v = linalg.eigsh(list['id'][1]*emax - H, k=4)
        e = -(e - emax)
    else:
        e, v = linalg.eigsh(H)
    print('Spectrum:')
    print(e)
    print(v[0])
    nks = 2*(nk(L, list, v[:,0])+1)
    plt.plot(nks)
    plt.show()
    print(np.sum(nks))
