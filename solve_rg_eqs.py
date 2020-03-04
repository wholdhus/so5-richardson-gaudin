import numpy as np
from scipy.optimize import root, minimize

VERBOSE=True
TOL=10**-8
MAXIT=10000
FACTOR=100

lmd = {'maxiter': MAXIT,
       'xtol': TOL,
       'ftol': TOL, 'factor':FACTOR}

hybrd = {'maxfev': MAXIT,
         'xtol': TOL,
         'factor': FACTOR}


def log(msg):
    if VERBOSE:
        print(msg)

def rationalZ(x, y):
    return x*y/(x-y)

def reZ(x,y,u,v): # Re(Z(z1,z2)), z1 = x+iy, x2 = u+iv
    return ((x*u-y*v)*(x-u)+(y*u+x*v)*(y-v))/((x-u)**2+(y-v)**2)

def imZ(x,y,u,v):
    return ((x*u-y*v)*(v-y)+(y*u+x*v)*(x-u))/((x-u)**2+(y-v)**2)

def dZ_rr(x,y,u,v):
    # derivative of real part with respect to (2nd) real part
    return (((u+v-x)*x +(v-u)*y - y**2)*(v*y + u*(x+y)-x*(v+x) -y**2)
            )/((u-x)**2+(v-y)**2)**2


def dZ_ii(x,y,u,v):
    return dZ_rr(x,y,u,v)


def dZ_ri(x,y,u,v):
    # d Re(Z(z1, z2))/d Im(z2)
    return (2*(v*x-u*y)*((u-x)*x + (v-y)*y)
            )/(((u-x)**2+(v-y)**2)**2)


def dZ_ir(x,y,u,v):
    # d Im(Z(z1,z2))/d Re(z2)
    return -1*dZ_ri(x,y,u,v)


def trigZ(x, y):
    return 1./np.tan(x-y)


def unpack_vars(vars, Ne, Nw):
    # Variable vector: [re(e)..., im(e)..., re(w)..., im(w)...]
    ces = vars[:Ne] + 1j*vars[Ne:2*Ne]
    cws = vars[2*Ne:2*Ne+Nw] + 1j*vars[2*Ne+Nw:]

    return ces, cws


def pack_vars(ces, cws):
    # Takes separate, complex vectors
    # Combines them into variable vector format
    es = np.concatenate((ces.real, ces.imag))
    ws = np.concatenate((cws.real, cws.imag))
    vars = np.concatenate((es, ws))
    return vars


def rgEqs(vars, k, g, dims):
    c1 = 1

    L, Ne, Nw = dims

    Zf = rationalZ
    L = len(k)//2

    kr = k[:L]
    ki = k[L:]

    ers = vars[:Ne]
    eis = vars[Ne:2*Ne]
    wrs = vars[2*Ne:2*Ne+Nw]
    wis = vars[2*Ne+Nw:]

    set1_r = np.zeros(Ne)
    set1_i = np.zeros(Ne)
    set2_r = np.zeros(Nw)
    set2_i = np.zeros(Nw)

    for i, er in enumerate(ers):
        ei = eis[i]
        js = np.arange(Ne) != i
        set1_r[i] = ((2*reZ(ers[js], eis[js], er, ei).sum()
                      - reZ(wrs, wis, er, ei).sum()
                      - reZ(kr, ki, er, ei).sum())
                   + c1/g)
        set1_i[i] = ((2*imZ(ers[js], eis[js], er, ei).sum()
                      - imZ(wrs, wis, er, ei).sum()
                      - imZ(kr, ki, er, ei).sum()))
    for i, wr in enumerate(wrs):
        wi = wis[i]
        js = np.arange(Nw) != i
        set2_r[i] = (reZ(wrs[js], wis[js], wr, wi).sum()
                     - reZ(ers, eis, wr, wi).sum()
                    )
        set2_i[i] = (imZ(wrs[js], wis[js], wr, wi).sum()
                     - imZ(ers, eis, wr, wi).sum()
                    )
    eqs = np.concatenate((set1_r, set1_i, set2_r, set2_i))
    return g*eqs

def rg_scalar(vars, k, g, dims):
    return np.sum(np.abs(rgEqs(vars, k, g, dims)))


def rg_jac(vars, k, g, dims):
    """
    ASSUMING NE = NW!

    f1 is function on RHS of first set of equations
    f2 is function on RHS of second set of equations
    """
    L, Ne, Nw = dims
    N = Ne
    jac = np.zeros((len(vars), len(vars)))

    ers = vars[:Ne]
    eis = vars[Ne:2*Ne]
    wrs = vars[2*Ne:2*Ne+Nw]
    wis = vars[2*Ne+Nw:]

    krs = k[:L]
    kis = k[L:]

    # print('Variables:')
    # print('g = {}'.format(g))
    # print('e_alpha = ')
    # print(ers)
    # print(eis)
    # print('w_beta = ')
    # print(wrs)
    # print(wis)
    # print('k = ')
    # print(krs)
    # print(kis)

    for i in range(N):
        for j in range(N):
            if i == j:
                ls = np.arange(N) != j
                # Re(f1), Re(e)
                jac[i, j] = (2*np.sum(dZ_rr(ers[ls], eis[ls], ers[i], eis[i]))
                               -np.sum(dZ_rr(wrs, wis, ers[i], eis[i]))
                               -np.sum(dZ_rr(krs, kis, ers[i], eis[i]))
                               )
                # Re(f1), Im(e)
                jac[i, j+N] = (2*np.sum(dZ_ri(ers[ls], eis[ls], ers[i], eis[i]))
                                  -np.sum(dZ_ri(wrs, wis, ers[i], eis[i]))
                                  -np.sum(dZ_ri(krs, kis, ers[i], eis[i]))
                                  )
                # Im(f1), Re(e)
                # -1 * previous by properties of Z!
                jac[i+N, j] = -1*jac[i, j+N]
                # Im(f1), Im(e)
                # same as dRe(f)/dRe(e)
                jac[i+N, j+N] = jac[i, j]

                # Re(f2), Re(w)
                jac[i+2*N, j+2*N] = (np.sum(dZ_rr(wrs[ls], wis[ls], wrs[i], wis[i]))
                                       -np.sum(dZ_rr(ers, eis, wrs[i], wis[i])))
                # Re(f2), Im(w)
                jac[i+2*N, j+3*N] = (np.sum(dZ_ri(wrs[ls], wis[ls], wrs[i], wis[i]))
                                       -np.sum(dZ_ri(ers, eis, wrs[i], wis[i])))
                # Im(f2), Re(w)
                jac[i+3*N, j+2*N] = -1*jac[i+2*N, j+3*N]
                # Im(f2), Im(w)
                jac[i+3*N, j+3*N] = jac[i+2*N, j+2*N]

            else: # i != j
                """
                For the following, there is a factor of -1 and
                index order is switched in the calls to the
                derivative functions, because dZ(x,y)/dx = - dZ(y,x)/dx
                and the derivative functions are calculated w.r.t. the
                real and imaginary parts of the second variable
                """
                # Re(f1), Re(e)
                jac[i, j] = -2*dZ_rr(ers[i], eis[i], ers[j], eis[j])
                # Re(f1), Im(e)
                jac[i, j+N] = -2*dZ_ri(ers[i], eis[i], ers[j], eis[j])
                # Im(f1), Re(e)
                jac[i+N, j] = -1*jac[i, j+N]
                # Im(f1), Im(e)
                jac[i+N, j+N] = jac[i, j]

                # Re(f2), Re(w)
                jac[i+2*N, j+2*N] = -1*dZ_rr(wrs[i], wis[i], wrs[j], wis[j])
                # Re(f2), Im(w)
                jac[i+2*N, j+3*N] = -1*dZ_ri(wrs[i], wis[i], wrs[j], wis[j])
                # Im(f2), Re(w)
                jac[i+3*N, j+2*N] = -1*jac[i+2*N, j+3*N]
                # Im(f2), Im(w)
                jac[i+3*N, j+3*N] = jac[i+2*N, j+2*N]
            """
            Cross derivatives (f1 / w and f2 / e) take the same
            form when i == j, i != j.
            Again, there is a factor of -1 and switched variables because these
            derivatives are w.r.t. the first complex variable.
            """
            # Re(f1), Re(w)
            jac[i, j+2*N] = dZ_rr(ers[i], eis[i], wrs[j], wis[j])
            # Re(f1), Im(w)
            jac[i, j+3*N] = dZ_ri(ers[i], eis[i], wrs[j], wis[j])
            # Im(f1), Re(w)
            jac[i+N, j+2*N] = -1*jac[i, j+3*N]
            # Im(f1), Im(w)
            jac[i+N, j+3*N] = jac[i, j+2*N]

            # Re(f2), Re(e)
            jac[i+2*N, j] = dZ_rr(wrs[i], wis[i], ers[j], eis[j])
            # Re(f2), Im(e)
            jac[i+2*N, j+N] = dZ_ri(wrs[i], wis[i], ers[j], eis[j])
            # Im(f2), Re(e)
            jac[i+3*N, j] = -1*jac[i+2*N,j+N]
            # Im(f2), Im(e)
            jac[i+3*N, j+N] = jac[i+2*N, j]

    return g*jac


def g0_guess(L, Ne, Nw, kc, imscale=0.01):
    k_r = kc[:L]
    k_i = kc[L:]
    double_e = np.arange(Ne)//2
    double_w = np.arange(Nw)//2
    er = k_r[double_e]
    wr = k_r[double_w]
    ei = 1.8*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Ne)])
    wi = -3.2*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Nw)])
    # ei += k_i[double_e]
    # wi += k_i[double_w]
    if Nw%2 == 1: # Nw is odd
        wi[-1] = 0
        wr[-1] = 0
    vars = np.concatenate((er, ei, wr, wi))
    return vars


def dvars(vars, pvars, dg, Ne, Nw):
    ces, cws = unpack_vars(vars, Ne, Nw)
    pes, pws = unpack_vars(pvars, Ne, Nw)
    de = (ces-pes)/dg
    dw = (cws-pws)/dg
    deriv = pack_vars(de, dw)

    return deriv


def increment_im_k(vars, dims, g, k, im_k, steps=100, sf=1):
    L, Ne, Nw = dims
    scale = 1 - np.linspace(0, sf, steps)
    for i, s in enumerate(scale):

        kc = np.concatenate((k, s*im_k))
        sol = root(rgEqs, vars, args=(kc, g, dims),
                   method='lm', jac=rg_jac, options=lmd)

        vars = sol.x
        er = np.abs(rgEqs(vars, kc, g, dims))
        if np.max(er) > 10**-10:
            log('Highish errors:')
            log('s = {}'.format(s))
            log(np.max(er))
        if np.max(er) > 0.001:
            print('This is too bad')
            return
    return vars, er


def solve_rgEqs(dims, gf, k, dg=0.01, g0=0.001, imscale_k=0.01, imscale_v=0.001):

    L, Ne, Nw = dims
    g1sc = 0.01*4/L
    if gf > g1sc*L:
        g1 = g1sc*L
        g1s = np.arange(g0, g1, 0.5*dg)
        g2s = np.append(np.arange(g1, gf, dg), gf)
    elif gf < -1*g1sc*L:
        g1 = -1*g1s*L
        g1s = -1*np.linspace(g0, -1*g1, dg)
        g2s = np.append(-1*np.arange(-1*g1, -1*gf, dg), gf)
    else:
        print('Woops: abs(gf) < abs(g1)')
        return
    log('Paths for g:')
    log(g1s)
    log(g2s)
    print('')
    # imscale=0.1*dg
    # kim = imscale_k*np.cos(np.pi*np.arange(L))
    kim = imscale_k*(-1)**np.arange(L)
    kc = np.concatenate((k, kim))

    vars = g0_guess(L, Ne, Nw, kc, imscale=imscale_v)
    log('Initial guesses:')
    es, ws = unpack_vars(vars, Ne, Nw)
    es -= g0*np.arange(1, Ne+1)
    ws -= g0*np.arange(1, Nw+1)
    if Nw%2==1:
        ws[-1] = 0
    print(es)
    print(ws)

    vars = pack_vars(es, ws)
    print('')
    print('Incrementing g with complex k from {} up to {}'.format(g1s[0], g1))
    for i, g in enumerate(g1s):
        log('g = {}'.format(g))
        prev_vars = vars
        sol = root(rgEqs, vars, args=(kc, g, dims),
                   method='lm', jac=rg_jac, options=lmd)
        vars = sol.x
        er = np.abs(rgEqs(vars, kc, g, dims))
        tries = 0
        while np.max(er) > 10**-7:
            tries += 1
            if tries > 10**3:
                log('Stopping')
                return
            log('{}th try, g = {}'.format(tries, g))
            log('Failed: {}'.format(sol.message))
            log('Error: {}'.format(np.max(er)))
            log('Retrying with new vars:')
            vars = prev_vars + 2*imscale_v*(np.random.rand(len(vars))-0.5)
            es, ws = unpack_vars(vars, Ne, Nw)
            log(es)
            log(ws)
            sol = root(rgEqs, vars, args=(kc, g, dims),
                       method='lm', jac=rg_jac, options=lmd)
            vars = sol.x
            er = np.abs(rgEqs(vars, kc, g, dims))

        vars = sol.x
        er = np.abs(rgEqs(vars, kc, g, dims))
        if np.max(er) > 10**-9:
            log('Highish errors:')
            log('g = {}'.format(g))
            log(np.max(er))
        if np.max(er) > 0.001 and i > 1:
            print('This is too bad')
            return
        ces, cws = unpack_vars(vars, Ne, Nw)
        # if i == 0:
        #     log('Status: {}'.format(sol.status))
        #     log('Msg: {}'.format(sol.message))
        #     log('Iterations: {}'.format(sol.nfev))
        #     # log('Error (according to solver): {}'.format(sol.maxcv))
        #     log('g = {}'.format(g))
        #     log('er: {}'.format(np.max(er)))
        #     log('k:')
        #     log(k + 1j*kim)
        #     log('es:')
        #     log(ces)
        #     log('omegas:')
        #     log(cws)
        #     input('Enter to continue')
    print('')
    print('Incrementing k to be real')
    vars, er = increment_im_k(vars, dims, g, k, kim, sf=0.99)
    print('')
    kc = np.concatenate((k, 0.01*kim))
    print('Now doing the rest of g steps')
    for i, g in enumerate(g2s):
        sol = root(rgEqs, vars, args=(kc, g, dims),
                   method='lm', jac=rg_jac, options=lmd)


        vars = sol.x
        er = np.abs(rgEqs(vars, kc, g, dims))
        if np.max(er) > 10**-9:
            log('Highish errors:')
            log('g = {}'.format(g))
            log(np.max(er))
        if np.max(er) > 0.001:
            print('This is too bad')
            return
    print('')
    print('Removing the last bit of imaginary stuff')
    vars, er = increment_im_k(vars, dims, g, k, 0.01*kim, steps=10, sf=1)


    ces, cws = unpack_vars(vars, Ne, Nw)
    print('')
    print('This should be about zero (final error):')
    print(np.max(er))
    print('')
    return ces, cws


def ioms(es, g, ks, Zf=rationalZ, extra_bits=False):
    L = len(ks)
    R = np.zeros(L, dtype=np.complex128)
    for i, k in enumerate(ks):
        Zke = Zf(k, es)
        R[i] = g*np.sum(Zke)
        if extra_bits:
            otherks = ks[np.arange(L) != i]
            Zkk = Zf(k, otherks)
            R[i] += -1*g*np.sum(Zkk) + 1.0
    return R

if __name__ == '__main__':

    r = rg_jac(np.arange(1,9), np.arange(1,9)/8, -10, (4, 2, 2))
    print('First row of Jacobian: ')
    print(r[0])
    print('First column of Jacobian: ')
    print(r[:, 0])
    print('Diagonal entries of the Jacobian: ')
    print(np.diagonal(r))

    print('Checking Zs')
    print(rationalZ(3+4j, 2-8j))
    print(reZ(3,4,2,-8))
    print(imZ(3,4,2,-8))

    L = int(input('Length: '))
    Ne = int(input('Nup: '))
    Nw = int(input('Ndown: '))
    gf = float(input('G: '))

    # dg = float(input('dg: '))
    # imk = float(input('Scale of imaginary part for k: '))
    # imv = float(input('Same for variable guess: '))

    # dg = 0.001*8/L
    dg = 0.01
    g0 = .1*dg
    # imk = .5*dg/(Ne+Nw)
    imk = g0
    imv = imk


    dims = (L, Ne, Nw)

    ks = (1.0*np.arange(L) + 1.0)/L
    es, ws = solve_rgEqs(dims, gf, ks, dg=dg, g0=g0, imscale_k=imk, imscale_v=imv)
    print('')
    print('Solution found:')
    print('e_alpha:')
    for e in es:
        print('{} + I*{}'.format(float(np.real(e)), np.imag(e)))
        print('')
    print('omega_beta')
    for e in ws:
        print('{} + I*{}'.format(float(np.real(e)), np.imag(e)))
        print('')
    rk = ioms(es, gf, ks)
    print('From RG, iom eigenvalues:')
    for r in rk:
        print(r)

    print('From RG, energy is:')
    print(np.sum(ks*rk))
    rge = np.sum(ks*rk)

    if L < 6:
        from exact_qs_so5 import iom_dict, form_basis, ham_op, find_min_ev
        from quspin.operators import quantum_operator
        basis = form_basis(2*L, Ne, Nw)

        ho = ham_op(L, gf, ks, basis)
        e, v = find_min_ev(ho, L, basis, n=100)
        print('Energy found:')
        print(rge)
        print('Smallest distance from ED result for GS energy:')
        diffs = np.abs(e-rge)
        print(np.min(diffs))
        print('This is the {}th energy'.format(np.argmin(diffs)))
        print('True low energies:')
        print(e[:10])
