import numpy as np
from scipy.optimize import root, brenth

VERBOSE=True

def log(msg):
    if VERBOSE:
        print(msg)

def rationalZ(x, y):
    return x*y/(x-y)


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


def rgEqs(vars, etas, Ne, Nw, g, c1=1.0):

    Zf = rationalZ
    L = len(etas)//2
    cetas = etas[:L] + 1j*etas[L:]
    ces, cws = unpack_vars(vars, Ne, Nw)

    set1 = np.zeros(Ne, dtype=np.complex128)
    set2 = np.zeros(Nw, dtype=np.complex128)

    for i, e in enumerate(ces):
        # If I leave out a factor of 2 this solves nicely?
        # Mathematica seems to suggest the factor of 2 was an error?
        set1[i] = (g*(2*Zf(ces[np.arange(Ne) != i], e).sum()
                      - Zf(cws, e).sum() - Zf(cetas, e).sum())
                   + c1)

    for i, w in enumerate(cws):
        set2[i] = Zf(cws[np.arange(Nw) != i], w).sum() - Zf(ces, w).sum()

    eqs = np.concatenate((set1, set2))
    return np.concatenate((eqs.real, eqs.imag))


def rgEqs1(es, ws, etas, g):
    Zf = rationalZ
    Ne = len(es)//2
    Nw = len(ws)//2
    L = len(etas)//2
    cetas = etas[:L] + 1j*etas[L:]
    ces = es[:Ne] + 1j*es[Ne:]
    cws = ws[:Nw] + 1j*ws[Nw:]
    eqs = np.zeros(Ne, dtype=complex)
    for i, e in enumerate(ces):
        eqs[i] = g*(
                2*Zf(ces[np.arange(Ne) != i], e).sum()
                - Zf(cws, e).sum() - Zf(cetas, e).sum()) + 1.
    return np.concatenate((eqs.real, eqs.imag))


def rgEqs2(ws, es):
    Zf = rationalZ
    Ne = len(es)//2
    Nw = len(ws)//2
    ces = es[:Ne] + 1j*es[Ne:]
    cws = ws[:Nw] + 1j*ws[Nw:]
    eqs = np.zeros(Nw, dtype=complex)
    for i, w in enumerate(cws):
        eqs[i] = (Zf(cws[np.arange(Nw) != i], w).sum()
                  - Zf(ces, w).sum())
    return np.concatenate((eqs.real, eqs.imag))


def extra_eq(w, es):
    eq = rationalZ(es, w).sum()
    return eq


def find_extra_w(es, epsilon=10**-6):
    below = np.min(es) - epsilon
    above = np.min(es) + epsilon
    w = brenth(extra_eq, below, above, args=(es))
    return w


def g0_guess(L, Ne, Nw, k, imscale=0.01, double=True):
    if double:
        ces = np.array([k[i//2] for i in range(Ne)], dtype=np.complex128)
        cws = np.array([k[i//2] for i in range(Nw)], dtype=np.complex128)
        # cws = np.zeros(Nw, dtype=np.complex128)
        ces += 1.8j*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Ne)])
        cws += -3.2j*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Nw)])
        #ces += 0.002*imscale*(-1)**np.arange(Ne)
        # cws += 0.2*imscale
    else:
        ces = np.array(k[:Ne], dtype=np.complex128)
        # cws = (np.array(k[:Ne], dtype=np.complex128) - 0.25*k[0])/2
        cws = np.zeros(Nw, dtype=np.complex128)
        cws += 1j*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Nw)])

    return ces, cws


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
        ceta = k + s*im_k
        eta = np.concatenate((ceta.real, ceta.imag))
        sol = root(rgEqs, vars, args=(eta, Ne, Nw, g),
                   method='lm')
        vars = sol.x
        er = np.abs(rgEqs(vars, eta, Ne, Nw, g))
        if np.max(er) > 10**-10:
            log('Highish errors:')
            log('s = {}'.format(s))
            log(np.max(er))
    return vars, er


def solve_rgEqs(dims, gf, k, dg=0.01, imscale_k=0.01, imscale_v=0.001):
    L, Ne, Nw = dims
    g1s = 0.01*4/L
    if gf > g1s*L:
        g1 = g1s*L
        g1s = np.arange(dg, g1, 0.5*dg)
        g2s = np.append(np.arange(g1, gf, dg), gf)
    elif gf < -1*g1s*L:
        g1 = -1*g1s*L
        g1s = -1*np.linspace(dg, -1*g1, dg)
        g2s = np.append(-1*np.arange(-1*g1, -1*gf, dg), gf)
    else:
        print('Woops: g too close to zero')
        return
    log('Paths for g:')
    log(g1s)
    log(g2s)
    print('')
    # imscale=0.1*dg
    kim = 1j*imscale_k*np.cos(np.pi*np.arange(L))
    # kim = np.zeros(L)
    ceta = k + kim
    eta = np.concatenate((ceta.real, ceta.imag))

    ces, cws = g0_guess(L, Ne, Nw, k, imscale=imscale_v)
    log('Initial guesses:')
    log(ces)
    log(cws)
    vars = pack_vars(ces, cws)
    log('Eqs with initial guess:')
    eq0 = rgEqs(vars, eta, Ne, Nw, dg)
    log(eq0[:len(eq0)//2] + 1j*eq0[len(eq0)//2:])
    print('')
    print('Incrementing g with complex k up to {}'.format(g1))
    for i, g in enumerate(g1s[1:]):
        sol = root(rgEqs, vars, args=(eta, Ne, Nw, g),
                   method='lm')
        vars = sol.x

        er = np.abs(rgEqs(vars, eta, Ne, Nw, g))
        if np.max(er) > 10**-9:
            log('Highish errors:')
            log('g = {}'.format(g))
            log(np.max(er))

    print('')
    print('Incrementing k to be real')
    vars, er = increment_im_k(vars, dims, g, k, kim, sf=0.99)
    print('')
    ceta = k + 0.01*kim
    eta = np.concatenate((ceta.real, ceta.imag))
    print('Now doing the rest of g steps')
    for i, g in enumerate(g2s):
        sol = root(rgEqs, vars, args=(eta, Ne, Nw, g),
                   method='lm')
        vars = sol.x
        er = np.abs(rgEqs(vars, eta, Ne, Nw, g))
        if np.max(er) > 10**-9:
            log('Highish errors:')
            log('g = {}'.format(g))
            log(np.max(er))
        if i > 10 and np.max(er) > 0.001:
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
    L = int(input('Length: '))
    Ne = int(input('Nup: '))
    Nw = int(input('Ndown: '))
    gf = float(input('G: '))
    dg = float(input('dg: '))
    imk = float(input('Scale of imaginary part for k: '))
    imv = float(input('Same for variable guess: '))

    # ks = np.array(
    #             [(2*i+1)*np.pi/L for i in range(L)])

    dims = (L, Ne, Nw)

    ks = (1.0*np.arange(L) + 1.0)/L
    es, ws = solve_rgEqs(dims, gf, ks, dg=dg, imscale_k=imk, imscale_v=imv)
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
        e, v = find_min_ev(ho, L, basis, n=10)
        print('Energy found:')
        print(rge)
        print('Smallest distance from ED result for GS energy:')
        diffs = np.abs(e-rge)
        print(np.min(diffs))
        print('This is the {}th energy'.format(np.argmin(diffs)))
        print('True low energies:')
        print(e[:10])
