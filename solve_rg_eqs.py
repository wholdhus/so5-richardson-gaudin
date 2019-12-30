import numpy as np
from scipy.optimize import root

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
        set1[i] = (g*(Zf(ces[np.arange(Ne) != i], e).sum()
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


def g0_guess(L, Ne, Nw, k, imscale=0.01, double=True):
    if double:
        ces = np.array([k[i//2] for i in range(Ne)], dtype=np.complex128)
        # cws = np.array([k[i//2] - k[0] for i in range(Nw)], dtype=np.complex128)
        cws = np.zeros(Nw, dtype=np.complex128)
        ces += 1j*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Ne)])
        cws += 1j*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Nw)])
    else:
        ces = np.array(k[:Ne], dtype=np.complex128)
        cws = (np.array(k[:Ne], dtype=np.complex128) - k[0])/2
        # cws = np.zeros(Nw, dtype=np.complex128)
    return ces, cws


def dvars(vars, pvars, dg, Ne, Nw):
    ces, cws = unpack_vars(vars, Ne, Nw)
    pes, pws = unpack_vars(pvars, Ne, Nw)
    de = (ces-pes)/dg
    dw = (cws-pws)/dg
    deriv = pack_vars(de, dw)

    return deriv


def solve_rgEqs(L, Ne, Nw, gf, k):

    gs = np.linspace(0, 1, 100)*gf

    dg = np.abs(gf/100)

    kim = .1j*np.cos(np.pi*np.arange(L))
    # kim = np.zeros(L)
    ceta = k + kim
    eta = np.concatenate((ceta.real, ceta.imag))

    ces, cws = g0_guess(L, Ne, Nw, k, double=False)
    log('Initial guesses:')
    log(ces)
    log(cws)

    #ws = np.concatenate((cws.real, cws.imag))
    #es = np.concatenate((ces.real, ces.imag))
    #log('Refining initial guesses')
    #sole = root(rgEqs1, es, args=(ws, eta, gs[1]))
    #es = sole.x
    #ces = es[:Ne] + 1j*es[Ne:]
    #log('New es:')
    #log(ces)

    vars = pack_vars(ces, cws)
    # last = vars
    log('Eqs with initial guess:')
    eq0 = rgEqs(vars, eta, Ne, Nw, gs[1])
    log(eq0[:len(eq0)//2] + 1j*eq0[len(eq0)//2:])

    print('Incrementing g with complex k')
    for i, g in enumerate(gs[1:]):
        sol = root(rgEqs, vars, args=(eta, Ne, Nw, g),
                   method='lm')
        vars = sol.x
        ces, cws = unpack_vars(vars, Ne, Nw)
        print('e_alpha at g = {}'.format(g))
        print(ces)
        print('')


        er = np.abs(rgEqs(vars, eta, Ne, Nw, g))
        if np.max(er) > 10**-12:
            print('Highish errors:')
            print('g = {}'.format(g))
            print(np.max(er))
        # Now adding newton method correction
        # deriv = dvars(vars, last, dg, Ne, Nw)
        # print(np.max(np.abs(deriv)))
        # last = vars
        # vars += deriv
    ces, cws = unpack_vars(vars, Ne, Nw)
    log('')
    log('E_alpha:')
    log(ces)
    log('omega_beta:')
    log(cws)
    print('')
    print('Incrementing k to be real')
    scale = 1 - np.linspace(0, 1, 10)
    for i, s in enumerate(scale):
        ceta = k + s*kim
        eta = np.concatenate((ceta.real, ceta.imag))
        sol = root(rgEqs, vars, args=(eta, Ne, Nw, g),
                   method='lm')
        vars = sol.x
        er = np.abs(rgEqs(vars, eta, Ne, Nw, g))
        if np.max(er) > 10**-12:
            log('Highish errors:')
            log('s = {}'.format(s))
            log(np.max(er))

    log('Final k values')
    print(ceta)
    ces, cws = unpack_vars(vars, Ne, Nw)

    print('This should be about zero:')
    print(np.max(er))
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
    L = 4
    Ne = 2
    Nw = 2
    gf = -0.5

    from exact_qs_so5 import iom_dict, form_basis
    from quspin.operators import quantum_operator
    basis = form_basis(2*L, Ne, Nw)


    #ks = np.array(
    #            [(2*i+1)*np.pi/L for i in range(L)],
    #            dtype=np.complex128)
    ks = 1.0*np.arange(L) + 1.0
    print('Ks:')
    print(ks)
    es, ws = solve_rgEqs(L, Ne, Nw, gf, ks)
    print('Solution found:')
    print('e_alpha:')
    print(es)
    print('omega_beta')
    print(ws)
    print('')
    rk = ioms(es, gf, ks)
    print('From RG, iom eigenvalues:')
    for r in rk:
        print(r)

    if L < 6:

        iomd = iom_dict(L, Ne, Nw, gf, ks, k1=0)
        Rk = quantum_operator(iomd, basis=basis)
        print('Checking first iom')
        es, v = Rk.eigh()
        print('Smallest distance from ED result:')
        print(np.min(np.abs(rk[0]-es)))
        print(np.min(np.abs(rk[0]+es)))
