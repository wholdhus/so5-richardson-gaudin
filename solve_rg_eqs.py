import numpy as np
from scipy.optimize import root

VERBOSE=True

def log(msg):
    if VERBOSE:
        print(msg)

def rationalZ(x, y):
    return x*y/(x-y)


def trigZ(x, y):
    return np.cot(x-y)


def rationalZF(x, y):
    return x/(x-y)


def rgEqs(vars, Ne, Nw, etas, g, Zf, cN=1, cZ=0):
    L = len(etas)//2
    etas = etas[:L] + 1j*etas[L:]
    rvars = vars[:len(vars)//2]
    imvars = vars[len(vars)//2:]
    es = rvars[:Ne] + 1j*imvars[:Ne]
    ws = rvars[Ne:] + 1j*rvars[Ne:]
    set1 = np.zeros(Ne, dtype=complex)
    set2 = np.zeros(Nw, dtype=complex)
    for i, e in enumerate(es):
        set1[i] = (g*(2*(Zf(es[es!=e], e)).sum() - Zf(ws, e).sum()
                   - Zf(etas, e).sum()) - (cN - cZ))
    for i, w in enumerate(ws):
        set2[i] = (g*(Zf(ws[ws!=w],w).sum() - Zf(es, w).sum()) - cZ)
    reqs = np.concatenate((set1.real, set2.real))
    imqs = np.concatenate((set1.imag, set2.imag))
    return np.concatenate((reqs, imqs))


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
        eqs[i] = (
                2*Zf(ces[ces!=e], e).sum()
                - Zf(cws, e).sum() - Zf(cetas, e).sum()) + 1./g
    return np.concatenate((eqs.real, eqs.imag))


def rgEqs2(ws, es, etas):
    Zf = rationalZ
    Ne = len(es)//2
    Nw = len(ws)//2
    ces = es[:Ne] + 1j*es[Ne:]
    cws = ws[:Nw] + 1j*ws[Nw:]
    eqs = np.zeros(Nw, dtype=complex)
    for i, w in enumerate(cws):
        eqs[i] = Zf(cws[cws!=w], w).sum() - Zf(ces, w).sum()
    return np.concatenate((eqs.real, eqs.imag))


def solve_rgEqs(L, Ne, Nw, gf):
    k = np.array(
                [(2*i+1)*np.pi/L for i in range(L)],
                dtype=np.complex128)

    gs = np.linspace(0, 1, 10)*gf
    dg = gf/(1.*len(gs))

    ces = k[:Ne] + gs[1]*.00001*np.cos(np.pi*np.arange(Ne))
    # ces = ces - .05 * np.cos(np.pi*np.arange(Ne))
    # cws = k[:Nw] - k[0] # not quite what the exact zero coupling solution is, but this seems to be ok
    cws = np.concatenate(([0.], k[[2*(i+1) for i in range(Nw-1)]]/2.0))
    print('Initial guesses:')
    print(ces)
    print(cws)

    es = np.concatenate((ces.real, ces.imag))
    ws = np.concatenate((cws.real, cws.imag))

    print('Incrementing g with complex k')
    kim = 1j*np.cos(np.pi*np.arange(L))*0
    ceta = k + kim
    eta = np.concatenate((ceta.real, ceta.imag))
    eprev = es
    wprev = ws
    for i, g in enumerate(gs[1:]):
        log(g)
        wsol = root(rgEqs2, ws, args=(es, eta), method='lm')
        ws = wsol.x
        esol = root(rgEqs1, es, args=(ws, eta, g), method='lm')
        es = esol.x

        e1 = np.max(np.abs(rgEqs1(es, ws, eta, g)))
        e2 = np.max(np.abs(rgEqs2(ws, es, eta)))
        if e1 > 10**-12 or e2 > 10**-12:
            log('Highish errors:')
            log('g = {}'.format(g))
            log(e1)
            log(e2)

    print('')
    print('E_alpha:')
    print(es[:Ne]+1j*es[Ne:])
    print('omega_beta:')
    print(ws[:Nw]+1j*ws[Nw:])
    print('')
    print('Incrementing k to be real')
    scale = 1 - np.linspace(0, 1, 10)
    for i, s in enumerate(scale):
        ceta = k + s*kim
        eta = np.concatenate((ceta.real, ceta.imag))

        esol = root(rgEqs1, es, args=(ws, eta, gf), method='lm')
        es = esol.x

        wsol = root(rgEqs2, ws, args=(es, eta), method='lm')
        ws = wsol.x

        e1 = np.max(np.abs(rgEqs1(es, ws, eta, gf)))
        e2 = np.max(np.abs(rgEqs2(ws, es, eta)))
        if max(e1, e2) > 10**-10:
            log('Highish errors:')
            log('s = {}'.format(s))
            log(e1)
            log(e2)

    print('This should be about zero:')
    print(np.max(np.abs(rgEqs1(es, ws, eta, g))))
    print('Same with this:')
    print(np.max(np.abs(rgEqs2(ws, es, eta))))
    print('E_alpha:')
    print(es[:Ne]+1j*es[Ne:])
    print('omega_beta:')
    print(ws[:Nw]+1j*ws[Nw:])


if __name__ == '__main__':
    L = 4
    Ne = 2
    Nw = 2
    gf = -.01
    solve_rgEqs(L, Ne, Nw, gf)
