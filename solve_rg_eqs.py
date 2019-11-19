import numpy as np
from scipy.optimize import root

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
    ces = es[:Ne] + 1j*es[Ne:]
    cws = ws[:Nw] + 1j*ws[Nw:]
    eqs = np.zeros(Ne, dtype=complex)
    for i, e in enumerate(ces):
        eqs[i] = (g*(2*Zf(ces[ces!=e], e).sum() - Zf(cws, e).sum()
                   - Zf(etas, e).sum()) - 1.)
    return np.concatenate((eqs.real, eqs.imag))


def rgEqs2(ws, es, etas, g):
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

    gs = np.linspace(0, 1, 100)*gf
    print(gs)

    kim = 1j*np.cos(np.pi*np.arange(L))
    ceta = k + kim

    ces = k[:Ne]
    cws = k[:Nw] - k[0] # not quite what the exact zero coupling solution is, but this seems to be ok
    # cws = np.concatenate(([0.], k[[2*(i+1) for i in range(Nw-1)]]/2.0))
    print('Initial guesses:')
    print(ces)
    print(cws)
    # cws = .1*np.exp(1j*np.random.rand(Nw))
    # vars = np.concatenate((np.concatenate((es.real, ws.real)),
    #                        np.concatenate((es.imag, ws.imag))))
    es = np.concatenate((ces.real, ces.imag))
    ws = np.concatenate((cws.real, cws.imag))

    print('Incrementing g with complex k')
    eta = np.concatenate((ceta.real, ceta.imag))
    for i, g in enumerate(gs[1:]):
        # sol = root(rgEqs, vars, args=(Ne, Nw, eta, g, Zf),
        #            method='lm')
        wsol = root(rgEqs2, ws, args=(es, eta, g), method='lm')
        ws = wsol.x
        esol = root(rgEqs1, es, args=(ws, eta, g), method='lm')
        es = esol.x

        e1 = np.max(np.abs(rgEqs1(es, ws, eta, g)))
        e2 = np.max(np.abs(rgEqs2(ws, es, eta, g)))
        if e1 > 10**-12 or e2 > 10**-12:
            print('Highish errors:')
            print('g = {}'.format(g))
            print(e1)
            print(e2)
    print('')
    print('E_alpha:')
    print(es[:Ne]+1j*es[Ne:])
    print('omega_beta:')
    print(ws[:Nw]+1j*ws[Nw:])
    print('')
    print('Errors at this point:')
    print(rgEqs1(es, ws, eta, g))
    print(rgEqs2(es, ws, eta, g))
    print('Incrementing k to be real')
    scale = 1 - np.linspace(0, 1, 10)
    for i, s in enumerate(scale):
        ceta = k + s*kim
        eta = np.concatenate((ceta.real, ceta.imag))

        wsol = root(rgEqs2, ws, args=(es, eta, g), method='lm')
        ws = wsol.x

        esol = root(rgEqs1, es, args=(ws, eta, g), method='lm')
        es = esol.x
        e1 = np.max(np.abs(rgEqs1(es, ws, eta, g)))
        e2 = np.max(np.abs(rgEqs2(ws, es, eta, g)))
        if max(e1, e2) > 10**-10:
            print('Highish errors:')
            print('s = {}'.format(s))
            print(e1)
            print(e2)



    print('This should be about zero:')
    print(np.max(np.abs(rgEqs1(es, ws, eta, g))))
    print('Same with this:')
    print(np.max(np.abs(rgEqs2(ws, es, eta, g))))
    print('E_alpha:')
    print(es[:Ne]+1j*es[Ne:])
    print('omega_beta:')
    print(ws[:Nw]+1j*ws[Nw:])



if __name__ == '__main__':
    L = 4
    Ne = 2
    Nw = 2
    gf = -1.
    solve_rgEqs(L, Ne, Nw, gf)
