import numpy as np
from scipy.optimize import root

def rationalZ(x, y):
    return x*y/(x-y)


def trigZ(x, y):
    return np.cot(x-y)


def rgEqs(vars, Ne, Nw, etas, g, Zf, cN=1, cZ=0):
    L = len(etas)//2
    etas = etas[:L] + 1j*etas[L:]
    rvars = vars[:len(vars)//2]
    imvars = vars[len(vars)//2:]
    es = rvars[:Ne] + 1j*imvars[:Ne]
    ws = rvars[Ne:] + 1j*rvars[Ne:]
    set1 = np.zeros(len(es), dtype=complex)
    set2 = np.zeros(len(ws), dtype=complex)
    for i, e in enumerate(es):
        set1[i] = (g*(2*(Zf(es[es!=e], e)).sum() - Zf(ws, e).sum()
                   - Zf(etas, e).sum()) - (cN - cZ))
    for i, w in enumerate(ws):
        set2[i] = (g*(Zf(ws[ws!=w],w).sum() - Zf(es, w).sum()) - cZ)
    reqs = np.concatenate((set1.real, set2.real))
    imqs = np.concatenate((set1.imag, set2.imag))
    return np.concatenate((reqs, imqs))


def solve_rgEqs(L, Ne, Nw, gf, Zf=rationalZ):
    k = np.array(
                [(2*i+1)*np.pi/L for i in range(L)],
                dtype=np.complex128)

    gs = np.linspace(0, 1, 1000)*gf
    es = k[:Ne]*1j*0.1*np.random.rand(Ne)
    ws = k[:Nw]*1j*0.1*np.random.rand(Nw)
    vars = np.concatenate((np.concatenate((es.real, ws.real)),
                           np.concatenate((es.imag, ws.imag))))
    kim = 1j*np.cos(np.pi*np.arange(L))
    print('Incrementing g with complex k')
    ceta = k + kim
    eta = np.concatenate((ceta.real, ceta.imag))
    for i, g in enumerate(gs[1:]):
        print('g = {}'.format(g))
        sol = root(rgEqs, vars, args=(Ne, Nw, eta, g, Zf),
                   method='lm')
        vars = sol.x
        print('This should be about zero')
        print(np.max(np.abs(rgEqs(vars, Ne, Nw, eta, g, Zf))))
    print('')
    print('Incrementing k to be real')
    scale = 1 - np.linspace(0, 1, 100)
    for i, s in enumerate(scale):
        print('s= {}'.format(s))
        ceta = k + s*kim
        eta = np.concatenate((ceta.real, ceta.imag))
        sol = root(rgEqs, vars, args=(Ne, Nw, eta, gf, Zf),
                   method='lm')
        vars = sol.x
    print('This should also be about zero')
    sol = root(rgEqs, vars, args=(Ne, Nw, np.concatenate((k, np.zeros(L))),
                                  gf, Zf),
               method='lm')
    vars = sol.x
    print(np.max(np.abs(rgEqs(vars, Ne, Nw, eta, gf, Zf))))
    print('max imag part:')
    rvars = vars[:len(vars)//2]
    imvars = vars[len(vars)//2:]
    es = rvars[:Ne] + 1j*imvars[:Ne]
    ws = rvars[Ne:] + 1j*rvars[Ne:]
    print('Sum of im parts:')
    print(np.sum(imvars))


if __name__ == '__main__':
    L = 6
    Ne = 2
    Nw = 2
    gf = -0.5
    solve_rgEqs(L, Ne, Nw, gf)
