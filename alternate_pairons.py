from exact_qs_so5 import iom_dict, form_basis, find_min_ev
import numpy as np
from quspin.operators import quantum_operator
from scipy.optimize import root

def Z(x, y):
    return x*y/(x-y)


def pairon_eqs(es, L, Ne, G, ks, rs):
    ces = es[:Ne] + 1j*es[Ne:]
    pairon_eqs = np.zeros(L, dtype=np.complex128)
    for i, k in enumerate(ks):
        pairon_eqs[i] = G*np.sum(Z(k, ces)) - rs[i]
    np.random.shuffle(pairon_eqs) # Does this help?
    some = pairon_eqs[:Ne]
    out = np.concatenate((some.real, some.imag))
    # out = np.concatenate((pairon_eqs.real, pairon_eqs.imag))
    # print(out)
    return out


def wairon_eqs(ws, ces, L, Nw):
    cws = ws[:Nw] + 1j*ws[Nw:]
    wairon_eqs = np.zeros(Nw, dtype=np.complex128)
    for i, w in enumerate(cws):
        otherws = cws[np.arange(Nw) != i]
        wairon_eqs[i] = Z(otherws, w).sum()
        wairon_eqs[i] += - Z(ces, w).sum()
    out = np.concatenate((wairon_eqs.real, wairon_eqs.imag))
    return out


def find_pairons(L, Nup, Ndown, G, ks, basis, trysparse=True):
    rs = np.zeros(L)
    for i in range(L):
        rd = iom_dict(L, G, ks, k1=i)
        r = quantum_operator(rd, basis=basis)
        rs[i], _ = find_min_ev(r, L, basis)
    guess = np.array(np.random.rand(Nup), dtype=np.complex128)
    rguess = np.concatenate((guess.real, guess.imag))

    sol = root(pairon_eqs, rguess, args=(L, Nup, G, ks, rs))
    es = sol.x[:Nup] + 1j*sol.x[Nup:]
    er = pairon_eqs(sol.x, L, Nup, G, ks, rs)
    print('Pairons found with maximum error:')
    print(np.max(np.abs(er)))

    # rguess2 = np.concatenate((ks[:Ndown], np.zeros(Ndown)))
    # sol2 = root(wairon_eqs, rguess2, args=(es, L, Ndown))
    # ws = sol2.x[:Ndown] + 1j*sol2.x[Ndown:]
    # er = wairon_eqs(sol.x, es, L, Ndown)
    # print('Wairons found with maximum error:')
    # print(np.max(np.abs(er)))
    ws = None
    return es, ws, rs

if __name__ == '__main__':
    L = 4
    Nup = 2
    Ndown = 2
    G = 0.5
    ks = 1.0 + np.arange(L)
    basis = form_basis(2*L, Nup, Ndown)
    es, ws, rs = find_pairons(L, Nup, Ndown, G, ks, basis)
    print('Pairons:')
    print(es)
