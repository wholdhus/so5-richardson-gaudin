import numpy as np
from scipy.optimize import root, brenth

VERBOSE=True

def log(msg):
    if VERBOSE:
        print(msg)

def rationalZ(x, y):
    return x*y/(x-y)

def reZ(x,y,u,v): # Re(Z(z1,z2)), z1 = x+iy, x2 = u+iv
    return ((x*u-y*v)*(x-u)+(y*u+x*v)*(y-v))/((x-u)**2+(y-v)**2)

def imZ(x,y,u,v):
    return ((x*u-y*v)*(v-y)+(y*u+x*v)*(x-u))/((x-u)**2+(y-v)**2)

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


def rgEqs(vars, k, Ne, Nw, g, c1=1.0):

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
        set1_r[i] = (g*(2*reZ(ers[js], eis[js], er, ei).sum()
                      - reZ(wrs, wis, er, ei).sum()
                      - reZ(kr, ki, er, ei).sum())
                   + c1)
        set1_i[i] = (g*(2*imZ(ers[js], eis[js], er, ei).sum()
                      - imZ(wrs, wis, er, ei).sum()
                      - imZ(kr, ki, er, ei).sum()))
    for i, wr in enumerate(wrs):
        wi = wis[i]
        js = np.arange(Nw) != i
        set2_r[i] = g*(reZ(wrs[js], wis[js], wr, wi).sum()
                     - reZ(ers, eis, wr, wi).sum()
                    )
        set2_i[i] = g*(imZ(wrs[js], wis[js], wr, wi).sum()
                     - imZ(ers, eis, wr, wi).sum()
                    )
    eqs = np.concatenate((set1_r, set1_i, set2_r, set2_i))
    return eqs

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

    print('Checking Zs')
    print(rationalZ(3+4j, 2-8j))
    print(reZ(3,4,2,-8))
    print(imZ(3,4,2,-8))

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
        e, v = find_min_ev(ho, L, basis, n=100)
        print('Energy found:')
        print(rge)
        print('Smallest distance from ED result for GS energy:')
        diffs = np.abs(e-rge)
        print(np.min(diffs))
        print('This is the {}th energy'.format(np.argmin(diffs)))
        print('True low energies:')
        print(e[:10])
