import numpy as np
from scipy.optimize import root

VERBOSE=True
TOL=10**-12
MAXIT=1000

def log(msg):
    if VERBOSE:
        print(msg)


def rationalZ(x, y):
    return 1/(x-y)

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

def unpack_vars(vars, N):
    return vars[:N] + 1j*vars[N:]


def pack_vars(ces):
    # Takes separate, complex vectors
    # Combines them into variable vector format
    es = np.concatenate((ces.real, ces.imag))
    return es

def rgEqs(vars, eta, g, dims):

    c1 = 1
    L, N = dims
    Zf = rationalZ
    eqs_c =  np.zeros(N, dtype=np.complex128)
    es = unpack_vars(vars, N)
    for i, e in enumerate(es):
        js = np.arange(N) != i
        eqs_c[i] = ((2*Zf(es[js], e).sum()
                      - Zf(eta, e).sum())
                   + c1/g)

    eqs = np.concatenate((np.real(eqs_c), np.imag(eqs_c)))

    return eqs


def rg_jac(vars, eta, g, dims):
    """
    ASSUMING NE = NW!

    f1 is function on RHS of first set of equations
    f2 is function on RHS of second set of equations
    """
    L, N = dims
    jac = np.zeros((len(vars), len(vars)))

    e_r = vars[:N]
    e_i = vars[N:2*N]

    eta_r = eta[:L]
    eta_i = eta[L:]

    for i in range(N):
        for j in range(N):
            if i == j:
                ls = np.arange(N) != j
                # Re(f1), Re(e)
                jac[i, j] = (2*np.sum(dZ_rr(e_r[ls], e_i[ls], e_r[i], e_i[i]))
                               -np.sum(dZ_rr(eta_r, eta_i, e_r[i], e_i[i]))
                               )
                # Re(f1), Im(e)
                jac[i, j+N] = (2*np.sum(dZ_ri(e_r[ls], e_i[ls], e_r[i], e_i[i]))
                                  -np.sum(dZ_ri(eta_r, eta_i, e_r[i], e_i[i]))
                                  )
                # Im(f1), Re(e)
                # -1 * previous by prope_rties of Z!
                jac[i+N, j] = -1*jac[i, j+N]
                # Im(f1), Im(e)
                # same as dRe(f)/dRe(e)
                jac[i+N, j+N] = jac[i, j]

            else: # i != j
                """
                For the following, there is a factor of -1 and
                index order is switched in the calls to the
                derivative functions, because dZ(x,y)/dx = - dZ(y,x)/dx
                and the derivative functions are calculated w.r.t. the
                real and imaginary parts of the second variable
                """
                # Re(f1), Re(e)
                jac[i, j] = -2*dZ_rr(e_r[i], e_i[i], e_r[j], e_i[j])
                # Re(f1), Im(e)
                jac[i, j+N] = -2*dZ_ri(e_r[i], e_i[i], e_r[j], e_i[j])
                # Im(f1), Re(e)
                jac[i+N, j] = -1*jac[i, j+N]
                # Im(f1), Im(e)
                jac[i+N, j+N] = jac[i, j]

    return jac/g


def g0_guess(dims, eta_r, imscale=0.01):
    L, N = dims
    e_r = eta_r[:N] - .001*imscale
    # e_i = imscale*np.random.rand(N)
    # e_i = imscale*np.ones(N)
    e_i = np.zeros(N)
    vars = np.concatenate((e_r, e_i))
    return vars


def increment_im_eta(vars, dims, g, eta_r, eta_i, steps=100, sf=1):
    L, N = dims
    scale = 1 - np.linspace(0, sf, steps)
    for i, s in enumerate(scale):

        eta = np.concatenate((eta_r, s*eta_i))
        sol = root(rgEqs, vars, args=(eta, g, dims),
                   method='lm', jac=rg_jac,
                   options={# 'maxiter': MAXIT,
                            'xtol': TOL})
        vars = sol.x
        er = np.abs(rgEqs(vars, eta, g, dims))
        if np.max(er) > 10**-10:
            log('Highish errors:')
            log('s = {}'.format(s))
            log(np.max(er))
        if np.max(er) > 0.001:
            print('This is too bad')
            return
    return vars, er


def solve_rgEqs(dims, gf, eta_r, dg=0.01, imscale_eta=0.01, imscale_v=0.001):

    L, N = dims
    # g1sc = 0.01*4/L
    g1 = 0.04
    if gf > g1:
        g1s = np.arange(2*dg, g1, dg)
        g2s = np.append(np.arange(g1, gf, dg), gf)
    elif gf < -1*g1:
        g1 *= -1
        g1s = -1*np.linspace(.1*dg, -1*g1, dg)
        g2s = np.append(-1*np.arange(-1*g1, -1*gf, dg), gf)
    else:
        print('Woops: g too close to zero')
        return
    log('Paths for g:')
    log(g1s)
    log(g2s)
    print('')

    eta_i = imscale_eta*(np.random.rand(L)-0.5)
    eta = np.concatenate((eta_r, eta_i))
    print(eta)

    vars = g0_guess(dims, eta, imscale=imscale_v)
    log('Initial guesses:')
    es = unpack_vars(vars, N)
    print(es)
    print('')
    print('Incrementing g with complex k from {} up to {}'.format(g1s[0], g1))
    for i, g in enumerate(g1s):
        log(g)
        sol = root(rgEqs, vars, args=(eta, g, dims),
                   method='lm', jac=rg_jac,
                   options={# # 'maxiter': MAXIT,
                            'xtol': TOL})
        vars = sol.x

        er = np.abs(rgEqs(vars, eta, g, dims))
        if np.max(er) > 10**-9:
            log('Highish errors:')
            log('g = {}'.format(g))
            log(np.max(er))
        if np.max(er) > 0.001 and i > 3:
            print('This is too bad')
            return

    log('Current solution:')
    log(unpack_vars(vars, N))
    print('')
    print('Incrementing k to be real')
    vars, er = increment_im_eta(vars, dims, g, eta_r, eta_i, sf=0.99)
    print('')
    eta = np.concatenate((eta_r, 0.01*eta_i))
    log('Current solution:')
    log(unpack_vars(vars, N))
    print('Now doing the rest of g steps')
    for i, g in enumerate(g2s):
        sol = root(rgEqs, vars, args=(eta, g, dims),
                   method='lm', jac=rg_jac,
                   options={# 'maxiter': MAXIT,
                            'xtol': TOL})
        vars = sol.x
        er = np.abs(rgEqs(vars, eta, g, dims))
        if np.max(er) > 10**-9:
            log('Highish errors:')
            log('g = {}'.format(g))
            log(np.max(er))
        if np.max(er) > 0.001:
            print('This is too bad')
            return
    print('')
    print('Removing the last bit of imaginary stuff')
    vars, er = increment_im_eta(vars, dims, g, eta_r, 0.01*eta_i, steps=10, sf=1)


    ces = unpack_vars(vars, N)
    print('')
    print('This should be about zero (final error):')
    print(np.max(er))
    print('')
    return ces

def ioms(ces, g, eta_r, Zf=rationalZ, extra_bits=False):
    L = len(eta_r)
    R = np.zeros(L, dtype=np.complex128)
    for i, e in enumerate(eta_r):
        Zke = Zf(e, es)
        R[i] = g*np.sum(Zke)
        if extra_bits:
            others = eta_r[np.arange(L) != i]
            Zkk = Zf(eta_r, others)
            R[i] += -1*g*np.sum(Zkk) + 1.0
    return R

if __name__ == '__main__':

    L = int(input('Length: '))
    N = int(input('N: '))
    gf = float(input('G: '))

    # dg = float(input('dg: '))
    # imk = float(input('Scale of imaginary part for k: '))
    # imv = float(input('Same for variable guess: '))

    # dg = 0.001*8/L
    dg = 0.001
    imk = 5*dg
    imv = .1*dg
    # ks = np.array(
    #             [(2*i+1)*np.pi/L for i in range(L)])

    dims = (L, N)

    eta_r = ((1.0*np.arange(L) + 1.0)/L)
    log('gf: {}'.format(gf))
    es = solve_rgEqs(dims, gf, eta_r, dg=dg, imscale_eta=imk, imscale_v=imv)
    print('')
    print('Solution found:')
    print('e_alpha:')
    for e in es:
        print('{} + I*{}'.format(float(np.real(e)), np.imag(e)))
        print('')

    rk = ioms(es, gf, eta_r)
    print('From RG, iom eigenvalues:')
    for r in rk:
        print(r)

    print('From RG, energy is:')
    print(np.sum(eta_r*rk))
    rge = np.sum(eta_r*rk)
