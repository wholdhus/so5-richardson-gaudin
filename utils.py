VERBOSE = True

import numpy as np

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
    if len(vars) != 2*(Ne+Nw):
        print('Cannot unpack variables! Wrong length!')
        return
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


def G_to_g(G, k):
    return G/(1-G*np.sum(k))


def g_to_G(g, k):
    return g/(1+g*np.sum(k))


"""
Richardson-Gaudin equations and Jacobian (derivative w.r.t. pairons)
"""

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


def rgEqs_q(vars, k, q, dims): # take q = 1/g instead
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
                      - reZ(kr, ki, er, ei).sum())/q
                   + c1)
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
    return q*eqs


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
