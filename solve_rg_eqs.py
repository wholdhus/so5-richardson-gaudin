import numpy as np
import pandas
from scipy.optimize import root, minimize
from scipy.special import binom
from traceback import print_exc

import concurrent.futures
import multiprocessing


VERBOSE=True
FORCE_GS=True
TOL=10**-12
TOL2=10**-7 # there are plenty of spurious minima around 10**-5
MAXIT=0 # let's use the default value
FACTOR=100
CPUS = multiprocessing.cpu_count()
JOBS = int(0.25*CPUS)
MAX_STEPS = 100

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


def g0_guess(L, Ne, Nw, kc, g0, imscale=0.01):
    k_r = kc[:L]
    k_i = kc[L:]
    double_e = np.arange(Ne)//2
    double_w = np.arange(Nw)//2
    # Initial guesses from perturbation theory
    er = k_r[double_e]*(1-g0*k_r[double_e])
    wr = k_r[double_w]*(1-g0*k_r[double_w]/3)
    ei = k_i[double_e]*(1-g0*k_r[double_e])
    wi = k_i[double_w]*(1-g0*k_r[double_w]/3)
    # Also adding some noise (could find at next order in pert. theory?)
    ei += .07*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Ne)])
    wi += -3.2*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Nw)])

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


def root_thread_job(vars, kc, g, dims, force_gs):
    L, Ne, Nw = dims
    # log('Trying with vars = ')
    # log(vars)
    sol = root(rgEqs, vars, args=(kc, g, dims),
               method='lm', jac=rg_jac, options=lmd)
    vars = sol.x
    er = max(abs(rgEqs(vars, kc, g, dims)))
    es, ws = unpack_vars(vars, Ne, Nw)
    if force_gs:
        # Running checks to see if this is deviating from ground state solution
        min_w = np.min(np.abs(ws))
        k_cplx = kc[:L] + 1j*kc[L:]
        k_distance = np.max(np.abs(es - k_cplx[np.arange(Ne)//2]))
        if min_w < 0.5*kc[0] or k_distance > 10**-3:
            er = 1

    return sol, vars, er


def root_threads(prev_vars, noise_scale, kc, g, dims, force_gs,
                 noise_factors=None):
        L, Ne, Nw = dims
        with concurrent.futures.ProcessPoolExecutor(max_workers=CPUS) as executor:
            if np.abs(g*L) < 0.05:
                # imaginary part of e is extremely close to imaginary part of kc
                # at low g, so lets use tiny noise
                noises_e = np.concatenate((noise_scale*2*(np.random.rand(JOBS, Ne) - 0.5),
                                           noise_scale*2*(np.random.rand(JOBS, Ne) - 0.5)*10**-3), axis=1)
            else:
                noises_e = noise_scale * 2 * (np.random.rand(JOBS, 2*Ne) - 0.5)
            # Real and im parts of w are both around the same distance from kc
            noises_w = noise_scale * 0.5 * (np.random.rand(JOBS, 2*Nw) - 0.5)
            noises = np.concatenate((noises_e, noises_w), axis=1)
            if noise_factors is not None:
                noises = noises * noise_factors
            log('Noise ranges from {} to {}'.format(np.min(noises), np.max(noises)))
            tries = [prev_vars + noises[n] for n in range(JOBS)]
            future_results = [executor.submit(root_thread_job,
                                              tries[n],
                                              kc, g, dims, force_gs)
                              for n in range(JOBS)]
            concurrent.futures.wait(future_results)
            for res in future_results:
                try:
                    yield res.result()
                except:
                    print_exc()

def find_root_multithread(vars, kc, g, dims, im_v, max_steps=MAX_STEPS,
                          use_k_guess=False, factor=1.1, force_gs=True,
                          noise_factors=None):
    vars0 = vars
    L, Ne, Nw = dims
    prev_vars = vars
    sol = root(rgEqs, vars, args=(kc, g, dims),
                   method='lm', jac=rg_jac, options=lmd)
    vars = sol.x
    er = max(abs(rgEqs(vars, kc, g, dims)))
    es, ws = unpack_vars(vars, Ne, Nw)
    if min(abs(ws)) < 0.5*abs(kc[0]) and FORCE_GS:
        print('Omega = 0 solution! Rerunning.')
        er = 1
    tries = 1

    sols = [-999 for i in range(JOBS)]
    ers = [-887 for i in range(JOBS)]

    noise_scale = im_v

    if er > TOL2:
        log('Bad initial guess. Trying with noise.')
        log('g = {}, er = {}'.format(g, er))
    while er > TOL2:
        if tries > max_steps:
            log('Stopping')
            return
        log('{}th try at g = {}'.format(tries, g))
        log('Smallest error from last set: {}'.format(er))
        # log('Retrying with {} sets of new vars:'.format(JOBS))
        if use_k_guess:
            # log('Using k + noise')
            vars0 = np.concatenate(
                                 (np.concatenate((kc[np.arange(Ne)//2],
                                                  kc[L+np.arange(Ne)//2])),
                                  np.concatenate((kc[np.arange(Nw)//2],
                                                  kc[L+np.arange(Nw)//2])))
                                  )
        noise_scale *= factor
        for i, r in enumerate(root_threads(vars0, noise_scale,
                              kc, g, dims, force_gs,
                              noise_factors=noise_factors)):
            sols[i], _, ers[i] = r
        er = np.min(ers)
        sol = sols[np.argmin(ers)]
        vars = sol.x

        tries += 1
    # if not use_k_guess:
    #     # It's more important that I see how my initial guess performed here.
    #     log('Solution - initial guess')
    #     log(vars - vars0)
    return sol

def increment_im_k(vars, dims, g, k, im_k, steps=100, max_steps=MAX_STEPS):
    L, Ne, Nw = dims
    # scale = 1 - np.linspace(0, sf, steps)
    ds = 1./steps
    s = 1.0
    prev_vars = vars
    prev_s = s
    while s > 0:
        log('s = {}'.format(s))
        prev_vars = vars
        prev_s = s

        kc = np.concatenate((k, s*im_k))
        im_v = min(np.linalg.norm(s*im_k), 10**-6)
        sol = find_root_multithread(vars, kc, g, dims, im_v,
                                    max_steps=max_steps)
        vars = sol.x
        er = max(abs(rgEqs(vars, kc, g, dims)))
        if er > 0.001:
            print('This is too bad')
            return
        if er < TOL and ds < 0.08:
            # log('Error is small. Increasing ds')
            ds *= 1.1
            prev_s = s
            prev_vars = vars
            s -= ds
        elif er > TOL2:
            log('Badd error: {}'.format(er))
            if ds > 10**-5:
                log('Backing up and decreasing ds')
                ds *= 0.5
                vars = prev_vars
                s = prev_s - ds
        else:
            prev_s = s
            prev_vars =vars
            s -= ds
    # running at s = 0
    kc = np.concatenate((k, np.zeros(L)))
    sol = find_root_multithread(vars, kc, g, dims, 0,
                                max_steps=MAX_STEPS)
    vars = sol.x
    er = max(abs(rgEqs(vars, kc, g, dims)))
    return vars, er


def increment_im_k_q(vars, dims, q, k, im_k, steps=100):
    # print('dims')
    # print(dims)
    L, Ne, Nw = dims
    ds = 1./steps
    s = 1.0
    prev_vars = vars
    prev_s = s
    while s > 0:
        log('s = {}'.format(s))
        prev_vars = vars
        prev_s = s

        kc = np.concatenate((k, s*im_k))
        im_v = min(np.linalg.norm(s*im_k), 10**-6)
        sol = root(rgEqs_q, vars, args=(kc, q, dims),
                   method='lm', options=lmd)
        vars = sol.x
        er = max(abs(rgEqs_q(vars, kc, q, dims)))
        if er > 0.001:
            print('This is too bad')
            return
        if er < TOL and ds < 0.08:
            # log('Error is small. Increasing ds')
            ds *= 1.1
            prev_s = s
            prev_vars = vars
            s -= ds
        elif er > TOL2:
            log('Badd error: {}'.format(er))
            if ds > 10**-5:
                log('Backing up and decreasing ds')
                ds *= 0.5
                vars = prev_vars
                s = prev_s - ds
        else:
            prev_s = s
            prev_vars =vars
            s -= ds
    # running at s = 0
    kc = np.concatenate((k, np.zeros(L)))
    sol = root(rgEqs_q, vars, args=(kc, q, dims),
               method='lm', options=lmd)
    vars = sol.x
    er = max(abs(rgEqs_q(vars, kc, q, dims)))
    return vars, er


def ioms(es, g, ks, extra_bits=False):
    L = len(ks)
    R = np.zeros(L, dtype=np.complex128)
    for i, k in enumerate(ks):
        R_r = g*np.sum(reZ(k, 0, np.real(es), np.imag(es)))
        R_i = g*np.sum(imZ(k, 0, np.real(es), np.imag(es)))
        R[i] = R_r + 1j*R_i
        if extra_bits:
            otherks = ks[np.arange(L) != i]
            Zkk = rationalZ(k, otherks)
            R[i] += -1*g*np.sum(Zkk) + 1.0
    return R


def calculate_energies(varss, gs, ks, Ne):
    energies = np.zeros(len(gs))
    Rs = np.zeros((len(gs), len(ks)))
    log('Calculating R_k, energy')
    for i, g in enumerate(gs):
        ces, cws = unpack_vars(varss[:, i], Ne, Ne)
        R = ioms(ces, g, ks)
        Rs[i, :] = np.real(R)
        log(R)
        const = 3*g*np.sum(ks**2)/(1+g*np.sum(ks))
        energies[i] = np.sum(ks*np.real(R))*2/(1 + g*np.sum(ks)) - const
        log(energies[i])
    return energies, Rs

def calculate_n_k(Rs, gs):
    dRs = np.zeros(np.shape(Rs))
    nks = np.zeros(np.shape(Rs))
    L = np.shape(Rs)[1]
    for k in range(L):
        Rk = Rs[:, k]
        dRk = np.gradient(Rk, gs)
        nk = 2*(Rk - gs * dRk)

        dRs[:, k] = dRk
        nks[:, k] = nk
    return dRs, nks


def bootstrap_g0(dims, g0, kc,
                 imscale_v=0.001):
    L, Ne, Nw = dims
    if Ne%2 != 0 or Nw%2 != 0:
        print('Error! I need even spin up and down :<')
        return
    Ns = np.arange(2, Ne+1, 2)
    vars = g0_guess(L, 2, 2, kc, g0, imscale=imscale_v)
    er = 10**-6
    for N in Ns:
        log('')
        log('Now using {} fermions'.format(2*N))
        log('')
        dims = (L, N, N)
        # Solving for 2N fermions using extrapolation from previous solution
        if N == 2:
            sol = find_root_multithread(vars, kc, g0, dims, imscale_v,
                                        max_steps=MAX_STEPS,
                                        use_k_guess=True)
        else:
            # The previous solution matches to roughly the accuracy of the solution
            # for the shared variables
            noise_factors = 10*er*np.ones(len(vars))
            # But we will still need to try random stuff for the 4 new variables
            noise_factors[N-2:N] = 1
            noise_factors[2*N-2:2*N] = 1
            noise_factors[3*N-2:3*N] = 1
            noise_factors[4*N-2:4*N] = 1
            sol = find_root_multithread(vars, kc, g0, dims, imscale_v,
                                        max_steps=MAX_STEPS,
                                        use_k_guess=False,
                                        noise_factors=noise_factors)
        vars = sol.x
        er = max(abs(rgEqs(vars, kc, g0, dims)))
        log('Error with {} fermions: {}'.format(2*N, er))
        # Now using this to form next guess
        if N < Ne:
            vars_guess = g0_guess(L, N+2, N+2, kc, g0, imscale=imscale_v)
            es, ws = unpack_vars(vars, N, N)
            esg, wsg = unpack_vars(vars_guess, N+2, N+2)
            es = np.append(es, esg[-2:])
            ws = np.append(ws, wsg[-2:])
            vars = pack_vars(es, ws)

    return sol


def solve_rgEqs(dims, Gf, k, dg=0.01, g0=0.001, imscale_k=0.001,
                  imscale_v=0.001, skip=4):
    L, Ne, Nw = dims
    N = Ne + Nw

    gf = G_to_g(Gf, k)

    gs = np.linspace(g0*np.sign(gf), gf, int(np.abs(gf/dg)))
    Gs = g_to_G(gs, k)

    kim = imscale_k*(-1)**np.arange(L)
    kc = np.concatenate((k, kim))
    vars = g0_guess(L, Ne, Nw, kc, np.sign(gf)*g0, imscale=imscale_v)
    log('Initial guesses:')
    es, ws = unpack_vars(vars, Ne, Nw)
    print(es)
    print(ws)

    keep_going = True
    i = 0
    varss = []
    gss = []

    while i<len(gs) and keep_going:
        g = gs[i]
        if i == 0:
            print('First, boostrapping from 4 to {} fermions'.format(Ne+Nw))
            sol = bootstrap_g0(dims, g0, kc, imscale_v)
        else:
            sol = find_root_multithread(vars, kc, g, dims, imscale_v,
                                        max_steps=10, # if we loose it here, we don't get it back usually
                                        use_k_guess=True)
        try:
            vars = sol.x
            er = max(abs(rgEqs(vars, kc, g, dims)))
            if er > 0.001 and i > 1:
                print('This is too bad')
                return
            ces, cws = unpack_vars(vars, Ne, Nw)
            if i == 0:
                log('Status: {}'.format(sol.status))
                log('Msg: {}'.format(sol.message))
                log('Iterations: {}'.format(sol.nfev))
                log('g = {}'.format(g))
                log('er: {}'.format(er))
                log('Solution vector:')
                log(vars)
                k_full = k + 1j*kim
                log('es - k:')
                log(ces - k_full[np.arange(Ne)//2])
                log('omegas - k:')
                log(cws - k_full[np.arange(Nw)//2])
            elif i % skip == 0 or g == gf and i > 0:
                log('Removing im(k) at g = {}'.format(g))
                vars_r, er_r = increment_im_k(vars, dims, g, k, kim,
                                            steps=10*L,
                                            max_steps=10*JOBS)
                varss += [vars_r]
                es, ws = unpack_vars(vars_r, Ne, Nw)
                print('Variables after removing im(k)')
                print(es)
                print(ws)
                gss += [g]
                log('Stored values at {}'.format(g))

            i += 1
            log('Finished with g = {}'.format(g))
        except Exception as e:
            raise
            print('Error during g incrementing')
            # print(e)
            keep_going = False
    varss = np.array(varss)
    if not keep_going:
        print('Terminated at g = {}'.format(g))
        gs = gs[:i-1]
        gf = gs[-1]
        varss = varss[:, :i-1]
    print('')
    print('Final error:')
    print(er)

    gss = np.array(gss)
    # print(np.shape(varss))

    output_df = pandas.DataFrame({})
    output_df['g'] = gss
    output_df['G'] = g_to_G(gss, k)

    for n in range(Ne):
        output_df['Re(e_{})'.format(n)] = varss[:, n]
        output_df['Im(e_{})'.format(n)] = varss[:, n+Ne]
        output_df['Re(omega_{})'.format(n)] = varss[:, n+2*Ne]
        output_df['Im(omega_{})'.format(n)] = varss[:, n+3*Ne]
    output_df['energy'], Rs = calculate_energies(np.transpose(varss), gss, k, Ne)
    dRs, nks = calculate_n_k(Rs, gss)
    for n in range(L):
        # print(np.shape(Rs[:, n]))
        output_df['R_{}'.format(n)] = Rs[:, n]
        output_df['N_{}'.format(n)] = nks[:, n]



    # ces, cws = unpack_vars(varss[:, -1], Ne, Nw)
    return output_df


def solve_rgEqs_2(dims, Gf, k, dg=0.01, g0=0.001, imscale_k=0.001,
                  imscale_v=0.001, skip=4):
    L, Ne, Nw = dims
    N = Ne + Nw

    Gstar = 1./np.sum(k)
    gf = G_to_g(0.9*Gstar, k)

    gs = np.linspace(g0*np.sign(gf), gf, int(np.abs(gf/dg)))

    kim = imscale_k*(-1)**np.arange(L)
    kc = np.concatenate((k, kim))
    vars = g0_guess(L, Ne, Nw, kc, np.sign(gf)*g0, imscale=imscale_v)
    log('Initial guesses:')
    es, ws = unpack_vars(vars, Ne, Nw)
    print(es)
    print(ws)

    keep_going = True
    i = 0
    varss = []
    gss = []

    while i<len(gs) and keep_going:
        g = gs[i]
        if i == 0:
            print('First, boostrapping from 4 to {} fermions'.format(Ne+Nw))
            sol = bootstrap_g0(dims, g0, kc, imscale_v)
        else:
            sol = find_root_multithread(vars, kc, g, dims, imscale_v,
                                        max_steps=4, # if we loose it here, we don't get it back usually
                                        use_k_guess=True)
        try:
            vars = sol.x
            er = max(abs(rgEqs(vars, kc, g, dims)))
            if er > 0.001 and i > 1:
                print('This is too bad')
                return
            ces, cws = unpack_vars(vars, Ne, Nw)
            if i == 0:
                log('Status: {}'.format(sol.status))
                log('Msg: {}'.format(sol.message))
                log('Iterations: {}'.format(sol.nfev))
                log('g = {}'.format(g))
                log('er: {}'.format(er))
                log('Solution vector:')
                log(vars)
                k_full = k + 1j*kim
                log('es - k:')
                log(ces - k_full[np.arange(Ne)//2])
                log('omegas - k:')
                log(cws - k_full[np.arange(Nw)//2])
            elif i % skip == 0 or g == gf and i > 0:
                log('Removing im(k) at g = {}'.format(g))
                try:
                    vars_r, er_r = increment_im_k(vars, dims, g, k, kim,
                                                steps=10*L,
                                                max_steps=10*JOBS)
                    varss += [vars_r]
                    es, ws = unpack_vars(vars_r, Ne, Nw)
                    print('Variables after removing im(k)')
                    print(es)
                    print(ws)
                    gss += [g]
                    log('Stored values at {}'.format(g))
                except Exception as e:
                    print('Failed while incrementing im part')
                    print(e)
                    print('Continuing....')

            i += 1
            log('Finished with g = {}'.format(g))
        except Exception as e:
            print('Error during g incrementing')
            print(e)
            keep_going = False
    if not keep_going:
        print('Terminated at g = {}'.format(g))
        gs = gs[:i-1]
        gf = gs[-1]
    print('')
    print('Done incrementing g. Error:')
    print(er)
    print('Now incrementing 1/g!')
    q0 = 1./gs[-1]
    qf = 1./(G_to_g(Gf, k))
    print('Final q: {}'.format(qf))
    qs = np.linspace(q0, qf, int(np.abs((qf-q0)/dg)))
    print(qs)
    i = 0
    keep_going = True
    while i<len(qs) and keep_going:
        q = qs[i]
        g = 1./q
        # sol = find_root_multithread(vars, kc, g, dims, imscale_v,
        #                             max_steps=10, # if we loose it here, we don't get it back usually
        #                             use_k_guess=True)

        sol = root(rgEqs_q, vars, args=(kc, q, dims),
                   method='lm', options=lmd) # need jacobian?
        try:
            # print('Got a solution?')
            vars = sol.x
            er = max(abs(rgEqs_q(vars, kc, q, dims)))
            if er > 0.001 and i > 1:
                print('This is too bad')
                return
            ces, cws = unpack_vars(vars, Ne, Nw)
            # if g_to_G(g, ks) <= 1.1*Gstar:
            #     print('Still below Gstar. Better not try anything')
            if i % skip == 0 or q == qf:
                try:
                    log('Removing im(k) at g = {}'.format(g))
                    vars_r, er_r = increment_im_k_q(vars, dims, q, k, kim,
                                                    steps=10*L)
                    varss += [vars_r]
                    es, ws = unpack_vars(vars_r, Ne, Nw)
                    print('Variables after removing im(k)')
                    print(es)
                    print(ws)
                    gss += [g]
                    log('Stored values at {}'.format(g))
                except:
                    pass
            i += 1
            log('Finished with g = {}'.format(g))
        except:
            raise
        # except Exception as e:
        #     print('Error during g incrementing')
        #     print(e)
        #     keep_going = False
    if not keep_going:
        print('Terminated at g = {}'.format(g))
        qs = qs[:i-1]
        qf = qs[-1]

    varss = np.array(varss)
    gss = np.array(gss)


    output_df = pandas.DataFrame({})
    output_df['g'] = gss
    output_df['G'] = g_to_G(gss, k)

    for n in range(Ne):
        output_df['Re(e_{})'.format(n)] = varss[:, n]
        output_df['Im(e_{})'.format(n)] = varss[:, n+Ne]
        output_df['Re(omega_{})'.format(n)] = varss[:, n+2*Ne]
        output_df['Im(omega_{})'.format(n)] = varss[:, n+3*Ne]
    output_df['energy'], Rs = calculate_energies(np.transpose(varss), gss, k, Ne)
    dRs, nks = calculate_n_k(Rs, gss)
    for n in range(L):
        # print(np.shape(Rs[:, n]))
        output_df['R_{}'.format(n)] = Rs[:, n]
        output_df['N_{}'.format(n)] = nks[:, n]
    # ces, cws = unpack_vars(varss[:, -1], Ne, Nw)
    return output_df


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    L = int(input('Length: '))
    Ne = int(input('Nup: '))
    Nw = int(input('Ndown: '))
    # print('Predicted Gc: ')
    # Gc = 8./((L+1)*(2*L+Ne+Nw))
    # print(Gc)

    Gf = float(input('G: '))
    JOBS = int(input('Number of concurrent jobs to run: '))
    dg = float(input('dg: '))
    N = Ne + Nw

    # dg = 0.01/L
    g0 = .1*dg/L
    imk = dg
    imv = .1*g0/N


    dims = (L, Ne, Nw)
    # antiperiodic bc
    ks = np.arange(1, 2*L+1, 2)*0.5*np.pi/L
    # gf = G_to_g(Gf, ks)
    # print('Input G corresponds to g = {}'.format(gf))

    output_df = solve_rgEqs_2(dims, Gf, ks, dg=dg, g0=g0, imscale_k=imk,
                              imscale_v=imv, skip=5*L)
    print('')
    print('Solution found:')

    # output_df.to_csv('{}_{}_{}.csv'.format(L, Ne+Nw, Gf))


    Gf_actual = np.array(output_df['G'])[-1]
    # Gf_actual = Gf
    rge = np.array(output_df['energy'])[-1]

    print('Energy: ')
    print(rge)

    dimH = binom(2*L, Ne)*binom(2*L, Nw)
    G = np.array(output_df['G'])
    E = np.array(output_df['energy'])
    if Gf_actual > 0:
        good_inds = np.logical_and(G <= Gf, G >= 0)
    else:
        good_inds = np.logical_and(G >= Gf, G <= 0)
    print(good_inds)
    G = G[good_inds]
    E = E[good_inds]
    dE = np.gradient(E, G)
    d2E = np.gradient(dE, G)
    d3E = np.gradient(d2E, G)
    plt.figure(figsize=(12,8))
    plt.subplot(2,2,1)
    plt.scatter(G, E)

    plt.title('Energy')
    plt.subplot(2,2,2)
    plt.scatter(G[5:-5], dE[5:-5])
    plt.title('dE')
    plt.subplot(2,2,3)
    plt.scatter(G[5:-5], d2E[5:-5])
    plt.title('d2E')
    plt.subplot(2,2,4)
    plt.scatter(G[5:-5], d3E[5:-5])
    plt.title('d3E')
    plt.show()

    print('Hilbert space dimension: {}'.format(dimH))
    keep_going = input('Input 1 to diagonalize: ')
    if keep_going == '1':
        from exact_qs_so5 import iom_dict, form_basis, ham_op, ham_op_2
        from quspin.operators import quantum_operator
        basis = form_basis(2*L, Ne, Nw)

        # ho = ham_op(L, Gf, ks, basis)
        ho = ham_op_2(L, Gf_actual, ks, basis)
        # e, v = ho.eigsh(k=10, which='SA')
        e, v = ho.eigsh(k=10, which='SA')
        print('Smallest distance from ED result for GS energy:')
        diffs = abs(e-rge)
        print(min(diffs))
        print('This is the {}th energy'.format(np.argmin(diffs)))
