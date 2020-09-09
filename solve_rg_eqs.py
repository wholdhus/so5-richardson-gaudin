import numpy as np
import pandas
from scipy.optimize import root, minimize
from scipy.special import binom
from traceback import print_exc

import concurrent.futures
import multiprocessing

from utils import * # There are a lot of boring functions in this other file

VERBOSE=False
FORCE_GS=True
TOL=10**-11
TOL2=10**-7 # there are plenty of spurious minima around 10**-5
MAXIT=0 # let's use the default value
FACTOR=100
CPUS = multiprocessing.cpu_count()
JOBS = max(CPUS//2, 2)
if CPUS > 10:
    # NOT MY LAPTOP, WILLING TO WAIT A WHILE
    MAX_STEPS_1 = 1000 * JOBS
    MAX_STEPS_2 = 10 * JOBS
else:
    MAX_STEPS_1 = 100 * JOBS
    MAX_STEPS_2 = 10 * JOBS

lmd = {'maxiter': MAXIT,
       'xtol': TOL,
       'ftol': TOL, 'factor':FACTOR}

hybrd = {'maxfev': MAXIT,
         'xtol': TOL,
         'factor': FACTOR}


def gc_guess(L, Ne, Nw, kc, g0, imscale=0.01):
    k_r = kc[:L]
    k_i = kc[L:]
    er = np.zeros(Ne)
    wr = np.zeros(Nw)

    ei = .07*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Ne)])
    wi = -3.2*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Nw)])
    vars = np.concatenate((er, ei, wr, wi))
    return vars



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
    # coefficients are more or less arbitrary
    ei += .07*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Ne)])
    wi += -3.2*imscale*np.array([((i+2)//2)*(-1)**i for i in range(Nw)])

    if Nw%2 == 1:
        wi[-1] = 0 # the additional pairon is zero in this case.
        wr[-1] = 0
    vars = np.concatenate((er, ei, wr, wi))
    return vars


def root_thread_job(vars, kc, g, dims, force_gs):
    """
    Process to run in paralell to look for solutions using multiple
    tries with different noise.
    Inputs:
        vars (numpy array, length 2(Ne + Nw)): Initial guess for variables at g
        g (float): coupling
        dims (tuple): (L, Ne, Nw)
        force_gs (bool): if True, will return error 1 in cases where the solution can't be ground state
    outputs:
        sols (scipy OptimizeResult object): result of solving RG equations
        vars (numpy array): variables obtained. Should be equal to sol.x
        er (float): Maximum absolute residual value of the RG equations for the solution
    """
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    sol = root(rgEqs, vars, args=(kc, g, dims),
               method='lm', jac=rg_jac, options=lmd)
    vars = sol.x
    er = max(abs(rgEqs(vars, kc, g, dims)))
    es, ws = unpack_vars(vars, Ne, Nw)
    if force_gs:
        # Running checks to see if this is deviating from ground state solution
        min_w = np.min(np.abs(ws)) # none of the omegas should get too small
        k_cplx = kc[:L] + 1j*kc[L:]
        k_distance = np.max(np.abs(es - k_cplx[np.arange(Ne)//2])) # the e_alpha should be around the ks
        if k_distance > 10**-3:
            er = 1
        elif min_w < 0.5*kc[0] and (Ne+Nw)%2 == 0:
            er = 1
    # if len(es) >= 12: # this doesn't happen for really small systems
    #     if np.max(np.real(es)) > 3 * np.sort(np.real(es))[-3]:
    #         er = 2  # so I know why this is the error
    if np.isnan(er):
        er = 10
    return sol, vars, er


def root_threads(prev_vars, noise_scale, kc, g, dims, force_gs, noise_factors=None):
    """
    Runs root_thread_vars on multiple cores/processors
    Inputs:
        prev_vars (numpy array, length 2(Ne + Nw)): Initial guess for variables at g
        noise_scale (float): order of magnitude of noise to add to guesses
        kc (numpy array): same as before
        g (float): same as before
        dims (tuple): same as before
        force_gs (bool): same as before
        noise_factors (None or array): If not None, multiplies noise by an additional factor.
        force_gs (bool): if True, will return error 1 in cases where the solution can't be ground state
    """
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
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


def find_root_multithread(vars, kc, g, dims, im_v, max_steps=MAX_STEPS_1,
                          use_k_guess=False, factor=1.01, force_gs=True,
                          noise_factors=None):
    vars0 = vars
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    prev_vars = vars
    sol = root(rgEqs, vars, args=(kc, g, dims),
               method='lm', jac=rg_jac, options=lmd)
    vars = sol.x
    er = max(abs(rgEqs(vars, kc, g, dims)))
    es, ws = unpack_vars(vars, Ne, Nw)
    if min(abs(ws)) < 0.5*min(abs(es)) and force_gs and Nw%2==0:
        # changing to flag only when omega is much smaller than e
        # since all are zero at the critical coupling
        print('Omega = 0 solution! Rerunning.')
        er = 1
    elif len(es) >= 12: # this doesn't happen for really small systems
        if np.max(np.real(es)) > 3 * np.sort(np.real(es))[-3]:
            er = 2  # so I know why this is the error
            print('One of the e_alpha ran away! Rerunning.')
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
        if use_k_guess:
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

        tries += JOBS
    return sol


def increment_im_k(vars, dims, g, k, im_k, steps=100, max_steps=MAX_STEPS_2,
                   force_gs=False):
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
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
                                    max_steps=max_steps,
                                    force_gs=force_gs,
                                    factor=1.5)
        # sol = root(rgEqs, vars, args=(kc, g, dims),
        #            jac=rg_jac, method='lm', options=lmd)
        vars = sol.x
        er = max(abs(rgEqs(vars, kc, g, dims)))
        if er > 0.001:
            print('This is too bad')
            return
        if er < TOL and ds < 0.08:
            ds *= 1.2
            prev_s = s
            prev_vars = vars
            s -= ds
        elif er > TOL2:
            log('Bad error: {} at s = {}'.format(er, s))
            if ds > 10**-3: # Going lower doesn't seem to help
                log('Backing up and decreasing ds')
                ds *= 0.5
                vars = prev_vars
                s = prev_s - ds
            else:
                log('ds is already as small as we can make it.')
                return # exiting this part
        else:
            prev_s = s
            prev_vars =vars
            s -= ds
    # running at s = 0
    kc = np.concatenate((k, np.zeros(L)))
    sol = find_root_multithread(vars, kc, g, dims, .001/L,
                                max_steps=max_steps, force_gs=False)
    vars = sol.x
    er = max(abs(rgEqs(vars, kc, g, dims)))
    return vars, er


def increment_im_k_q(vars, dims, q, k, im_k, steps=100):
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
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
            log('Bad error: {} at s = {}'.format(er, s))
            if ds > 10**-3: # lower doesn't really help us
                log('Backing up and decreasing ds')
                ds *= 0.5
                vars = prev_vars
                s = prev_s - ds
            else:
                return
        else:
            prev_s = s
            prev_vars =vars
            s -= ds
    kc = np.concatenate((k, np.zeros(L)))
    sol = root(rgEqs_q, vars, args=(kc, q, dims),
               method='lm', options=lmd)
    vars = sol.x
    er = max(abs(rgEqs_q(vars, kc, q, dims)))
    return vars, er


def bootstrap_g0(dims, g0, kc,
                 imscale_v=0.001):
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    N = min(2, Ne)
    vars = g0_guess(L, N, N, kc, g0, imscale=imscale_v)
    force_gs=True
    while N <= Ne:
        log('')
        log('Now using {} fermions'.format(2*N))
        log('')
        dims = (L, N, N)
        # Solving for 2N fermions using extrapolation from previous solution
        if N == 2:
            sol = find_root_multithread(vars, kc, g0, dims, imscale_v,
                                        max_steps=MAX_STEPS_1,
                                        use_k_guess=False,
                                        force_gs=force_gs)
        else:
            # The previous solution matches to roughly the accuracy of the solution
            # for the shared variables
            noise_factors = 10**-8*np.ones(4*N)
            # But we will still need to try random stuff for the 4 new variables
            noise_factors[N-2:N] = 1
            noise_factors[2*N-2:2*N] = 1
            noise_factors[3*N-2:3*N] = 1
            noise_factors[4*N-2:4*N] = 1
            sol = find_root_multithread(vars, kc, g0, dims, imscale_v,
                                        max_steps=MAX_STEPS_1,
                                        use_k_guess=False,
                                        noise_factors=noise_factors,
                                        force_gs=force_gs)
            print(vars)
        vars = sol.x
        er = max(abs(rgEqs(vars, kc, g0, dims)))
        log('Error with {} fermions: {}'.format(2*N, er))
        # Now using this to form next guess
        es, ws = unpack_vars(vars, N, N)
        if N != Ne:
            incr = min(2, Ne-N) # 2 or 1
            N += incr
            vars_guess = g0_guess(L, N, N, kc, g0, imscale=imscale_v)
            esg, wsg = unpack_vars(vars_guess, N, N)
            es = np.append(es, esg[-1*incr:])
            ws = np.append(ws, wsg[-1*incr:])
            vars = pack_vars(es, ws)
        else:
            N = Ne + 1 # so we exit # TODO: make this less hacky

    return sol



"""
Code for getting observables from the pairons
"""

def ioms(dims, g, ks, es):
    Z = rationalZ
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    R = np.zeros(L, dtype=np.complex128)
    for i, k in enumerate(ks):
        R[i] = g*np.sum((1-.5*vs[i])*Z(k, es))
        otherks = ks[np.arange(L) != i]
        othervs = vs[np.arange(L) != i]
        R[i] -= g*np.sum(((.5*vs[i]-1)*(.5*othervs-1)-1)
                           *Z(k, otherks))
        R[i] += vs[i]*.5
    return R


def calculate_energies(dims, gs, ks, varss):
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    energies = np.zeros(len(gs))
    Rs = np.zeros((len(gs), L))
    qks = (0.5*vs-1)**2-3*(0.5*vs-1) - 1
    log('Casimirs')
    log(qks)
    log('Calculating R_k, energy')
    for i, g in enumerate(gs):
        ces, cws = unpack_vars(varss[:, i], Ne, Nw)
        R = ioms(dims, g, ks, ces)
        Rs[i, :] = np.real(R)
        # if np.abs(np.imag(R)).any() > 10**-12:
        #     log('Woah R_{} is compelex'.format(i))
        #     log(R)
        const = g*np.sum(qks*ks**2)/(1+g*np.sum(ks))
        energies[i] = np.sum(ks*np.real(R))*2/(1 + g*np.sum(ks)) - const
        # log(energies[i])
    return energies, Rs


def calculate_n_k(dims, gs, Rs):
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    dRs = np.zeros(np.shape(Rs))
    nks = np.zeros(np.shape(Rs))
    L = np.shape(Rs)[1]
    for k in range(L):
        Rk = Rs[:, k]
        dRk = np.gradient(Rk, gs)
        nk = 2*(Rk - gs * dRk)
        dRs[:, k] = dRk
        nks[:, k] = nk
    nks += vs
    return dRs, nks

"""
Code for getting pairons and observables
"""

def solve_rgEqs(dims, Gf, k, dg=0.01, g0=0.001, imscale_k=0.001,
                imscale_v=0.001, skip=4):
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    N = Ne + Nw
    gf = G_to_g(Gf, k)

    kim = imscale_k*(-1)**np.arange(L)
    kc = np.concatenate((k, kim))
    vars = g0_guess(L, Ne, Nw, kc, g0, imscale=imscale_v)

    keep_going = True
    i = 0
    varss = []
    gss = []

    G = g0*np.sign(Gf)
    G_prev = G
    while keep_going and np.abs(G) <= np.abs(Gf):
        g = G_to_g(G, k)
        log('g = {}'.format(g))
        if i == 0:
            print('Bootstrapping from 4 to {} fermions'.format(Ne+Nw))
            try:
                sol = bootstrap_g0(dims, g, kc, imscale_v)
                vars = sol.x
            except Exception as e:
                print('Failed at the initial step.')
                print('Quitting without output')
                return
            print('')
            print('Now incrementing from g = {} to {}'.format(g0, gf))
            print('')
            i += 1
        else:
            sol = root(rgEqs, vars, args=(kc, g, dims),
                       method='lm', options=lmd)#, jac=rg_jac)
        try:
            prev_vars = vars # so we can revert if needed
            vars = sol.x
            er = max(abs(rgEqs(vars, kc, g, dims)))
            ces, cws = unpack_vars(vars, Ne, Nw)
            if i % skip == 0 or np.abs(g - gf) < TOL2:
                log('Removing im(k) at g = {}'.format(g))
                try:
                    vars_r, er_r = increment_im_k(vars, dims, g, k, kim,
                                                  steps=max(L, 10), max_steps=MAX_STEPS_2,
                                                  force_gs=False)
                    es, ws = unpack_vars(vars_r, Ne, Nw)
                    log('Variables after removing im(k)')
                    log(es)
                    log(ws)
                    varss += [vars_r]
                    gss += [g]
                except Exception as e:
                    log('Failed while incrementing im part')
                    log('Continuing....')
                    # er = 1 # so we decrease step size
            if er < TOL and dg < .01*gf:
                print('Increasing dg from {} to {}'.format(dg, dg*2))
                dg *= 2 # we can take bigger steps
                i += 1
            elif er > TOL2 and dg > min_dg:
                print('Decreasing dg from {} to {}'.format(dg, dg*0.5))
                dg *= 0.5
                print('Stepping back from {} to {}'.format(g, g_prev))
                G = G_prev # resetting to last value
                vars = prev_vars
            elif er > 10*TOL2 and dg < min_dg:
                print('Very high error: {}'.format(er))
                print('Cannot make dg smaller!')
                print('Stopping!')
                keep_going = False
            else:
                # Not incrementing i if we're stepping back
                i += 1
            if np.abs(G - Gf) < TOL2:
                print('At Gf')
                keep_going = False
            elif np.abs(G - Gf) < 1.5*dg: # close enough
                print('Close enough to gf')
                G = Gf
            else:
                G += dg * np.sign(Gf)
            G_prev = G
        except Exception as e:
            print('Error during g incrementing')
            print(e)
            print('Quitting the g increments')
            keep_going=False
    varss = np.array(varss)
    gs = np.array(gss)
    if len(gss) == 0:
        print('Failed with no data')
        return
    if not keep_going:
        print('Terminated at g = {}'.format(g))
        gf = g
        # varss = varss[:, :i-1]
    print('')
    print('Final error:')
    print(er)
    print('Recorded {} values'.format(len(gss)))
    gss = np.array(gss)
    output_df = pandas.DataFrame({})
    output_df['g'] = gss
    output_df['G'] = g_to_G(gss, k)
    for n in range(Ne):
        output_df['Re(e_{})'.format(n)] = varss[:, n]
        output_df['Im(e_{})'.format(n)] = varss[:, n+Ne]
        output_df['Re(omega_{})'.format(n)] = varss[:, n+2*Ne]
        output_df['Im(omega_{})'.format(n)] = varss[:, n+3*Ne]
    output_df['energy'], Rs = calculate_energies(dims, gss, k,
                                                 np.transpose(varss))
    dRs, nks = calculate_n_k(dims, gss, Rs)
    for n in range(L):
        output_df['R_{}'.format(n)] = Rs[:, n]
        output_df['N_{}'.format(n)] = nks[:, n]
    return output_df


def solve_rgEqs_2(dims, Gf, k, dg=0.01, g0=0.001, imscale_k=0.001,
                  imscale_v=0.001, skip=4):
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    N = Ne + Nw + np.sum(vs)
    Gstar = 1./np.sum(k)
    gf = G_to_g(0.6*Gstar, k)
    kim = imscale_k*(-1)**np.arange(L)
    kc = np.concatenate((k, kim))
    keep_going = True
    i = 0
    varss = []
    gss = []
    n_fails = 0
    dg0 = dg
    g = g0
    min_dg = np.abs(gf - g0) * 10**-5 # I don't want to do more than 10**5 steps
    max_dg = np.abs(gf - g0) * 10**-2
    print('Incrementing from {} to {}'.format(g0, gf))
    g_prev = g0
    while keep_going and g != gf:
        rat = g_to_G(g, k)*np.sum(k)
        log('g = {}, G/Gc = {}'.format(np.round(g,4), np.round(rat,4)))
        if i == 0:
            print('Bootstrapping from 4 to {} fermions'.format(Ne+Nw))
            try:
                sol = bootstrap_g0(dims, g, kc, imscale_v)
                vars = sol.x
            except Exception as e:
                raise(e)
                print('Failed at the initial step.')
                print('Quitting without output')
                return
            print('')
            print('Now incrementing from g = {} to {}'.format(g0, gf))
            print('')
            i += 1
        else:
            sol = root(rgEqs, vars, args=(kc, g, dims),
                       method='lm', options=lmd, jac=rg_jac,)
        try:
            prev_vars = vars # so we can revert if needed
            vars = sol.x
            er = max(abs(rgEqs(vars, kc, g, dims)))
            ces, cws = unpack_vars(vars, Ne, Nw)
            if np.isnan(ces).any() or np.isnan(cws).any():
                print('Solution is NAN. Ending g loop')
                keep_going = False
            if i == 0:
                pass
            elif i % skip == 0 or g == gf and i > 0:
                log('Removing im(k) at g = {}'.format(g))
                try:
                    vars_r, er_r = increment_im_k(vars, dims, g, k, kim,
                                                  steps=max(L, 10),
                                                  max_steps=MAX_STEPS_2,
                                                  force_gs=False)
                    es, ws = unpack_vars(vars_r, Ne, Nw)
                    log('Variables after removing im(k)')
                    log(es)
                    log(ws)
                    varss += [vars_r]
                    gss += [g]
                except Exception as e:
                    print(e)
                    log('Failed while incrementing im part')
                    log('Continuing....')
                    er = 1 # so we decrease step size
            """
            Code adjusting step sizes
            """
            if er < TOL and dg < .01*gf:
                print('Increasing dg from {} to {}'.format(dg, dg*2))
                dg *= 1.2 # we can take bigger steps
                i += 1
            elif er > TOL2 and dg > min_dg:
                print('Decreasing dg from {} to {}'.format(dg, dg*0.5))
                dg *= 0.5
                print('Stepping back from {} to {}'.format(g, g_prev))
                g = g_prev
                vars = prev_vars
            elif er > 10*TOL2 and dg < min_dg:
                print('Very high error: {}'.format(er))
                print('Cannot make dg smaller!')
                print('Stopping!')
                keep_going = False
            else:
                i += 1
            """
            Code incrementing g for next step
            """
            g_prev = g
            if np.abs(g - gf) < TOL2:
                print('At gf')
                keep_going = False
            elif np.abs(g - gf) < 1.1*dg: # close enough
                print('Close enough to gf')
                g = gf
            else:
                g += dg * np.sign(gf)
        except Exception as e:
            print('Error during g incrementing')
            # print(e)
            print('Quitting the g increments')
            keep_going=False

    print('Done incrementing g at {}. Error:'.format(gf))
    print(er)
    print('Now incrementing 1/g!')
    q0 = 1./gf
    qf = 1./(G_to_g(Gf, k))
    min_dq = np.abs(q0 - qf) * 10**-4 # Taking at most 10**5 steps
    max_dq = np.abs(q0 - qf) * 10**-2 # taking at least 100
    log('Final q: {}'.format(qf))
    dq = dg0
    i = 0
    q = q0
    keep_going = True
    while keep_going:
        rat = g_to_G(1/q, k)*np.sum(k)
        log('q = {}, G/Gc = {}'.format(np.round(q,4),np.round(rat,4)
                        ))
        g = 1./q
        sol = root(rgEqs_q, vars, args=(kc, q, dims),
                   method='lm', options=lmd) # need jacobian?
        try:
            prev_vars = vars
            vars = sol.x
            er = max(abs(rgEqs_q(vars, kc, q, dims)))
            ces, cws = unpack_vars(vars, Ne, Nw)
            if i % skip == 0 or q == qf:
                try:
                    log('Removing im(k) at q = {}'.format(q))
                    vars_r, er_r = increment_im_k_q(vars, dims, q, k, kim,
                                                    steps=max(L, 10))
                    es, ws = unpack_vars(vars_r, Ne, Nw)
                    gss += [g]
                    varss += [vars_r]
                    log('Variables after removing im(k)')
                    log(es)
                    log(ws)
                except Exception as e:
                    print(e)
                    i += 1
                    print('Failed while incrementing im part')
                    print('Continuing ...')
                    er = 1 # so we decrease step size

            """
            Changing step sizes if appropriate
            """
            if er < TOL and dq < max_dq: # Let's allow larger steps for q
                print('Increasing dq from {} to {}'.format(dq, 2*dq))
                dq *= 2
                i += 1
            elif er > TOL2 and dq > min_dq:
                print('Decreasing dq from {} to {}'.format(dq, 0.5*dq))
                q_prev = q - dq*np.sign(qf) # resetting to last value
                dq *= 0.1
                print('Stepping back from {} to {}'.format(q, q_prev))
                q = q_prev
                vars = prev_vars
            elif er > 10**-4 and dq < min_dq:
                print('Very high error: {}'.format(er))
                print('Cannot make dq smaller!')
                print('Stopping!')
                keep_going=False
            else:
                i += 1
            """
            Checking if we are done or close to done
            """
            if np.abs(q-qf) < TOL2:
                print('DID QF!!!!!!!!!')
                keep_going = False
            elif np.abs(q - qf) < 1.5*dq: # close enough
                print('SKIPPING TO QF')
                q = qf
            else:
                q += dq * np.sign(qf)

        except Exception as e:
            print('Error during g incrementing')
            # print(e)
            keep_going = False

    print('Terminated at q = {}'.format(q))
    print('Error: {}'.format(er))
    qf = q
    varss = np.array(varss)
    print('Shape of varter')
    print(np.shape(varss))
    gss = np.array(gss)
    output_df = pandas.DataFrame({})
    output_df['g'] = gss
    output_df['G'] = g_to_G(gss, k)
    for n in range(Ne):
        output_df['Re(e_{})'.format(n)] = varss[:, n]
        output_df['Im(e_{})'.format(n)] = varss[:, n+Ne]
        output_df['Re(omega_{})'.format(n)] = varss[:, n+2*Ne]
        output_df['Im(omega_{})'.format(n)] = varss[:, n+3*Ne]
    output_df['energy'], Rs = calculate_energies(dims, gss, k,
                                                 np.transpose(varss))
    dRs, nks = calculate_n_k(dims, gss, Rs)
    for n in range(L):
        output_df['R_{}'.format(n)] = Rs[:, n]
        output_df['N_{}'.format(n)] = nks[:, n]
    print('')
    print(['!' for i in range(40)])
    print('Finished!')
    print(['!' for i in range(40)])
    return output_df


def solve_Gs_list(dims, Gfs, k, dg=0.01, g0=0.001, imscale_k=0.001,
                  imscale_v=0.001):
    if len(dims) == 3:
        L, Ne, Nw = dims
        vs = np.zeros(L)
    else:
        L, Ne, Nw, vs = dims
    dg0 = dg
    N = Ne + Nw + np.sum(vs)
    Gstar = 1./np.sum(k)
    gf = G_to_g(0.6*Gstar, k)
    kim = imscale_k*(-1)**np.arange(L)
    kc = np.concatenate((k, kim))
    vars = g0_guess(L, Ne, Nw, kc, np.sign(gf)*g0, imscale=imscale_v)
    keep_going = True
    i = 0
    varss = []
    gss = []
    n_fails = 0
    min_dg = np.abs(gf - g0) * 10**-5 # I don't want to do more than 10**5 steps
    max_dg = np.abs(gf - g0) * 10**-2
    print('Incrementing from {} to {}'.format(g0, gf))
    g_prev = g0
    Gf_ind = 0
    gfs = G_to_g(Gfs, k)
    qfs = 1./G_to_g(Gfs, k)

    sol = bootstrap_g0(dims, g0, kc, imscale_v)
    vars = sol.x
    g = g0 + dg

    while keep_going and g <= gf:
        rat = g_to_G(g, k)*np.sum(k)
        log('g = {}, G/Gc = {}'.format(np.round(g,4), np.round(rat,4)))

        sol = root(rgEqs, vars, args=(kc, g, dims),
                   method='lm', options=lmd, jac=rg_jac,)
        try:
            prev_vars = vars # so we can revert if needed
            vars = sol.x
            er = max(abs(rgEqs(vars, kc, g, dims)))
            ces, cws = unpack_vars(vars, Ne, Nw)
            if np.isnan(ces).any() or np.isnan(cws).any():
                print('Solution is NAN. Ending g loop')
                keep_going = False
            """
            Code adjusting step sizes
            """
            if er < TOL and dg < .01*gf:
                print('Increasing dg from {} to {}'.format(dg, dg*2))
                dg *= 1.2 # we can take bigger steps
            elif er > TOL2 and dg > min_dg:
                print('Decreasing dg from {} to {}'.format(dg, dg*0.5))
                dg *= 0.5
                print('Stepping back from {} to {}'.format(g, g_prev))
                g = g_prev
                vars = prev_vars
            elif er > 10*TOL2 and dg < min_dg:
                print('Very high error: {}'.format(er))
                print('Cannot make dg smaller!')
                print('Stopping!')
                keep_going = False
            """
            Code removing imaginary parts and storing results
            """
            if g+dg > gfs[Gf_ind] and g < gfs[Gf_ind]:
                log('Removing im(k) at g = {}'.format(g))
                try:
                    sol = root(rgEqs, vars, args=(kc, gfs[Gf_ind], dims),
                               method='lm', options=lmd, jac=rg_jac,)
                    vars_r, er_r = increment_im_k(vars, dims, gfs[Gf_ind], k, kim,
                                                  steps=max(L, 10),
                                                  max_steps=MAX_STEPS_2,
                                                  force_gs=False)
                    es, ws = unpack_vars(vars_r, Ne, Nw)
                    log('Variables after removing im(k)')
                    log(es)
                    log(ws)
                    varss += [vars_r]
                    gss += [gfs[Gf_ind]]
                except Exception as e:
                    print(e)
                    log('Failed while incrementing im part')
                    log('Continuing....')
                    er = 1 # so we decrease step size
                Gf_ind += 1
            """
            Code incrementing g for next step
            """
            g_prev = g
            i += 1
            g += dg
        except Exception as e:
            print('Error during g incrementing')
            # print(e)
            print('Quitting the g increments')
            keep_going=False

    print('Done incrementing g at {}. Error:'.format(gf))
    print(er)
    print('Now incrementing 1/g!')
    q0 = 1./gf
    qf = 1./(G_to_g(Gfs[-1], k))
    min_dq = np.abs(q0 - qf) * 10**-3 # Taking at most 10**5 steps
    max_dq = np.abs(q0 - qf) * 10**-2 # taking at least 100
    log('Final q: {}'.format(qf))
    dq = dg0
    i = 0
    q = q0
    keep_going = True
    while keep_going and q >= qf: # q decreases
        rat = g_to_G(1/q, k)*np.sum(k)
        log('q = {}, G/Gc = {}'.format(np.round(q,4),np.round(rat,4)
                        ))
        g = 1./q
        sol = root(rgEqs_q, vars, args=(kc, q, dims),
                   method='lm', options=lmd) # need jacobian?
        try:
            prev_vars = vars
            vars = sol.x
            er = max(abs(rgEqs_q(vars, kc, q, dims)))
            ces, cws = unpack_vars(vars, Ne, Nw)

            """
            Changing step sizes if appropriate
            """
            if er < TOL and dq < max_dq: # Let's allow larger steps for q
                print('Increasing dq from {} to {}'.format(dq, 2*dq))
                dq *= 2
            elif er > TOL2 and dq > min_dq:
                print('Decreasing dq from {} to {}'.format(dq, 0.5*dq))
                q_prev = q - dq*np.sign(qf) # resetting to last value
                dq *= 0.1
                print('Stepping back from {} to {}'.format(q, q_prev))
                q = q_prev
                vars = prev_vars
            elif er > 10**-4 and dq < min_dq:
                print('Very high error: {}'.format(er))
                print('Cannot make dq smaller!')
                print('Stopping!')
                keep_going=False
            """
            Removing imaginary parts if needed
            """
            if q - dq < qfs[Gf_ind] and q > qfs[Gf_ind]:
                try:
                    sol = root(rgEqs_q, vars, args=(kc, qfs[Gf_ind], dims),
                               method='lm', options=lmd) # need jacobian?
                    log('Removing im(k) at q = {}'.format(qfs[Gf_ind]))
                    vars_r, er_r = increment_im_k_q(vars, dims, qfs[Gf_ind], k, kim,
                                                    steps=max(L, 10))
                    es, ws = unpack_vars(vars_r, Ne, Nw)
                    gss += [gfs[Gf_ind]]
                    varss += [vars_r]
                    log('Variables after removing im(k)')
                    log(es)
                    log(ws)
                except Exception as e:
                    print(e)
                    i += 1
                    print('Failed while incrementing im part')
                    print('Continuing ...')
                    er = 1 # so we decrease step size
                Gf_ind += 1
            i += q
            q -= dq
        except Exception as e:
            print('Error during g incrementing')
            # print(e)
            keep_going = False

    print('Terminated at q = {}'.format(q))
    print('Error: {}'.format(er))
    qf = q
    varss = np.array(varss)
    print('Shape of varter')
    print(np.shape(varss))
    gss = np.array(gss)
    output_df = pandas.DataFrame({})
    output_df['g'] = gss
    output_df['G'] = g_to_G(gss, k)
    for n in range(Ne):
        output_df['Re(e_{})'.format(n)] = varss[:, n]
        output_df['Im(e_{})'.format(n)] = varss[:, n+Ne]
        output_df['Re(omega_{})'.format(n)] = varss[:, n+2*Ne]
        output_df['Im(omega_{})'.format(n)] = varss[:, n+3*Ne]
    output_df['energy'], Rs = calculate_energies(dims, gss, k,
                                                 np.transpose(varss))
    dRs, nks = calculate_n_k(dims, gss, Rs)
    for n in range(L):
        output_df['R_{}'.format(n)] = Rs[:, n]
        output_df['N_{}'.format(n)] = nks[:, n]
    print('')
    print(['!' for i in range(40)])
    print('Finished!')
    print(['!' for i in range(40)])
    return output_df


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    L = int(input('Length: '))
    Ne = int(input('Nup: '))
    Nw = int(input('Ndown: '))

    Gf = float(input('G: '))
    # JOBS = int(input('Number of concurrent jobs to run: '))

    dg = float(input('dg: '))
    N = Ne + Nw

    # dg = 0.01/L
    g0 = .1*dg/L
    imk = dg
    imv = .1*g0/N


    dims = (L, Ne, Nw, np.zeros(L))
    # antiperiodic bc
    ks = np.arange(1, 2*L+1, 2)*0.5*np.pi/L
    # gf = G_to_g(Gf, ks)
    # print('Input G corresponds to g = {}'.format(gf))
    if Gf > 1./np.sum(ks):
        print('Need to use 1/g for some of this')
        output_df = solve_rgEqs_2(dims, Gf, ks, dg=dg, g0=g0, imscale_k=imk,
                                  imscale_v=imv, skip=L)
    else:
        print('just using g')
        output_df = solve_rgEqs(dims, Gf, ks, dg=dg, g0=g0, imscale_k=imk,
                                imscale_v=imv, skip=4*L)
    print('')
    print('Solution found:')

    # output_df.to_csv('{}_{}_{}.csv'.format(L, Ne+Nw, Gf))


    Gf_actual = np.array(output_df['G'])[-1]

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
    plt.scatter(L*G, E)

    plt.title('Energy')
    plt.subplot(2,2,2)
    plt.scatter(L*G[5:-5], dE[5:-5])
    plt.title('dE')
    plt.subplot(2,2,3)
    plt.scatter(L*G[5:-5], d2E[5:-5])
    plt.title('d2E')
    plt.subplot(2,2,4)
    plt.scatter(L*G[5:-5], d3E[5:-5])
    plt.title('d3E')
    plt.show()

    print('Hilbert space dimension: {}'.format(dimH))
    keep_going = input('Input 1 to diagonalize: ')
    if keep_going == '1':
        from exact_diag import iom_dict, form_basis, ham_op, ham_op_2
        from quspin.operators import quantum_operator
        basis = form_basis(2*L, Ne, Nw)

        ho = ham_op_2(L, Gf_actual, ks, basis)
        e, v = ho.eigsh(k=10, which='SA')
        print('Smallest distance from ED result for GS energy:')
        diffs = abs(e-rge)
        print(min(diffs))
        print('This is the {}th energy'.format(np.argmin(diffs)))
