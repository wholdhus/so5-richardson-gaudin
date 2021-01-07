from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator, quantum_LinearOperator
from exact_diag import form_basis, ham_op, find_nk, ham_op_2, find_skz, reduce_state
import numpy as np
from tqdm import tqdm
from scipy.linalg import eigh_tridiagonal
from scipy.special import binom
import matplotlib.pyplot as plt

from utils import log

VERBOSE = False

def akboth(omega, celts, delts, e0, ep, em, epsilon=10**-10):

    gps = celts*epsilon/((e0 - ep + omega)**2 + epsilon**2)
    gms = delts*epsilon/((e0 - em - omega)**2 + epsilon**2)

    gp2 = celts/(e0 - ep + omega - 1j*epsilon)
    gm2 = delts/(e0 - em - omega - 1j*epsilon)

    d1 = np.linalg.norm(gps - gp2.imag)
    d2 = np.linalg.norm(gms - gm2.imag)
    if d1 > 10**-10 or d2 > 10**-10:
        log('Woah these should be zero')
        log('Diff in g^+')
        log(np.linalg.norm(gps - gp2.imag))
        log('Diff in g^-')
        log(np.linalg.norm(gms - gm2.imag))
    # return (np.sum(gps) - np.sum(gms)).imag/(-1*np.pi), (np.sum(gm2) - np.sum(gp2)).imag/np.pi
    return np.sum(gp2).imag/np.pi, np.sum(gm2.imag)/np.pi


def matrix_elts(k, v0, vp, vm, bp, bm, bf, operators=None):
    """
    Gets matrix elements between creation and annihilation
    at kth site with initial vector v0. v0 should already
    be transformed to the full basis.
    bp, bm are basis for N+1, N-1.
    bf is the basis for all N
    """
    if operators is None:
        log('Creating at {}th spot'.format(k))
        kl = [[1.0, k]]
        cl = [['+|', kl]]
        dl = [['-|', kl]]
        cp_lo = quantum_LinearOperator(cl, basis=bf, check_symm=False,
                                   check_herm=False)
        cm_lo = quantum_LinearOperator(dl, basis=bf, check_symm=False,
                                   check_herm=False)
    else:
        cp_lo, cm_lo = operators
        log(cp_lo)
        log(cm_lo)
    cpv0 = cp_lo.dot(v0)
    cmv0 = cm_lo.dot(v0)
    cpv = reduce_state(cpv0, bf, bp, test=False)
    cmv = reduce_state(cmv0, bf, bm, test=False)
    lc = len(vp[0, :])
    ld = len(vm[0, :])
    # lc = len(vp[:,0])
    # ld = len(vm[:,0])
    celts = np.zeros(lc, dtype=np.complex128)
    delts = np.zeros(ld, dtype=np.complex128)
    # log('Finding creation matrix elts.')

    for i in tqdm(range(lc)):
        v = bp.get_vec(vp[:, i], sparse=False)
        # v = vp[:, i]
        celts[i] = np.vdot(v, cpv0)
    # log('Finding annihilation matrix elts.')
    for j in tqdm(range(ld)):
        v = bm.get_vec(vm[:, j], sparse=False)
        # v = vm[:, j]
        delts[j] = np.vdot(v, cmv0)
    return np.abs(celts)**2, np.abs(delts)**2


def find_spectral_fun(L, N, G, ks, steps=1000, k=None, n_states=-999,
                      eta=None, couplings=None, subtract_ef=False,
                      exactly_solvable=True,
                      combine_states=True,
                      savefile=None, rescale_H=False):
    Nup = N//2
    Ndown = N//2
    if k is None:
        k = L + Nup//2
    basis = form_basis(2*L, Nup, Ndown)
    basism = form_basis(2*L, Nup-1, Ndown)
    basisp = form_basis(2*L, Nup+1, Ndown)
    # basisf = spinful_fermion_basis_1d(2*L, Nf=([(Nup-1, Ndown), (Nup, Ndown), (Nup+1, Ndown)]))
    basisf = spinful_fermion_basis_1d(2*L)
    h = ham_op_2(L, G, ks, basis, couplings=couplings, exactly_solvable=exactly_solvable)
    hp = ham_op_2(L, G, ks, basisp, couplings=couplings, exactly_solvable=exactly_solvable)
    hm = ham_op_2(L, G, ks, basism, couplings=couplings, exactly_solvable=exactly_solvable)

    if n_states == -999:
        if G != -999:
            e, v = h.eigh()
        else:
            # log('Probably degenerate ground state')
            e, v = h.eigh()
            # log('Energies:')
            # log(e)
        e0 = e[0]
        v0 = basis.get_vec(v[:,0], sparse=False)
        ep, vp = hp.eigh()
        em, vm = hm.eigh()
    else:
        e, v = h.eigsh(k=n_states, which='SA')
        e0 = e[0]
        v0 = basis.get_vec(v[:,0], sparse=False)
        ep, vp = hp.eigsh(k=n_states, which='SA')
        em, vm = hm.eigsh(k=n_states, which='SA')
    if combine_states:
        n_zero = 0

        v00 = np.zeros(basis.Ns, dtype=np.complex128)
        for i, ei in enumerate(e):
            if np.abs(ei - e0) < 10**-8:
                v00 += v[:, i]
                n_zero += 1
        if n_zero > 1:
            print('Combined {} degenerate states'.format(n_zero))
        v0 = basis.get_vec(v00, sparse=False)
        v0 *= 1./np.linalg.norm(v0)
    mu = 0
    if subtract_ef:
        mu = (ep[0] - e0)
        # log('Fermi energy: {}'.format(mu))
        e0 -= N*mu
        ep -= (N+1)*mu
        em -= (N-1)*mu
    if rescale_H: # H/G
        e0 *= 1./np.abs(G)
        ep *= 1./np.abs(G)
        em *= 1./np.abs(G)
    celts, delts = matrix_elts(k, v0, vp, vm, basisp, basism, basisf)
    if savefile is not None:
        log('Saving matrix elements to file')
        np.save(savefile+'_plus', np.array([celts, ep]))
        np.save(savefile+'_minus', np.array([delts, em]))
    print('{} nonzero creation elements'.format(len(celts[np.abs(celts) > 10**-10])))
    print('{} nonzero annihilation elements'.format(len(delts[np.abs(delts) > 10**-10])))

    # log('Largest matrix elements: Creation')
    # log(np.max(celts))
    # log(np.argmax(celts))
    # log('Annihilation')
    # log(np.max(delts))
    # log(np.argmax(delts))

    if np.shape(steps) == ():
        ak_plus = np.zeros(steps)
        ak_minus = np.zeros(steps)

        all_es = np.concatenate((ep, em))
        lowmega = min((e0 - max(all_es),
                   min(all_es) - e0))
        highmega = max((max(all_es) - e0,
                    e0 - min(all_es)))
        omegas = np.linspace(1.5*lowmega, 1.5*highmega, steps)
    else:
        omegas = steps
        ak_plus, ak_minus = np.zeros(len(omegas)), np.zeros(len(omegas))
    if eta is None:
        eta = np.mean(np.diff(omegas))
    for i, o in enumerate(omegas):
        ak_plus[i], ak_minus[i] = akboth(o, celts, delts, e0, ep, em, epsilon=eta)

    ns = find_nk(L, v[:,0], basis)
    return ak_plus, ak_minus, omegas, ns


def find_degenerate_spectral_fun(L, N, ks, steps=1000, k=None,
                                 eta=None, couplings=(1,1,1)):
    Nup = N//2
    Ndown = N//2
    if k is None:
        k = L + Nup//2
    basis = form_basis(2*L, Nup, Ndown)
    basism = form_basis(2*L, Nup-1, Ndown)
    basisp = form_basis(2*L, Nup+1, Ndown)
    basisf = spinful_fermion_basis_1d(2*L)
    h = ham_op_2(L, -1, ks, basis, couplings=couplings, no_kin=True)
    hp = ham_op_2(L, -1, ks, basisp, couplings=couplings, no_kin=True)
    hm = ham_op_2(L, -1, ks, basism, couplings=couplings, no_kin=True)

    es, v = h.eigh()

    v00 = np.zeros(len(es), dtype=np.complex128)
    e0 = 0
    # zero_inds = np.abs(es) < 10**-14

    # zero_states = v[:, zero_inds]
    n_zero = 0
    for i, e in enumerate(es):
        if np.abs(e - es[0]) < 10**-8:
            v00 += v[:, i]
            n_zero += 1
    log('{} out of {} states have degenerate energy'.format(n_zero, len(es)))
    e0 = es[0] # should be zero
    if n_zero == 1: # nondegenerate g.s.
        v00 = v[:,0]
    else:
        # v00 = v[:,0]
        v00 *= 1./np.linalg.norm(v00)
    v0 = basis.get_vec(v00, sparse=False)
    # e, v = h.eigh()
    ep, vp = hp.eigh()
    em, vm = hm.eigh()

    mu = (ep[0] - em[0])/2
    log('Fermi energy: {}'.format(mu))
    e0 -= N*mu
    ep -= (N+1)*mu
    em -= (N-1)*mu

    celts, delts = matrix_elts(k, v0, vp, vm, basisp, basism, basisf)
    log('Largest matrix elements: Creation')
    log(np.max(celts))
    log(np.argmax(celts))
    log('Annihilation')
    log(np.max(delts))
    log(np.argmax(delts))

    if np.shape(steps) == ():
        ak_plus = np.zeros(steps)
        ak_minus = np.zeros(steps)

        all_es = np.concatenate((ep, em))
        lowmega = min((e0 - max(all_es),
                   min(all_es) - e0))
        highmega = max((max(all_es) - e0,
                    e0 - min(all_es)))
        omegas = np.linspace(1.5*lowmega, 1.5*highmega, steps)
    else:
        omegas = steps
        ak_plus, ak_minus = np.zeros(len(omegas)), np.zeros(len(omegas))
    if eta is None:
        eta = np.mean(np.diff(omegas))
    for i, o in enumerate(omegas):
        ak_plus[i], ak_minus[i] = akboth(o, celts, delts, e0, ep, em, epsilon=eta)

    ns = find_nk(L, v00, basis)
    return ak_plus, ak_minus, omegas, ns


def check_nonint_spectral_fun(L, N, disp, steps=1000):
    ks = np.arange(L, 2*L)
    plt.figure(figsize=(12, 8))
    colors = ['orange', 'blue', 'pink', 'red','cyan','yellow','magenta','purple']
    # omegas = np.linspace(0, 1.5*np.max(disp), steps)
    for i, k in enumerate(ks):
        log('')
        ap, am, om, ns = find_spectral_fun(L, N, 0, disp, steps, k=k)
        a1 = ap + am
        peak_i = np.argmax(a1)
        peak_om = om[peak_i]
        log('Omega at peak value:')
        log(peak_om)
        log('This should be:')
        log(disp[k-L])
        log('omega - epsilon_k:')
        log(peak_om - disp[k-L])
        log('Integral')
        log(np.trapz(a1, om))
        plt.scatter(om, ap, label = '{}th level, A^+'.format(k), color=colors[i%len(colors)], marker='^')
        plt.scatter(om, am, label = '{}th level, A^-'.format(k), color=colors[i%len(colors)], marker='v')
        plt.axvline(disp[i], color=colors[i%len(colors)])
    plt.xlim(0, 1.5*max(disp))
    plt.legend()
    plt.xlabel('Omega')
    plt.ylabel('A(k,omega)')
    plt.title('L = {}, N = {}'.format(L, N))

    # for d in disp:
    #     plt.axvline(d)

    plt.show()


def lanczos(v0, H, order, k=None, re_orth=True):
    if k is None:
        k = order
    # Normalizing v0:
    v0 *= 1./np.linalg.norm(v0)

    lvs = np.zeros((len(v0), order), dtype=np.complex128)
    lvs[:, 0] = v0
    alphas = np.zeros(order, dtype=np.float64)
    betas = np.zeros(order, dtype=np.float64)
    last_v = np.zeros(len(v0))
    last_lambda = 0
    last_other = 0
    beta = 0
    converged = False
    i = 0
    stop = False
    while i < order and not converged and not stop:
        v = lvs[:, i]
        hv = H.dot(v)
        alphas[i] = np.vdot(hv, v)
        w = hv - beta * last_v - alphas[i] * v
        if re_orth and i > 0:
            log('Reorthonormalizing {} vectors!'.format(i))
            for j in range(i):
                tau = lvs[:, j] # already normalized
                coeff = np.vdot(w, tau)
                w += -1*coeff*tau
        last_v = v
        if i + 1 < order:
            betas[i+1] = np.linalg.norm(w)
            if betas[i+1] < 10**-6 and i > 2:
                log('{}th beta too small'.format(i+1))
                log(beta)
                stop = True
            else:
                lvs[:, i+1] = w/betas[i+1]
            evals = np.sort(eigh_tridiagonal(alphas[:i+1], betas[1:i+1], eigvals_only=True))
            log('{}th step: ground state converged?'.format(i))
            cvg = np.abs((evals[0] - last_lambda)/evals[0])
            log(cvg)
            log('Beta? {}'.format(betas[i+1]))
            if i >= k:
                log('{}th step: change in {}th eigenvalue'.format(i, k))
                cvg2 = np.abs(evals[k] - last_other)/evals[k]
                log(cvg2)
                last_other = evals[k]
                if cvg2 < 10**-12:
                    # converged=True
                    pass
            last_lambda = min(evals)
        i += 1
    lvs[:, i-1] *= 1./np.linalg.norm(lvs[:, i-1])
    return alphas[:i-1], betas[1:i-1], lvs


def lanczos_coeffs(v0, h, op, full_basis, target_basis, order,
                   k=None, max_digits=None):
    op_v0 = op.dot(v0)
    if np.linalg.norm(op_v0) < 10**-12:
        log('Woops, null vector!')
        return np.zeros(order), np.zeros(order)
    log('Initial state created. Reducing to smaller basis')
    v = reduce_state(op_v0, full_basis, target_basis, test=False)

    log('Performing {}th order Lanczos algorithm'.format(k))
    alphas, betas, vec = lanczos(v, h, order, k=k)
    if k is not None:
        alphas = alphas[:k]
        betas = betas[:k-1]
    es, vs = eigh_tridiagonal(alphas, betas)
    coeffs = np.abs(vs[0, :])**2 # first entries squared
    log('Lanczos coefficients')
    log(np.round(coeffs, 5))
    log('Eigenvalues')
    log(es)
    if max_digits is not None:
        # return np.round(coeffs, max_digits), np.round(es, max_digits)
        coeffs[np.abs(coeffs) < 10**-max_digits] = 0
        es[np.abs(es) < 10**-max_digits] = 0
    return coeffs, es


def lanczos_akw(L, N, G, ks, order, kf=None, steps=1000, couplings=None,
                eta=None):

    Nup = N//2
    Ndown = N//2
    if kf is None:
        kf = L + Nup//2
    log('Recommended: kf = {}'.format(L + Nup//2))
    log('Given: kf = {}'.format(kf))
    basis = form_basis(2*L, Nup, Ndown)
    basism = form_basis(2*L, Nup-1, Ndown)
    basisp = form_basis(2*L, Nup+1, Ndown)
    basisf = spinful_fermion_basis_1d(2*L)
    h = ham_op_2(L, G, ks, basis, couplings=couplings)
    hp = ham_op_2(L, G, ks, basisp, couplings=couplings)
    hm = ham_op_2(L, G, ks, basism, couplings=couplings)
    e, v = h.eigsh(k=1, which='SA')
    ef, _ = h.eigsh(k=1, which='LA')

    e0 = e[0]
    ef = ef[0]
    v0 = basis.get_vec(v[:,0], sparse=False)

    log('')
    kl = [[1.0, kf]]
    cl = [['+|', kl]]
    dl = [['-|', kl]]

    log('')
    log('Performing Lanczos for c^+')
    if Nup < 4*L - 2:
        dim_up = binom(4*L, Nup+1)*binom(4*L, Ndown)
        # up_order = min(dim_up//2, order)
        up_order = order
        up_ev = up_order//2
        cp_lo = quantum_LinearOperator(cl, basis=basisf, check_symm=False,
                                       check_herm=False)
        coeffs_plus, e_plus = lanczos_coeffs(v0, hp, cp_lo, basisf, basisp, up_order,
                                             k=up_ev, max_digits=10)
        coeffs_plus, e_plus = coeffs_plus[:up_ev], e_plus[:up_ev]
    else:
        log('Woops, too many fermions, raising will mess things up!')
        coeffs_plus, e_plus = np.zeros(order), np.zeros(order)
    log('')
    log('Performing Lanczos for c^-')
    log('')
    if Nup > 1:
        dim_down = binom(4*L, Nup-1)*binom(4*L, Ndown)
        # down_order = min(dim_down//2, order)
        down_order = order
        down_ev = down_order//2
        cm_lo = quantum_LinearOperator(dl, basis=basisf, check_symm=False,
                                       check_herm=False)
        coeffs_minus, e_minus = lanczos_coeffs(v0, hm, cm_lo, basisf, basism, down_order,
                                               k=down_ev, max_digits=10)
        coeffs_minus, e_minus = coeffs_minus[:down_ev], e_minus[:down_ev]
    else:
        log('Woops, not enough fermions.')
        coeffs_minus, e_minus = np.zeros(L), np.zeros(L)

    if np.shape(steps) == ():
        aks_p = np.zeros(steps)
        aks_m = np.zeros(steps)

        relevant_es = []
        for i, c in enumerate(coeffs_plus):
            if c > 10**-8:
                relevant_es += [e_plus[i]]
        for i, c in enumerate(coeffs_minus):
            if c > 10**-8:
                relevant_es += [e_minus[i]]
        lowmega = min((min(e0 - relevant_es), min(relevant_es - e0)))
        highmega = max((max(e0 - relevant_es), max(relevant_es - e0)))
        omegas = np.linspace(1.2*lowmega, 1.2*highmega, steps)
    else:
        omegas = steps
        aks_p, aks_m = np.zeros(len(omegas)), np.zeros(len(omegas))
    if eta is None:
        epsilon = np.mean(np.diff(omegas))
    else:
        epsilon = eta
    #epsilon = 0.1
    for i, o in enumerate(omegas):
        aks_p[i], aks_m[i] = akboth(o, coeffs_plus/2, coeffs_minus/2,
                                    e0, e_plus, e_minus, epsilon=epsilon)
    return aks_p, aks_m, omegas


def method_comparison_plots(params=None):
    if params is None:
        L = int(input('L: '))
        N = int(input('N: '))
        G = float(input('G: '))
        kf = int(input('Which k?: '))
    else:
        L = params['L']
        N = params['N']
        G = params['G']
        kf = params['kf']

    ks = np.array([(2*i+1)*np.pi/(2*L) for i in range(L)])
    steps = 1000


    # check_nonint_spectral_fun(L, N, ks, steps=1000)
    dimH = binom(2*L, N//2)**2


    log('Hilbert space dimension:')
    log(dimH)
    colors = ['blue','magenta','green','orange','purple','red','cyan']
    styles = [':', '-.', '--']
    xmin = 0
    xmax = 0
    plt.figure(figsize=(12, 8))
    ap, am, os, ns = find_spectral_fun(L, N, G, ks, steps=steps, k=kf)
    ap_100, am_100, os_100 = lanczos_akw(L, N, G, ks, order=100, kf=kf, steps=steps)
    # os = os_100 # this should capture the full range of peaks
    ap_20, am_20, os_20 = lanczos_akw(L, N, G, ks, order=20, kf=kf, steps=os)

    ap_s_20, am_s_20, os_s_20, ns = find_spectral_fun(L, N, G, ks, steps=os, k=kf, n_states=20)
    ap_s_100, am_s_100, os_s_100, ns = find_spectral_fun(L, N, G, ks, steps=os, k=kf, n_states=100)

    plt.plot(os, am+ap, label='Full diagonalization'),
    plt.scatter(os_s_20, ap_s_20 + am_s_20, label = 'Sparse, 20 states', marker='1', color='magenta', s=20)
    plt.scatter(os_s_100, ap_s_100 + am_s_100, label = 'Sparse, 100 states', marker='2', color='green', s=20)
    plt.scatter(os_20, ap_20 + am_20, label = 'Lanczos, 20 steps', marker='o', color='orange', s=20)
    plt.scatter(os_100, ap_100 + am_100, label = 'Lanczos, 100 steps', marker='+', color='purple', s=20)
    plt.legend()
    plt.title('L = {}, N = {}, k = {}'.format(L, N, np.round(ks[kf-L], 2)))
    plt.xlabel('omega')
    plt.ylabel('A(k,omega)')
    # plt.xlim(min(os_s_100), max(os_s_100))
    # plt.xlim(-20, -10)
    # plt.ylim(-0.01, 0.1)
    f = input('Filename to save figure, or blank to display plot: ')
    if f == '':
        plt.show()
    else:
        plt.savefig(f)

def plot_multiple_ks():
    L = int(input('L: '))
    N = int(input('N: '))
    u = float(input('u: '))
    eta = float(input('eta: '))
    if eta < 0:
        eta = None
    if u != -999:
        G = 2.*u/L
    else:
        G = u # this is how to get no k e
    print('Equivalent G:')
    print(G)
    # kf = int(input('Which k?: '))

    ks = np.array([(2*i+1)*np.pi/(L) for i in range(L//2)]) # actually only the positive ones
    steps = 1000

    colors = ['red','orange','magenta','pink']
    styles = [':', '-.', '--']
    xmin = 0
    xmax = 0
    plt.figure(figsize=(12, 8))
    i = 0
    for j, k in enumerate(ks):
        ki = j + L//2
        print('***********************************')
        print('Full diagonalization for k = {}:'.format(k))
        print('')
        if G != -999:
            a_plus, a_minus, os, ns = find_spectral_fun(L//2, N, G, ks, steps, k=ki, eta=0.1)
        else:
            a_plus, a_minus, os, ns = find_degenerate_spectral_fun(L//2, N, ks, steps, k=ki, eta=0.1)
        plt.plot(os, a_minus+a_plus, color=colors[i%len(colors)], label='k = {}'.format(np.round(k,2)))
        # plt.scatter(os, a_minus, label='A^-, k = {}'.format(np.round(k, 2)), marker='v', color=colors[i%len(colors)])
        # plt.scatter(os, a_plus, label='A^+, k = {}'.format(np.round(k,2)), marker='^', color=colors[i%len(colors)])
        # xmin = min((min(os[a_plus + a_minus > 10**-3]), xmin))
        # xmax = max((max(os[a_plus + a_minus > 10**-3]), xmax))
        print('')
        print('Integral')
        print(np.trapz(a_plus+a_minus, os))
        i += 1

    plt.xlabel('omega')
    plt.ylabel('A(k, omega)')
    # plt.xlim(xmin, xmax)
    plt.title('L = {}, N = {}, u/t = {}'.format(L, N, u))
    plt.legend()


    f = input('Filename to save figure, or blank to display plot')
    if f == '':
        plt.show()
    else:
        plt.savefig(f)

    n = input('Type nk to plot nk: ')
    if n == 'nk':
        plt.scatter(np.concatenate((-1*ks[::-1], ks)), ns)
        plt.xlabel('k')
        plt.ylabel('<n_k>')
        plt.show()
    else:
        return

def plot_spectral_functions(L, N, G, ks, k=None):
    Nup = N//2
    Ndown = N//2
    if k is None:
        k = L + Nup//2
    basis = form_basis(2*L, Nup, Ndown)
    basism = form_basis(2*L, Nup-1, Ndown)
    basisp = form_basis(2*L, Nup+1, Ndown)
    basisf = spinful_fermion_basis_1d(2*L)
    h = ham_op_2(L, G, ks, basis)
    hp = ham_op_2(L, G, ks, basisp)
    hm = ham_op_2(L, G, ks, basism)
    e, v = h.eigsh(k=1, which='SA')
    e0 = e[0]
    v0 = basis.get_vec(v[:,0], sparse=False)
    ep, vp = hp.eigh()
    em, vm = hm.eigh()

    celts, delts = matrix_elts(k, v0, vp, vm, basisp, basism, basisf)

    vc = vp[:, np.argmax(celts)]

    vd = vm[:, np.argmax(delts)]

    n0 = find_nk(L, v[:,0], basis)
    nc = find_nk(L, vc, basisp)
    nd = find_nk(L, vd, basism)

    s0 = find_skz(L, v[:,0], basis)
    sc = find_skz(L, vc, basisp)
    sd = find_skz(L, vd, basism)

    ka = np.concatenate((-1*ks[::-1], ks))
    plt.figure(figsize=(12,8))

    plt.subplot(3,1,1)
    plt.scatter(ka, n0, label = np.round(np.sum(n0),0))
    plt.scatter(ka, nc, label = np.round(np.sum(nc),0))
    plt.scatter(ka, nd, label = np.round(np.sum(nd),0))
    plt.xlabel('k')
    plt.ylabel('n_k')
    plt.ylim(0, 2.2)
    plt.legend()

    plt.subplot(3,1,2)
    plt.scatter(ka, s0, label = np.round(np.sum(s0),1))
    plt.scatter(ka, sc, label = np.round(np.sum(sc),1))
    plt.scatter(ka, sd, label = np.round(np.sum(sd),1))
    plt.xlabel('k')
    plt.ylabel('s_k^z')
    plt.ylim(-0.5, 0.5)
    plt.legend()

    akp, akm, o, _ = find_spectral_fun(L, N, G, ks, k=k,
                                       eta=0.1)
    plt.subplot(3,1,3)
    plt.plot(o, akp, label = 'A^+')
    plt.plot(o, akm, label = 'A^-')
    plt.xlabel('omega')
    plt.ylabel('A^+/-(k,omega)')
    plt.legend()

    plt.show()

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # plot_multiple_ks()
    L = 4
    N = 12
    G = 0.5
    ks = np.array([(2*i+1)*np.pi/(2*L) for i in range(L)]) # actually only the positive ones
    kf = L + N//4
    plot_spectral_functions(L, N, G, ks, k=kf)
    # params = {'L': 6, 'N': 6, 'G': 2./3, 'kf': 6}
    # method_comparison_plots(params=params)
    # plot_multiple_ks()
    # method_comparison_plots()
