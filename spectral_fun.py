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
        log('Received omega values! Reusing')
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


def lanczos(v0, H, steps, n_states=None, re_orth=True):
    if n_states is None:
        n_states = steps
    # Normalizing v0:
    v0 *= 1./np.linalg.norm(v0)
    lvs = np.zeros((len(v0), steps), dtype=np.complex128)
    lvs[:, 0] = v0
    alphas = np.zeros(steps, dtype=np.float64)
    betas = np.zeros(steps, dtype=np.float64)
    last_v = np.zeros(len(v0))
    last_lambda = 0
    last_other = 0
    beta = 0
    converged = False
    i = 0
    stop = False
    while i < steps and not converged and not stop:
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
        if i + 1 < steps: # still at least one more step
            betas[i+1] = np.linalg.norm(w)
            if betas[i+1] < 10**-9 and i > 2:
                print('???????????????????')
                log('{}th beta too small'.format(i+1))
                print('STOPSTOPSTOPSTOP')
                print('???????????????????')
                log(beta)
                stop = True
            else:
                lvs[:, i+1] = w/betas[i+1]
            evals = np.sort(eigh_tridiagonal(alphas[:i+1], betas[1:i+1], eigvals_only=True))
            log('{}th step: ground state converged?'.format(i))
            cvg = np.abs((evals[0] - last_lambda)/evals[0])
            log(np.round(cvg, 4))
            log('Beta? {}'.format(betas[i+1]))
            if i >= n_states:
                log('{}th step: change in {}th eigenvalue'.format(i, n_states))
                cvg2 = np.abs(evals[n_states] - last_other)/evals[n_states]
                log(cvg2)
                last_other = evals[n_states]
                if cvg2 < 10**-10:
                    print('!!!!!!!!!!!!!')
                    print('Converged')
                    print('!!!!!!!!!!!!!')
                    converged = True
                    # pass
            last_lambda = min(evals)
        i += 1
    if i >= steps:
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        print('Finished {}th steps'.format(i))
        print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    lvs[:, i-1] *= 1./np.linalg.norm(lvs[:, i-1])
    return alphas[:i-1], betas[1:i-1], lvs


def lanczos_coeffs(v0, h, op, full_basis, target_basis, steps,
                   n_states=None, max_digits=None):
    op_v0 = op.dot(v0)
    if np.linalg.norm(op_v0) < 10**-12:
        log('Woops, null vector!')
        return np.zeros(steps), np.zeros(steps)
    log('Initial state created. Reducing to smaller basis')
    v = reduce_state(op_v0, full_basis, target_basis, test=False)

    log('Performing {}th order Lanczos algorithm'.format(n_states))
    alphas, betas, vec = lanczos(v, h, steps, n_states=n_states)
    if n_states is not None:
        alphas = alphas[:n_states]
        betas = betas[:n_states-1]
    es, vs = eigh_tridiagonal(alphas, betas)
    coeffs = np.abs(vs[0, :])**2 # first entries squared
    log('Lanczos coefficients')
    log(np.round(coeffs, 5))
    log('Eigenvalues')
    log(es)
    if max_digits is not None:
        # Setting coefficients < max_digits to 0
        coeffs[np.abs(coeffs) < 10**-max_digits] = 0
        es[np.abs(es) < 10**-max_digits] = 0
    return coeffs, es


def lanczos_akw(L, N, G, ks, lanczos_steps, kf=None, omega_steps=1000, couplings=None,
                eta=None, lanczos_states=None):

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
        up_order = lanczos_steps
        if lanczos_states is None:
            up_ev = up_order//2
        else:
            up_ev = lanczos_states
        cp_lo = quantum_LinearOperator(cl, basis=basisf, check_symm=False,
                                       check_herm=False)
        coeffs_plus, e_plus = lanczos_coeffs(v0, hp, cp_lo, basisf, basisp, up_order,
                                             n_states=up_ev, max_digits=None)
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
        down_order = lanczos_steps
        if lanczos_states is None:
            down_ev = down_order//2
        else:
            down_ev = lanczos_states
        cm_lo = quantum_LinearOperator(dl, basis=basisf, check_symm=False,
                                       check_herm=False)
        coeffs_minus, e_minus = lanczos_coeffs(v0, hm, cm_lo, basisf, basism, down_order,
                                               n_states=down_ev, max_digits=None)
        coeffs_minus, e_minus = coeffs_minus[:down_ev], e_minus[:down_ev]
    else:
        log('Woops, not enough fermions.')
        coeffs_minus, e_minus = np.zeros(L), np.zeros(L)

    if np.shape(omega_steps) == ():
        aks_p = np.zeros(omega_steps)
        aks_m = np.zeros(omega_steps)

        relevant_es = []
        for i, c in enumerate(coeffs_plus):
            if c > 10**-16:
                relevant_es += [e_plus[i]]
        for i, c in enumerate(coeffs_minus):
            if c > 10**-16:
                relevant_es += [e_minus[i]]
        lowmega = min((min(e0 - relevant_es), min(relevant_es - e0)))
        highmega = max((max(e0 - relevant_es), max(relevant_es - e0)))
        omegas = np.linspace(1.2*lowmega, 1.2*highmega, steps)
    else:
        omegas = omega_steps
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
