from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator, quantum_LinearOperator
# from quspin.tools.misc import dot
from exact_qs_so5 import form_basis, find_min_ev, ham_op, find_nk, ham_op
import numpy as np
# from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from scipy.linalg import eigh_tridiagonal
from scipy.special import binom


def reduce_state(v, full_basis, target_basis, test=False):
    # v *= 1./np.linalg.norm(v)
    fdim = len(v)
    v_out = np.zeros(len(target_basis.states), dtype=np.complex128)
    for i, s in enumerate(target_basis.states):
        # full_ind = np.where(full_basis.states == s)[0][0]
        v_out[i] = v[fdim - s - 1]
        # v_out[i] = v[full_ind]
    if test:
        vf = target_basis.get_vec(v_out, sparse=False)
        print('<vin|vout>')
        print(np.vdot(v, vf))
        print('| |vin> - |vout> |')
        print(np.linalg.norm(v - vf))
        print('Equal?')
        print((v == vf).all())
        print('Norms')
        print(np.linalg.norm(v))
        print(np.linalg.norm(v_out))
    return v_out/np.linalg.norm(v_out)


def akboth(omega, celts, delts, e0, ep, em, epsilon=10**-10):

    gps = celts*epsilon/((e0 - ep + omega)**2 + epsilon**2)
    gms = delts*epsilon/((e0 - em - omega)**2 + epsilon**2)

    gp2 = celts/(e0 - ep + omega - 1j*epsilon)
    gm2 = delts/(e0 - em - omega - 1j*epsilon)

    d1 = np.linalg.norm(gps - gp2.imag)
    d2 = np.linalg.norm(gms - gm2.imag)
    if d1 > 10**-10 or d2 > 10**-10:
        print('Woah these should be zero')
        print('Diff in g^+')
        print(np.linalg.norm(gps - gp2.imag))
        print('Diff in g^-')
        print(np.linalg.norm(gms - gm2.imag))
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
        print('Creating at {}th spot'.format(k))
        kl = [[1.0, k]]
        cl = [['+|', kl]]
        dl = [['-|', kl]]
        cp_lo = quantum_LinearOperator(cl, basis=bf, check_symm=False,
                                   check_herm=False)
        cm_lo = quantum_LinearOperator(dl, basis=bf, check_symm=False,
                                   check_herm=False)
    else:
        cp_lo, cm_lo = operators
        print(cp_lo)
        print(cm_lo)
    cpv0 = cp_lo.dot(v0)
    cmv0 = cm_lo.dot(v0)
    cpv = reduce_state(cpv0, bf, bp, test=True)
    cmv = reduce_state(cmv0, bf, bm, test=True)
    lc = len(vp[0, :])
    ld = len(vm[0, :])
    # lc = len(vp[:,0])
    # ld = len(vm[:,0])
    celts = np.zeros(lc, dtype=np.complex128)
    delts = np.zeros(ld, dtype=np.complex128)
    print('Finding creation matrix elts.')

    for i in tqdm(range(lc)):
        v = bp.get_vec(vp[:, i], sparse=False)
        # v = vp[:, i]
        celts[i] = np.vdot(v, cpv0)
    print('Finding annihilation matrix elts.')
    for j in tqdm(range(ld)):
        v = bm.get_vec(vm[:, j], sparse=False)
        # v = vm[:, j]
        delts[j] = np.vdot(v, cmv0)
    return np.abs(celts)**2, np.abs(delts)**2


def find_spectral_fun(L, N, G, ks, steps=1000, k=None, n_states=-999):
    Nup = N//2
    Ndown = N//2
    if k is None:
        k = L + Nup//2
    basis = form_basis(2*L, Nup, Ndown)
    basism = form_basis(2*L, Nup-1, Ndown)
    basisp = form_basis(2*L, Nup+1, Ndown)
    basisf = spinful_fermion_basis_1d(2*L)
    h = ham_op(L, G, ks, basis, rescale_g=True)
    hp = ham_op(L, G, ks, basisp, rescale_g=True)
    hm = ham_op(L, G, ks, basism, rescale_g=True)
    if n_states == -999:
        e, v = h.eigsh(k=1, which='SA')
        e0 = e[0]
        v0 = basis.get_vec(v[:,0], sparse=False)
        # e, v = h.eigh()
        ep, vp = hp.eigh()
        em, vm = hm.eigh()
    else:
        e, v = h.eigsh(k=1, which='SA')
        e0 = e[0]
        v0 = basis.get_vec(v[:,0], sparse=False)
        ep, vp = hp.eigsh(k=n_states, which='SA')
        em, vm = hm.eigsh(k=n_states, which='SA')

    celts, delts = matrix_elts(k, v0, vp, vm, basisp, basism, basisf)
    print('Largest matrix elements: Creation')
    print(np.max(celts))
    print(np.argmax(celts))
    print('Annihilation')
    print(np.max(delts))
    print(np.argmax(delts))

    if np.shape(steps) == ():
        ak_plus = np.zeros(steps)
        ak_minus = np.zeros(steps)

        relevant_es = []
        for i, c in enumerate(celts):
            if c > 10**-8:
                relevant_es += [ep[i]]
        for i, c in enumerate(delts):
            if c > 10**-8:
                relevant_es += [em[i]]
        lowmega = min((min(e0 - relevant_es), min(relevant_es - e0)))
        highmega = max((max(e0 - relevant_es), max(relevant_es - e0)))
        omegas = np.linspace(1.2*lowmega, 1.2*highmega, steps)
    else:
        omegas = steps
        ak_plus, ak_minus = np.zeros(len(omegas)), np.zeros(len(omegas))
    epsilon = np.mean(np.diff(omegas))
    # epsilon = .1
    for i, o in enumerate(omegas):
        ak_plus[i], ak_minus[i] = akboth(o, celts, delts, e0, ep, em, epsilon=epsilon)

    ns = find_nk(L, v[:,0], basis)
    return ak_plus, ak_minus, omegas, ns


def check_nonint_spectral_fun(L, N, disp, steps=1000):
    ks = np.arange(L, 2*L)
    plt.figure(figsize=(12, 8))
    colors = ['orange', 'blue', 'pink', 'red','cyan','yellow','magenta','purple']
    # omegas = np.linspace(0, 1.5*np.max(disp), steps)
    for i, k in enumerate(ks):
        print('')
        ap, am, om, ns = find_spectral_fun(L, N, 0, disp, steps, k=k)
        a1 = ap + am
        peak_i = np.argmax(a1)
        peak_om = om[peak_i]
        print('Omega at peak value:')
        print(peak_om)
        print('This should be:')
        print(disp[k-L])
        print('omega - epsilon_k:')
        print(peak_om - disp[k-L])
        print('Integral')
        print(np.trapz(a1, om))
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
            print('Reorthonormalizing {} vectors!'.format(i))
            for j in range(i):
                tau = lvs[:, j] # already normalized
                coeff = np.vdot(w, tau)
                w += -1*coeff*tau
        last_v = v
        if i + 1 < order:
            betas[i+1] = np.linalg.norm(w)
            if betas[i+1] < 10**-6 and i > 2:
                print('{}th beta too small'.format(i+1))
                print(beta)
                stop = True
            else:
                lvs[:, i+1] = w/betas[i+1]
            evals = np.sort(eigh_tridiagonal(alphas[:i+1], betas[1:i+1], eigvals_only=True))
            print('{}th step: ground state converged?'.format(i))
            cvg = np.abs((evals[0] - last_lambda)/evals[0])
            print(cvg)
            print('Beta? {}'.format(betas[i+1]))
            if i >= k:
                print('{}th step: change in {}th eigenvalue'.format(i, k))
                cvg2 = np.abs(evals[k] - last_other)/evals[k]
                print(cvg2)
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
        print('Woops, null vector!')
        return np.zeros(order), np.zeros(order)
    print('Initial state created. Reducing to smaller basis')
    v = reduce_state(op_v0, full_basis, target_basis, test=True)

    print('Performing {}th order Lanczos algorithm'.format(k))
    alphas, betas, vec = lanczos(v, h, order, k=k)
    if k is not None:
        alphas = alphas[:k]
        betas = betas[:k-1]
    es, vs = eigh_tridiagonal(alphas, betas)
    coeffs = np.abs(vs[0, :])**2 # first entries squared
    print('Lanczos coefficients')
    print(np.round(coeffs, 5))
    print('Eigenvalues')
    print(es)
    if max_digits is not None:
        # return np.round(coeffs, max_digits), np.round(es, max_digits)
        coeffs[np.abs(coeffs) < 10**-max_digits] = 0
        es[np.abs(es) < 10**-max_digits] = 0
    return coeffs, es


def lanczos_akw(L, N, G, ks, order, kf=None, steps=1000):

    Nup = N//2
    Ndown = N//2
    if kf is None:
        kf = L + Nup//2
    print('Recommended: kf = {}'.format(L + Nup//2))
    print('Given: kf = {}'.format(kf))
    basis = form_basis(2*L, Nup, Ndown)
    basism = form_basis(2*L, Nup-1, Ndown)
    basisp = form_basis(2*L, Nup+1, Ndown)
    basisf = spinful_fermion_basis_1d(2*L)
    h = ham_op(L, G, ks, basis, rescale_g=True)
    hp = ham_op(L, G, ks, basisp, rescale_g=True)
    hm = ham_op(L, G, ks, basism, rescale_g=True)
    e, v = h.eigsh(k=1, which='SA')
    ef, _ = h.eigsh(k=1, which='LA')

    e0 = e[0]
    ef = ef[0]
    v0 = basis.get_vec(v[:,0], sparse=False)

    print('')
    kl = [[1.0, kf]]
    cl = [['+|', kl]]
    dl = [['-|', kl]]

    print('')
    print('Performing Lanczos for c^+')
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
        print('Woops, too many fermions, raising will mess things up!')
        coeffs_plus, e_plus = np.zeros(order), np.zeros(order)
    print('')
    print('Performing Lanczos for c^-')
    print('')
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
        print('Woops, not enough fermions.')
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
    epsilon = np.mean(np.diff(omegas))
    #epsilon = 0.1
    for i, o in enumerate(omegas):
        aks_p[i], aks_m[i] = akboth(o, coeffs_plus/2, coeffs_minus/2,
                                    e0, e_plus, e_minus, epsilon=epsilon)
    return aks_p, aks_m, omegas


def method_comparison_plots():
    L = int(input('L: '))
    N = int(input('N: '))
    G = float(input('G: '))
    kf = int(input('Which k?: '))

    ks = np.array([(2*i+1)*np.pi/(2*L) for i in range(L)])
    steps = 1000


    # check_nonint_spectral_fun(L, N, ks, steps=1000)
    dimH = binom(2*L, N//2)**2


    print('Hilbert space dimension:')
    print(dimH)
    colors = ['blue','magenta','green','orange','purple','red','cyan']
    styles = [':', '-.', '--']
    xmin = 0
    xmax = 0
    plt.figure(figsize=(12, 8))
    ap_100, am_100, os_100 = lanczos_akw(L, N, G, ks, order=100, kf=kf, steps=steps)
    os = os_100 # this should capture the full range of peaks
    ap_20, am_20, os_20 = lanczos_akw(L, N, G, ks, order=20, kf=kf, steps=os)

    ap_s_20, am_s_20, os_s_20, ns = find_spectral_fun(L, N, G, ks, steps=os_full, k=kf, n_states=20)
    ap_s_100, am_s_100, os_s_100, ns = find_spectral_fun(L, N, G, ks, steps=os, k=kf, n_states=100)

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
    G = float(input('G: '))
    kf = int(input('Which k?: '))

    ks = np.array([(2*i+1)*np.pi/(2*L) for i in range(L)])
    steps = 1000

    # check_nonint_spectral_fun(L, N, ks, steps=1000)
    dimH = binom(2*L, N//2)**2
    colors = ['blue','magenta','green','orange','purple','red','cyan']
    styles = [':', '-.', '--']
    xmin = 0
    xmax = 0
    plt.figure(figsize=(12, 8))
    if dimH < 4000:
        i = 0
        for j, k in enumerate(ks):
            ki = j + L
            print('***********************************')
            print('Full diagonalization for k = {}:'.format(k))
            print('')
            a_plus, a_minus, os, ns = find_spectral_fun(L, N, G, ks, steps, k=ki)
            plt.plot(os, a_minus+a_plus, color=colors[i%len(colors)])
            plt.scatter(os, a_minus, label='A^-, k = {}'.format(np.round(k, 2)), marker='v', color=colors[i%len(colors)])
            plt.scatter(os, a_plus, label='A^+, k = {}'.format(np.round(k,2)), marker='^', color=colors[i%len(colors)])
            xmin = min((min(os), xmin))
            xmax = max((max(os), xmax))
            print('')
            print('Integral')
            print(np.trapz(a_plus+a_minus, os))
            i += 1

        plt.xlabel('omega')
        plt.ylabel('A(k, omega)')
        plt.xlim(xmin, xmax)
        plt.title('L = {}, N = {}, G = {}'.format(L, N, G))
        plt.legend()
    else:
        print('Too big for full diagonalization, :(')
        a_plus, a_minus, os, ns = find_spectral_fun(L, N, G, ks, steps, k=kf, n_states=100)
        plt.plot(os, a_minus, label='Naiive sparse, 100 states', color = 'm')
        print('')
        print('Integral')
        print(np.trapz(a_plus + a_minus, os))

    f = input('Filename to save figure, or blank to display plot')
    if f == '':
        plt.show()
    else:
        plt.savefig(f)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plot_multiple_ks()
