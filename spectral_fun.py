from quspin.basis import spinful_fermion_basis_1d
from quspin.operators import quantum_operator
# from quspin.tools.misc import matvec
from exact_qs_so5 import hamiltonian_dict, form_basis, find_min_ev, ham_op, find_nk
import numpy as np
# from scipy.sparse.linalg import eigsh
from tqdm import tqdm
from scipy.linalg import eigh_tridiagonal


def reduce_state(v, full_basis, target_basis, test=False):
    v_out = np.zeros(len(target_basis.states), dtype=np.complex128)
    for i, s in enumerate(target_basis.states):
        full_ind = np.where(full_basis.states == s)[0][0]
        v_out[i] = v[full_ind]
    if test:
        vf = target_basis.get_vec(v_out, sparse=False)
        print('<vin|vout>')
        print(np.vdot(v, vf))
        print('| |vin> - |vout> |')
        print(np.linalg.norm(v - vf))
        print('Equal?')
        print((v == vf).all())
    return v_out/np.linalg.norm(v_out)


def akboth(omega, celts, delts, e0, ep, em, epsilon=10**-10):
    # gps = -2j*celts*epsilon/((omega - ep)**2 + epsilon**2)
    # gms = 2j*delts*epsilon/((omega + em)**2 + epsilon**2)
    gps = -2j*celts*epsilon/((omega + e0 - ep)**2 + epsilon**2)
    gms = 2j*delts*epsilon/((omega - e0 + em)**2 + epsilon**2)

    gp2 = celts/(omega - ep +  e0 + 1j*epsilon)
    gm2 = delts/(omega + em - e0 - 1j*epsilon)
    # gp2 = celts/(omega - ep +  1j*epsilon)
    # gm2 = delts/(omega + em - 1j*epsilon)
    return (np.sum(gps) - np.sum(gms)).imag/(-2*np.pi), (np.sum(gm2) - np.sum(gp2)).imag/np.pi


def matrix_elts(k, v0, vp, vm, bp, bm, bf):
    """
    Gets matrix elements between creation and annihilation
    at kth site with initial vector v0. v0 should already
    be transformed to the full basis.
    bp, bm are basis for N+1, N-1.
    bf is the basis for all N
    """
    print('Creating at {}th spot'.format(k))
    kl = [[1.0, k]]
    cl = [['+|', kl]]
    dl = [['-|', kl]]
    cd = {'static': cl}
    dd = {'static': dl}
    cp = quantum_operator(cd, basis=bf, check_symm=False,
                          check_herm=False)
    cm = quantum_operator(dd, basis=bf, check_symm=False,
                          check_herm=False)

    cp_lo = cp.aslinearoperator()
    cm_lo = cm.aslinearoperator()
    cpv = cp_lo.dot(v0)
    cmv = cm_lo.dot(v0)
    lc = len(vp[0, :])
    ld = len(vm[0, :])
    # lc = len(vp[:,0])
    # ld = len(vm[:,0])
    celts = np.zeros(lc, dtype=np.complex128)
    delts = np.zeros(ld, dtype=np.complex128)
    print('Finding creation matrix elts.')

    for i in tqdm(range(lc)):
        v = bp.get_vec(vp[:, i], sparse=False)
        # cpv = matvec(cp_lo, v0)
        # cpv = cp_lo.dot(v0)
        celts[i] = np.vdot(v, cpv)
        # celts[i] = cp.matrix_ele(v, v0)
    print('Finding annihilation matrix elts.')
    for j in tqdm(range(ld)):
        v = bm.get_vec(vm[:, j], sparse=False)
        # cmv = matvec(cm_lo, v0)
        # cmv = cm_lo.dot(v0)
        delts[j] = np.vdot(v, cmv)
        # delts[j] = cm.matrix_ele(v, v0)
    return np.abs(celts)**2, np.abs(delts)**2


def find_spectral_fun(L, N, G, ks, k=None, n_states=-999, steps=None):
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
        e, v = h.eigh()
        ep, vp = hp.eigh()
        em, vm = hm.eigh()
    else:
        # e, v = find_min_ev(h, L, basis, n_states)
        # ep, vp = find_min_ev(hp, L, basisp, n_states)
        # em, vm = find_min_ev(hm, L, basism, n_states)
        # e, v = eigsh(h.aslinearoperator(), k=1, which='SA')
        # ep, vp = eigsh(hp.aslinearoperator(), k=n_states, which='SA')
        # em, vm = eigsh(hm.aslinearoperator(), k=n_states, which='SA')
        e, v = h.eigsh(k=1, which='SA')
        ep, vp = hp.eigsh(k=n_states, which='SA')
        em, vm = hm.eigsh(k=n_states, which='SA')
    if steps is None:
        steps = 10*len(e)

    e0 = e[0]
    v0 = basis.get_vec(v[:,0], sparse=False)

    celts, delts = matrix_elts(k, v0, vp, vm, basisp, basism, basisf)
    print('Largest matrix elements: Creation')
    # print(np.round(celts, 5))
    print(np.max(celts))
    print(np.argmax(celts))
    print('Annihilation')
    print(np.max(delts))
    print(np.argmax(delts))

    aks1 = np.zeros(steps)
    aks2 = np.zeros(steps)
    omegas = np.linspace(-1*np.max(np.abs(e)), np.max(np.abs(e)), steps)
    epsilon = np.mean(np.diff(omegas))
    for i, o in enumerate(omegas):
        aks1[i], aks2[i] = akboth(o, celts, delts, e0, ep, em, epsilon=epsilon)
    #     print(o)
    #     print(aks1[i])
    #     print(aks2[i])
    ns = find_nk(L, v[:,0], basis)
    return aks1, aks2, omegas, ns


def check_nonint_spectral_fun(L, N, disp, steps=1000):
    ks = np.arange(L, 2*L)
    plt.figure(figsize=(12, 8))
    for k in ks:
        print('')
        a1, a2, om, ns = find_spectral_fun(L, N, 0, disp, k=k, steps=steps)
        peak_i = np.argmax(a1)
        peak_om = om[peak_i]
        print('Omega at peak value:')
        print(peak_om)
        print('This should be:')
        print(disp[k-L])
        print('omega - epsilon_k:')
        print(peak_om - disp[k-L])
        print('Integrals')
        print(np.trapz(a1, om))
        print(np.trapz(a2, om))
        plt.scatter(om, a2, label = '{}th level'.format(k))
    plt.xlim(0, 1.5*max(disp))
    plt.legend()
    plt.xlabel('Omega')
    plt.ylabel('A(k,omega)')
    plt.title('L = {}, N = {}'.format(L, 2*N))

    for d in disp:
        plt.axvline(d)

    plt.show()


def lanczos(v0, H, order, test=True, k=None):
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
        # print('Reorthonormalizing!')
        for j in range(i):
            tau = lvs[:, j] # already normalized
            coeff = np.vdot(w, tau)
            w += -1*coeff*tau
        last_v = v
        if i + 1 < order:
            betas[i+1] = np.linalg.norm(w)
            if i > 1 and betas[i+1] < 10**-6:
                print('{}th step: beta too small'.format(i))
                print(beta)
                stop = True
            lvs[:, i+1] = w/betas[i+1]
            evals = np.sort(eigh_tridiagonal(alphas[:i+1], betas[1:i+1], eigvals_only=True))
            # print('Converged?')
            # cvg = np.abs((min(evals) - last_lambda)/min(evals))
            # print(cvg)
            if i >= k:
                print('{}th step: change in {}th eigenvalue'.format(i, k))
                cvg2 = np.min(np.abs(evals[k] - last_other))
                print(cvg2)
                last_other = evals[k]
                if cvg2 < 10**-7:
                    converged=True
            last_lambda = min(evals)
        i += 1
    lvs[:, i-1] *= 1./np.linalg.norm(lvs[:, i-1])

    if test:
        es, _ = np.linalg.eig(H)
        print('Testing:')
        evs, _ = eigh_tridiagonal(alphas, betas[1:])
        print('Eigenvalues of Lanczos matrix')
        print(evs)
        for i, e in enumerate(evs):
            diff_e = np.abs(es - e)
            if np.min(diff_e) > 10**-6:
                print('{}th eigenvalue is bad: difference is...'.format(i))
                print(np.min(diff_e))
    return alphas, betas[1:], lvs


def lanczos_coeffs(v0, h, op, full_basis, target_basis, order, k=None):
    # Get lanczos vectors, matrix
    # H should be in the target basis
    # op is c dagger or c or whatever
    # Follow formula
    # $$$$Profit????
    op_v0 = op.matvec(v0)
    print('Initial state created. Reducing to smaller basis')
    v = reduce_state(op_v0, full_basis, target_basis)
    print('Performing {}th order Lanczos algorithm'.format(k))
    alphas, betas, vec = lanczos(v, h, order, test=False, k=k)

    es, vs = eigh_tridiagonal(alphas, betas)
    coeffs = np.abs(vs[0, :])**2 # first entries squared
    return coeffs, es


def lanczos_akw(L, N, G, order, k=None):
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
    e, v = h.eigsh(k=1, which='SA')
    ef, _ = h.eigsh(k=1, which='LA')
    steps = 2*order

    e0 = min(e)
    ef = ef[0]
    print('')
    print('Eigenvalues:')
    print(e0)
    print(ef)
    v0 = basis.get_vec(v[:,0], sparse=False)
    kl = [[1.0, k]]
    cl = [['+|', kl]]
    dl = [['-|', kl]]
    cd = {'static': cl}
    dd = {'static': dl}
    cp = quantum_operator(cd, basis=basisf, check_symm=False,
                          check_herm=False)
    cm = quantum_operator(dd, basis=basisf, check_symm=False,
                          check_herm=False)

    cp_lo = cp.aslinearoperator()
    cm_lo = cm.aslinearoperator()
    print('')
    print('Performing Lanczos for c^+')
    coeffs_plus, e_plus = lanczos_coeffs(v0, hp, cp_lo, basisf, basisp, order)
    print('')
    print('Performing Lanczos for c^-')
    print('')
    coeffs_minus, e_minus = lanczos_coeffs(v0, hm, cm_lo, basisf, basism, order)

    aks1 = np.zeros(steps)
    aks2 = np.zeros(steps)
    e_mag = np.max(np.abs((e0, ef)))
    omegas = np.linspace(-5*e_mag, 5*e_mag, steps)
    epsilon = np.mean(np.diff(omegas))
    Gcs = np.zeros(steps)
    for i, o in enumerate(omegas):
        aks1[i], aks2[i] = akboth(o, coeffs_plus, coeffs_minus,
                                  e0, e_plus, e_minus, epsilon=epsilon)
    return aks1, aks2, omegas


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    dim = 200
    # mat = 0.01 * np.random.rand(20, 20) + np.diag(np.arange(20))
    mat = np.random.rand(dim, dim)
    mat += np.transpose(mat)
    print('Am I HErmitian?')
    print((mat == np.transpose(mat)).all())
    print('Performing Lanczos on a random matrix!')
    a, b, v = lanczos(0.5 - np.random.rand(dim), mat, 60, test=True)
