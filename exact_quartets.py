from exact_diag import antiperiodic_ham, np, form_basis, quantum_operator, spinful_fermion_basis_1d, reduce_state
from exact_diag import quartet_wavefunction, iso_wavefunction, casimir_dict, find_nk, ham_op_2
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

def ke_op(l, s, k, basis):
    all_k = np.concatenate((k[::-1], k))
    creation_lst = [[s*all_k[i], i] for i in range(l)]
    return quantum_operator({'static': [['n|', creation_lst], ['|n', creation_lst]]}, basis=basis,
                                   check_herm=False, check_symm=False, check_pcon=False)
l = int(sys.argv[1])
N = int(sys.argv[2])
Nup = N//2
Ndown = N//2
k = np.arange(1, 2*l+1, 2)*0.5*np.pi/l
Gc = 1./np.sum(k)
# N = Nup + Ndown

pscales = np.arange(-1, 1.1, .1)
labels = ['S2', 'Gt', 'Gs', 'Gn', 'Gn_2']
titles = [r'$S^2$ perturbation', r'$G_T = G_c+\alpha$', r'$G_S = G_c+\alpha$', r'$G_N = G_c+\alpha$', r'$G_N = G_c+\alpha$, rescaled K.E.']
es_all = [{l: np.zeros(len(pscales)) for l in labels} for i in range(5)]
basis_plus2 = spinful_fermion_basis_1d(2*l, Nf = (Nup+1, Ndown+1))
basis_minus2 = spinful_fermion_basis_1d(2*l, Nf = (Nup-1, Ndown-1))
basis = spinful_fermion_basis_1d(2*l, Nf=(Nup, Ndown))
basis_plus1 = spinful_fermion_basis_1d(2*l, Nf = (Nup+1, Ndown))
basis_minus1 = spinful_fermion_basis_1d(2*l, Nf = (Nup-1, Ndown))
bases = [basis_plus2, basis_plus1, basis, basis_minus1, basis_minus2]
Ns = [N+2, N+1, N, N-1, N-2]
for b in bases:
    print(b.Ns)
    
print('Scaling G_t')
for i, p in enumerate(tqdm(pscales)):
    for j, b in enumerate(bases):
        hp = ham_op_2(l, Gc, k, b, couplings=(1.+p,1.,1.))
        if b.Ns < 10000:
            ep, vp = hp.eigh() 
        else:
            ep, vp = hp.eigsh(k=20, which='SA')
        es_all[j]['Gt'][i] = ep[0]

print('Scaling G_s')
for i, p in enumerate(tqdm(pscales)):
    for j, b in enumerate(bases):
        hp = ham_op_2(l, Gc, k, b, couplings=(1.,1.+p,1.))
        if b.Ns < 10000:
            ep, vp = hp.eigh() 
        else:
            ep, vp = hp.eigsh(k=20, which='SA')
        es_all[j]['Gs'][i] = ep[0]

print('Scaling G_n')
for i, p in enumerate(tqdm(pscales)):
    for j, b in enumerate(bases):
        hp = ham_op_2(l, Gc, k, b, couplings=(1.,1.,1.+p))
        if b.Ns < 6000:
            ep, vp = hp.eigh() 
        else:
            ep, vp = hp.eigsh(k=20, which='SA')
        es_all[j]['Gn'][i] = ep[0]

print('Scaling G_n with correction')
for i, p in enumerate(tqdm(pscales)):
    for j, b in enumerate(bases):
        hp = ham_op_2(l, Gc, k, b, couplings=(1.,1.,1.+p))
        hp += ke_op(l, p, k, b)
        if b.Ns < 10000:
            ep, vp = hp.eigh() 
        else:
            ep, vp = hp.eigsh(k=20, which='SA')
        es_all[j]['Gn_2'][i] = ep[0]

plt.figure(figsize=(6,6), dpi=400)
sp = 1
for i in range(5):
    la = labels[i]
    qs = 0.5*(es_all[0][la]+es_all[4][la]-2*es_all[2][la])
    ps = 0.5*(es_all[1][la]+es_all[3][la]-2*es_all[2][la])
    if la != 'S2':
        plt.subplot(2,2, sp)
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        plt.plot(pscales[:-1], qs[:-1], label=r'$\Delta_4(G_c)$')
        plt.plot(pscales[:-1], ps[:-1], ls=':', label=r'$\Delta_2(G_c)$')
        plt.xlabel(r'$\alpha$')
        plt.title(titles[i])
        sp += 1

    if la == 'Gt':
        plt.legend()
plt.tight_layout(pad=1.0)
plt.savefig('figs/gaps_L{}N{}.png'.format(2*l, Nup+Ndown))

plt.figure(figsize=(6,6), dpi=400)
sp = 1
q_kin = k[N//4]-k[N//4-1]
p_kin = .5*q_kin
for i in range(5):
    la = labels[i]
    qs = 0.5*(es_all[0][la]+es_all[4][la]-2*es_all[2][la])
    ps = 0.5*(es_all[1][la]+es_all[3][la]-2*es_all[2][la])
    if la != 'S2':
        plt.subplot(2,2, sp)
        plt.axhline(0, color='gray')
        plt.axvline(0, color='gray')
        plt.plot(pscales, qs-q_kin, label=r'$\Delta_4(G_c) - \Delta_4(0)$')
        plt.plot(pscales, ps-p_kin, ls=':', label=r'$\Delta_2(G_c) - \Delta_2(0)$')
        plt.xlabel(r'$\alpha$')
        plt.title(titles[i])
        sp += 1

    if la == 'Gt':
        plt.legend()
plt.tight_layout(pad=1.0)
plt.savefig('figs/gaps_minus_ke_L{}N{}.png'.format(2*l, Nup+Ndown))

print('')
print('!!!!!!!!!!!!!!!! Now varying G !!!!!!!!!!!!!!')
print('')

Gs = np.arange(0, 2.1, .1)*Gc
labels = ['All', r'No $N N$', r'No $\vec S \cdot \vec S$', 'Only pairing']
couplings = [(1., 1., 1.), (1., 1., 0), (1., 0., 1.), (1., 0., 0.)]
e_all = [{l: np.zeros(len(pscales)) for l in labels} for i in range(5)]
basis_plus2 = spinful_fermion_basis_1d(2*l, Nf = (Nup+1, Ndown+1))
basis_minus2 = spinful_fermion_basis_1d(2*l, Nf = (Nup-1, Ndown-1))
basis_plus1 = spinful_fermion_basis_1d(2*l, Nf = (Nup+1, Ndown))
basis_minus1 = spinful_fermion_basis_1d(2*l, Nf = (Nup-1, Ndown))
bases = [basis_plus2, basis_plus1, basis, basis_minus1, basis_minus2]
Ns = [N+2, N+1, N, N-1, N-2]
for b in bases:
    print(b.Ns)

for Gi, G in enumerate(tqdm(Gs)):
    for bi, b in enumerate(bases):
        for ci, c in enumerate(couplings):
            hp = ham_op_2(l, G, k, b, couplings=c)
            if b.Ns < 10000:
                ep, vp = hp.eigh() 
            else:
                ep, vp = hp.eigsh(k=1, which='SA')
            e_all[bi][labels[ci]][Gi] = ep[0]
            
colors = ['red', 'blue', 'green', 'orange']
plt.figure(figsize=(5,5), dpi=400)
for li, la in enumerate(labels):
    qg = (e_all[0][la] + e_all[4][la] - 2*e_all[2][la])/2
    pg = (e_all[1][la] + e_all[3][la] - 2*e_all[2][la])/2
    plt.plot(Gs/Gc, qg, label=la, color=colors[li])
    plt.plot(Gs/Gc, pg, label=la, color=colors[li], ls=':')
plt.legend()
plt.xlabel('$g/g_c$')
plt.savefig('figs/gaps_G_L{}N{}.png'.format(2*l, Nup+Ndown))

q_kin = k[N//4]-k[N//4-1]
p_kin = .5*q_kin

plt.figure(figsize=(5,5), dpi=400)
for li, la in enumerate(labels):
    qg = (e_all[0][la] + e_all[4][la] - 2*e_all[2][la])/2
    pg = (e_all[1][la] + e_all[3][la] - 2*e_all[2][la])/2
    plt.plot(Gs/Gc, qg-q_kin, label=la, color=colors[li])
    plt.plot(Gs/Gc, pg-p_kin, label=la, color=colors[li], ls=':')
plt.legend()
plt.xlabel('$g/g_c$')
plt.savefig('figs/gaps_minus_ke_G_L{}N{}.png'.format(2*l, Nup+Ndown))
