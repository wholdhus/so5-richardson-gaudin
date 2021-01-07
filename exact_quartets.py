from exact_diag import antiperiodic_ham, np, form_basis, quantum_operator, spinful_fermion_basis_1d, reduce_state
from exact_diag import quartet_wavefunction, iso_wavefunction, casimir_dict, find_nk, ham_op_2
from tqdm import tqdm
import sys
import pickle
import json

try:
    with open('context.json') as f:
        context = json.load(f)
    RESULT_FP = context['results_filepath']
except:
    print('REQUIRED! context.json file with results_filepath entry')

MAX_ED_STATES = 10000
    
l = int(sys.argv[1])
N = int(sys.argv[2])
Nup = N//2
Ndown = N//2
k = np.arange(1, 2*l+1, 2)*0.5*np.pi/l
Gc = 1./np.sum(k)
N = Nup + Ndown
print('')
print('!!!!!!!!!!!!!!!! varying G !!!!!!!!!!!!!!')
print('')

Gs = np.arange(-1, 3.1, .1)*Gc
labels = ['All', r'No $N N$', r'No $\vec S \cdot \vec S$', 'Only pairing']
couplings = [(1., 1., 1.), (1., 1., 0), (1., 0., 1.), (1., 0., 0.)]
e_all = [{l: np.zeros(len(Gs)) for l in labels} for i in range(5)]
e_kin = [{l: np.zeros(len(Gs)) for l in labels} for i in range(5)]
Ns = [N+2, N+1, N, N-1, N-2]
bases = [spinful_fermion_basis_1d(2*l, Nf=(Ni//2 + Ni%2, Ni//2)) for Ni in Ns]
v0s = []
for b in bases:
    print(b.Ns)
    h0 = ham_op_2(l, 0, k, b)
    if b.Ns < MAX_ED_STATES:
        e0, v0 = h0.eigh() 
    else:
        e0, v0 = h0.eigsh(k=1, which='SA')
    v0s += [v0[:,0]/np.linalg.norm(v0[:,0])]

for Gi, G in enumerate(tqdm(Gs)):
    for bi, b in enumerate(bases):
        for ci, c in enumerate(couplings):
            hp = ham_op_2(l, G, k, b, couplings=c)
            if b.Ns < MAX_ED_STATES:
                ep, vp = hp.eigh() 
            else:
                ep, vp = hp.eigsh(k=1, which='SA')
            e_all[bi][labels[ci]][Gi] = ep[0]
            e_kin[bi][labels[ci]][Gi] = np.real(hp.matrix_ele(v0s[bi], v0s[bi]))
output_data = {'energies_ed': e_all, 'energies_nonint': e_kin, 'Ns': Ns, 'Gs': Gs}
filep = RESULT_FP + '/gap_results/exact_gaps_l{}_N{}.p'.format(l, N)
print('Saving data in file {}'.format(filep))
pickle.dump(output_data, open(filep, 'wb'))