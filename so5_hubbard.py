from spectral_fun import find_spectral_fun
from hubbard import hubbard_akw_1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
try:
    with open('context.json') as f:
        context = json.load(f)
    RESULT_FP = context['results_filepath']
except Exception as e:
    print('REQUIRED! context.json file with results_filepath entry')
    raise

L = int(input('Length: '))
N = int(input('N: '))
u = float(input('Coupling strength: '))
eta = float(input('Infinitesimal: '))
steps = int(input('steps: '))
do_hub = int(input('Type 0 to also run Hubbard model: '))
# k_i = int(input('Which k should I plot? '))
t = 1.
ks = np.array([(2*i+1)*np.pi/(L) for i in range(L//2)]) # actually only the positive ones
kps = np.arange(1, L//2+1)*2*np.pi/L
G = 2*u/L
spectral_functions = pd.DataFrame({})
plt.figure(figsize=(12,8))

acolors = ['red','orange','magenta','pink']
hcolors = ['blue','green','cyan','olive']

max_w = 0
min_w = 0

for i, kp in enumerate(kps):
    ki = i + L//2
    if do_hub == 1:
        hap, ham, ho, h_nk = hubbard_akw_1d(L, N, t, u, kp, order=None,
                                          lanczos_steps=None,
                                          steps=steps,
                                          eta=eta)
        spectral_functions['hubbard_k{}_akw'.format(i)] = hap + ham
        spectral_functions['hubbard_k{}_o'.format(i)] = ho
        plt.plot(ho, hap+ham, label = 'Hubbard, k = {}'.format(np.round(kp, 2)), color=hcolors[i])

    sap, sam, so, s_nk = find_spectral_fun(L//2, N, G, ks, steps=steps,
                                                       k=ki,
                                                       eta=eta)

    spectral_functions['so5_k{}_akw'.format(i)] = sap + sam
    spectral_functions['so5_k{}_o'.format(i)] = so

    plt.plot(so, sap+sam, label = 'SO(5), k = {}'.format(np.round(ks[i], 2)), color=acolors[i])

    sos = so[(sap + sam) > 5*10**-3]
    try:
        max_w = max([max_w, max(sos)])
        min_w = min([min_w, min(sos)])
    except:
        pass
plt.legend()

plt.ylabel('A(k,omega)')
plt.xlabel('omega/t')
plt.title('L = {}, N = {}, u = {}, eta = {}t'.format(L, N, u, eta))
print('Min, max omegas:')
print(min_w)
print(max_w)
if min_w != 0 and max_w != 0:
    plt.xlim(min_w, max_w)
    # w = min((-1*min_w, max_w))
    # plt.xlim(-1*w, w)
plt.savefig(RESULT_FP + '/figs/sf_comparison_L{}N{}U{}.png'.format(L, N, np.round(u/t, 2)))
spectral_functions.to_csv(RESULT_FP + 'sf_comparison_L{}N{}U{}.csv'.format(L, N, (np.round(u/t, 2))))
# if input('Type yes for n_k') == 'yes':
#     plt.figure()
#     plt.plot(np.concatenate((-1*ks[::-1], ks)), s_nk, label='SO(5)')
#     # plt.plot(np.concatenate((-1*kps[::-1], kps)), h_nk, label ='Hubbard')
#     plt.xlabel('k')
#     plt.ylabel('<n_k>')
#     plt.title('L = {}, N = {}, u = {}'.format(L, N, u))
#     plt.legend()
#     plt.show()
