from spectral_fun import find_spectral_fun
from hubbard import hubbard_akw_1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

L = int(input('Length: '))
N = int(input('N: '))
u = float(input('Coupling strength: '))
# eta = float(input('Infinitesimal: '))
steps = int(input('steps: '))
# k_i = int(input('Which k should I plot? '))
t = 1.
ks = np.array([(2*i+1)*np.pi/(L) for i in range(L//2)]) # actually only the positive ones
# ks_periodic = np.arange(1, L//2+1)*2*np.pi/L
G = u/L
spectral_functions = pd.DataFrame({})
plt.figure(figsize=(12,8))

for i, k in enumerate(ks):
    ki = i + L//2
    L
    hubbard_a, hubbard_o = hubbard_akw_1d(L, N, t, u, k, order=None,
                                          lanczos_steps=None,
                                          steps=steps)
    so5_a_plus, so5_a_minus, so5_o, _ = find_spectral_fun(L//2, N, G, ks, steps=steps,
                                                       k=ki)
    spectral_functions['hubbard_k{}_akw'.format(i)] = hubbard_a
    spectral_functions['hubbard_k{}_o'.format(i)] = hubbard_o
    spectral_functions['so5_k{}_akw'.format(i)] = so5_a_plus + so5_a_minus
    spectral_functions['so5_k{}_o'.format(i)] = so5_o
    print('Integrals at k = {}'.format(k))
    print(np.trapz(hubbard_a, hubbard_o))
    print(np.trapz(so5_a_plus + so5_a_minus, so5_o))
    plt.plot(so5_o, so5_a_plus + so5_a_minus, label = 'SO(5), k = {}'.format(k))
    plt.plot(hubbard_o, hubbard_a, label = 'Hubbard, k = {}'.format(k))
    # plt.title('k = {}'.format(k))
savefile = input('Filename for data:')
# spectral_functions['ks'] = np.concatenate((ks, np.empty(L-steps)*np.nan))
# if savefile != '':
#     spectral_functions.to_csv(savefile)
plt.legend()
plt.xlim(-10, 10)
plt.show()
