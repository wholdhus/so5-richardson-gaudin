def solve_rgEqs2Part(L, Ne, Nw, gf):
    k = np.array(
                [(2*i+1)*np.pi/L for i in range(L)],
                dtype=np.complex128)

    gs = np.linspace(0, 1, 100)*gf

    ces = k[:Ne]
    # ces = ces - .05 * np.cos(np.pi*np.arange(Ne))
    # cws = k[:Nw] - k[0] # not quite what the exact zero coupling solution is, but this seems to be ok
    cws = np.concatenate(([0.], k[[2*(i+1) for i in range(Nw-1)]]/2.0))
    print('Initial guesses:')
    print(ces)
    print(cws)

    es = np.concatenate((ces.real, ces.imag))
    ws = np.concatenate((cws.real, cws.imag))

    print('Incrementing g with complex k')
    kim = 1j*np.cos(np.pi*np.arange(L))
    ceta = k + kim
    eta = np.concatenate((ceta.real, ceta.imag))
    eprev = es
    wprev = ws
    for i, g in enumerate(gs[1:]):
        log(g)
        wsol = root(rgEqs2, ws, args=(es, eta))
        ws = wsol.x
        esol = root(rgEqs1, es, args=(ws, eta, g))
        es = esol.x

        e1 = np.max(np.abs(rgEqs1(es, ws, eta, g)))
        e2 = np.max(np.abs(rgEqs2(ws, es, eta)))
        if e1 > 10**-12 or e2 > 10**-12:
            log('Highish errors:')
            log('g = {}'.format(g))
            log(e1)
            log(e2)

    print('')
    print('E_alpha:')
    print(es[:Ne]+1j*es[Ne:])
    print('omega_beta:')
    print(ws[:Nw]+1j*ws[Nw:])
    print('')
    print('Incrementing k to be real')
    scale = 1 - np.linspace(0, 1, 10)
    for i, s in enumerate(scale):
        ceta = k + s*kim
        eta = np.concatenate((ceta.real, ceta.imag))

        esol = root(rgEqs1, es, args=(ws, eta, gf))
        es = esol.x

        wsol = root(rgEqs2, ws, args=(es, eta))
        ws = wsol.x

        e1 = np.max(np.abs(rgEqs1(es, ws, eta, gf)))
        e2 = np.max(np.abs(rgEqs2(ws, es, eta)))
        if max(e1, e2) > 10**-10:
            log('Highish errors:')
            log('s = {}'.format(s))
            log(e1)
            log(e2)

    log('This should be about zero:')
    print(np.max(np.abs(rgEqs1(es, ws, eta, g))))
    log('Same with this:')
    print(np.max(np.abs(rgEqs2(ws, es, eta))))
    return es[:Ne] + 1j*es[Ne:], ws[:Nw] + 1j*ws[Nw]
