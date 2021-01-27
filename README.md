# so5-richardson-gaudin
Solves the Richardson-Gaudin equations for integrable
SO(5) Hamiltonians.

## Dependencies
Along with standard packages (``numpy``, ``scipy``, ``pandas``,
``matplotlib``) included in the Anaconda distribution,
this code requires the ``quspin`` package for
exact-diagonalization.
To install ``quspin`` (if you have Anaconda installed),
run

    conda install -c weinbe58 quspin

in your terminal.

You should also have the ``multiprocessing`` and ``concurrent``
packages installed.

Some of these scripts require a file named ``context.json``
containing a filepath for saving CSVs of results. Format is as follows:

    {"results_filepath": "/path/to/your/folder"}

## Usage

To solve the rational SO(5) model, run

    python solve_rg_eqs.py

in your terminal and input the requested parameters.
Recommended: L = 4, Ne = Nw = 2, G = 1, and dg = 0.01.

## Files
``solve_rg_eqs.py`` solves the Richardson-Gaudin
 equations and derive observables for the SO(5) RG models.

``exact_diag.py`` performs exact diagonalization
on the same Hamiltonians using QuSpin

``spectral_fun.py`` computes spectral functions
using exact diagonalization
