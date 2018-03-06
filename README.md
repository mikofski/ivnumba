ivnumba
=======

Experimenting with numba for IV curves.

brentq.py is a pure python port of scipy's brentq.c. Like the scipy function, this brentq function takes a function to optimize. It's pretty slow due to the overhead of calling a python function within numba.

brentq_bishop.py hard-codes a jitted version of the function that we want to optimize. It is much faster.

numba_mpp_testing.py runs benchmarks and compares the results.
