# based on scipy/optimize/Zeros/brentq.c
# modified to python by
# William Holmgren william.holmgren@gmail.com @wholmgren
# University of Arizona, 2018

# /* Written by Charles Harris charles.harris@sdl.usu.edu */
#
# #include <math.h>
# #include "zeros.h"
#
# #define MIN(a, b) ((a) < (b) ? (a) : (b))
#
# /*
#   At the top of the loop the situation is the following:
#
#     1. the root is bracketed between xa and xb
#     2. xa is the most recent estimate
#     3. xp is the previous estimate
#     4. |fp| < |fb|
#
#   The order of xa and xp doesn't matter, but assume xp < xb. Then xa lies to
#   the right of xp and the assumption is that xa is increasing towards the root.
#   In this situation we will attempt quadratic extrapolation as long as the
#   condition
#
#   *  |fa| < |fp| < |fb|
#
#   is satisfied. That is, the function value is decreasing as we go along.
#   Note the 4 above implies that the right inequlity already holds.
#
#   The first check is that xa is still to the left of the root. If not, xb is
#   replaced by xp and the erval reverses, with xb < xa. In this situation
#   we will try linear erpolation. That this has happened is signaled by the
#   equality xb == xp;
#
#   The second check is that |fa| < |fb|. If this is not the case, we swap
#   xa and xb and resort to bisection.
#
# */

import numpy as np

from numba import njit, float64

# @njit([(float64, float64, float64)(float64, float64, float64, float64, float64, float64)])
@njit
def bishop88_jit(vd, photocurrent, saturation_current, resistance_series,
                 resistance_shunt, nNsVth):
    """
    Explicit calculation single-diode-model (SDM) currents and voltages using
    diode junction voltages [1].
    [1] "Computer simulation of the effects of electrical mismatches in
    photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
    https://doi.org/10.1016/0379-6787(88)90059-2
    :param numeric vd: diode voltages [V]
    :param numeric photocurrent: photo-generated current [A]
    :param numeric saturation_current: diode one reverse saturation current [A]
    :param numeric resistance_series: series resitance [ohms]
    :param numeric resistance_shunt: shunt resitance [ohms]
    :param numeric nNsVth: product of thermal voltage ``Vth`` [V], diode
        ideality factor ``n``, and number of series cells ``Ns``

    :returns: tuple containing currents [A], voltages [V], power [W],
    """
    a = np.exp(vd / nNsVth)
    b = 1.0 / resistance_shunt
    i = photocurrent - saturation_current * (a - 1.0) - vd * b
    v = vd - i * resistance_series
    retval = i, v, i*v
    return retval


@njit([float64(float64, float64, float64, float64, float64, float64)])
def bishop88_gradp_jit(vd, photocurrent, saturation_current, resistance_series,
             resistance_shunt, nNsVth):
    """root finders only need dp/dv"""
    a = np.exp(vd / nNsVth)
    b = 1.0 / resistance_shunt
    i = photocurrent - saturation_current * (a - 1.0) - vd * b
    v = vd - i * resistance_series
    c = saturation_current * a / nNsVth
    grad_i = - c - b  # di/dvd
    grad_v = 1.0 - grad_i * resistance_series  # dv/dvd
    # dp/dv = d(iv)/dv = v * di/dv + i
    grad = grad_i / grad_v  # di/dv
    grad_p = v * grad + i  # dp/dv
    return grad_p


@njit([float64(float64, float64, float64, float64, float64, float64, float64, float64, float64, float64, float64)])
def brentq_bishop(xa, xb, xtol, rtol, iter,
           vd, photocurrent, saturation_current, resistance_series,
           resistance_shunt, nNsVth):
    xpre = xa
    xcur = xb
    xblk = 0.
    # fpre
    # fcur
    fblk = 0.
    spre = 0.
    scur = 0.
    # sbis

    # the tolerance is 2*delta */
    # delta
    # stry, dpre, dblk
    # i

    fpre = bishop88_gradp_jit(xpre, photocurrent, saturation_current, resistance_series,
           resistance_shunt, nNsVth)
    fcur = bishop88_gradp_jit(xcur, photocurrent, saturation_current, resistance_series,
           resistance_shunt, nNsVth)
#     params->funcalls = 2
    if fpre*fcur > 0:
#         params->error_num = SIGNERR;
        return 0.
    if fpre == 0:
        return xpre
    if fcur == 0:
        return xcur

    iterations = 0
    for i in range(iter):
        iterations += 1
        if fpre*fcur < 0:
            xblk = xpre;
            fblk = fpre;
            spre = scur = xcur - xpre
        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol*abs(xcur))/2
        sbis = (xblk - xcur)/2
        if (fcur == 0 | (abs(sbis) < delta)):
            return xcur

        if (abs(spre) > delta) & (abs(fcur) < abs(fpre)):
            if (xpre == xblk):
                # interpolate
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                stry = (-fcur*(fblk*dblk - fpre*dpre)
                    /(dblk*dpre*(fblk - fpre)))
            if 2*abs(stry) < min(abs(spre), 3*abs(sbis) - delta):
                #good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = bishop88_gradp_jit(xcur, photocurrent, saturation_current, resistance_series,
           resistance_shunt, nNsVth)
        i += 1
#         params->funcalls++;
#     params->error_num = CONVERR;
    return xcur
