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


def brentq(f, xa, xb, xtol, rtol, iter, params):
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

    fpre = f(xpre, *params)
    fcur = f(xcur, *params)
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

        fcur = f(xcur, *params)
        i += 1
#         params->funcalls++;
#     params->error_num = CONVERR;
    return xcur
