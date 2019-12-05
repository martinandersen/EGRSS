#include <stdlib.h>
#include <math.h>
#include "egrss.h"

/**
\author Martin S. Andersen
\copyright BSD-2-clause

\brief Computes generators for the spline kernel.

Computes matrices \f$U \in \mathbb{R}^{n\times p}\f$ and \f$V \in \mathbb{R}^{n\times p}\f$ such that \f[K = \mathrm{tril}(UV^T) + \mathrm{triu}(VU^T,1)\f]
is the kernel matrix with elements
\f[ K_{ij} = \sum_{k=0}^{p-1} \frac{(-1)^k}{(p-1-k)!(p+k)!} (t_it_j)^{p-1-k} \min(t_i t_j)^{2k+1}, \qquad i,j \in 1,\ldots,n, \f]
where \f$t = (t_1,\ldots,t_n)\f$ is a monotonic sequence of nonnegative numbers.

\note The output arrays \f$U\f$ and \f$V\f$ are stored in row-major order.

@param[in] p
@param[in] n
@param[out] U     row-major array of size n-by-p
@param[in] ldu    leading dimension of U
@param[out] V     row-major array of size n-by-p
@param[in] ldv    leading dimension of V
@param[in] t      array of length n
@param[in] inct   stride of t

@return  The function returns the value \c 0 if successful. A return value \c INFO < 0 indicates that input argument \c -INFO is invalid. A return value \c INFO > 0 means that the input sequence \b t is not monotonic.

@see egrss_dsymv, egrss_dpotrf
*/
int egrss_dsplkgr(
    const int p,
    const int n,
    double *restrict U,
    const int ldu,
    double *restrict V,
    const int ldv,
    const double *restrict t,
    const int inct
) {

    int info = 0;

    /* Test input parameters */
    if (p < 0) info = -1;
    if (n < 0) info = -2;
    if (U==NULL) info = -3;
    if (ldu < p) info = -4;
    if (V==NULL) info = -5;
    if (ldv < p) info = -6;
    if (t==NULL) info = -7;
    if (inct == 0) info = -8;
    if (info != 0) {
      const char *func = __func__;
      egrss_err_param(func, info);
      return info;
    }

    // Check monotonicity
    int increasing = t[0]<=t[inct*(n-1)];

    // Swap U and V is sequence is decreasing
    if (!increasing) {
        double * tmp = U;
        U = V;
        V = tmp;
    }

    /* Compute U and V*/
    double thold = *t;
    for (int i=0;i<n;i++) {
        if ((increasing && *t>=thold) || (!increasing && *t<=thold)) {
            thold = *t;
        }
        else {
            /* Not a monotonic sequence */
            info = i+1;
            break;
        }
        double ti = (thold >= 0) ? thold : 0.0;;

        U[p-1] = 1.0;
        for (int k=p-2;k>=0;k--) {
            U[k] = U[k+1]*ti/(p-1-k);
        }
        V[0] = U[0]*ti/p;
        for (int k=1;k<p;k++) {
            V[k] = -V[k-1]*ti/(p+k);
        }

        U += ldu;
        V += ldv;
        t += inct;
    }

    return info;
}
