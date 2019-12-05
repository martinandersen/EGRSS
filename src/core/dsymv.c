#include <stdlib.h>
#include <math.h>
#include "egrss.h"

/**
\author Martin S. Andersen
\copyright BSD-2-clause

\brief Computes matrix-vector product with a symmetric extended generator representable p-semiseparable matrix.

The function computes the matrix-vector product \f$x \gets Ax\f$ where
\f[A = \mathrm{tril}(UV^T) + \mathrm{triu}(VU^T,1)\f]
with generators \f$U,V \in \mathbb{R}^{n \times p}\f$.

\note The input arrays \f$U\f$ and \f$V\f$ must be row-major arrays.

@param[in] p      number of columns in U and V
@param[in] n      number of rows in U and V
@param[in] U      row-major array of size n-by-p
@param[in] ldu    leading dimension of U
@param[in] V      row-major array of size n-by-p
@param[in] ldv    leading dimension of V
@param[in,out] x  array of length n
@param[in] incx   stride of x
@param[out] workspace   array of length at least 2p

@return  The function returns the value \c 0 if successful. A return value \c INFO < 0 indicates that input argument \c -INFO is invalid.
*/
int egrss_dsymv(
  const int p,
  const int n,
  const double *restrict U,
  const int ldu,
  const double *restrict V,
  const int ldv,
  double *restrict x,
  const int incx,
  double *restrict workspace
) {

  int info = 0;

  /* Test input parameters */
  if (p < 0) info = -1;
  if (n < 0) info = -2;
  if (U==NULL) info = -3;
  if (ldu < p) info = -4;
  if (V==NULL) info = -5;
  if (ldv < p) info = -6;
  if (x==NULL) info = -7;
  if (incx == 0) info = -8;
  if (workspace==NULL) info = -9;
  if (info != 0) {
    const char *func = __func__;
    egrss_err_param(func, info);
    return info;
  }

  double * restrict ub = workspace;
  double * restrict vb = workspace+p;

  /* Initialize workspace */
  for (int k=0;k<p;k++) { ub[k] = 0.0; }
  for (int k=0;k<p;k++) { vb[k] = 0.0; }
  for (int i=0;i<n;i++) {
    for (int k=0;k<p;k++) {
      ub[k] += U[i*ldu+k]*x[i];
    }
  }

  /* Compute S(U,V)*x */
  for (int i=0;i<n;i++) {

    /* Update ub */
    for(int k=0;k<p;k++) {ub[k] -= U[k]*(*x);}

    /* Update vb */
    for(int k=0;k<p;k++) {vb[k] += V[k]*(*x);}

    /* Overwrite x_i */
    *x = 0.0;
    for(int k=0;k<p;k++) {*x += U[k]*vb[k] + V[k]*ub[k];}

    U += ldu;
    V += ldv;
    x += incx;
  }

  return info;
}
