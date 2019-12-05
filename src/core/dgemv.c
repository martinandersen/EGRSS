#include <stdlib.h>
#include <math.h>
#include "egrss.h"

/**
\author Martin S. Andersen
\copyright BSD-2-clause

\brief Computes matrix-vector product with an extended generator representable (p,q)-semiseparable matrix.

Computes the matrix-vector product \f$x \gets Ax\f$ where
\f[A = \mathrm{tril}(UV^T) + \mathrm{triu}(PQ^T,1)\f]
with generators \f$U,V \in \mathbb{R}^{n \times p}\f$ and \f$P,Q \in \mathbb{R}^{n \times q}\f$.

\note The input arrays \f$U, V, P,\f$ and \f$Q\f$ must be row-major arrays.

@param[in] p      number of columns in U and V
@param[in] q      number of columns in P and Q
@param[in] n      number of rows in U, V, P, and Q
@param[in] U      row-major array of size n-by-p
@param[in] ldu    leading dimension of U
@param[in] V      row-major array of size n-by-p
@param[in] ldv    leading dimension of V
@param[in] P      row-major array of size n-by-q
@param[in] ldp    leading dimension of P
@param[in] Q      row-major array of size n-by-q
@param[in] ldq    leading dimension of Q
@param[in,out] x  array of length n
@param[in] incx   stride of x
@param[out] workspace   array of length at least p+q

@return  The function returns the value 0 if successful. A return value \c INFO < 0 indicates that input argument \c -INFO is invalid.
*/
int egrss_dgemv(
  const int p,
  const int q,
  const int n,
  const double *restrict U,  // p-by-m
  const int ldu,
  const double *restrict V,  // p-by-min(m,n)
  const int ldv,
  const double *restrict P,  // q-by-min(m,n)
  const int ldp,
  const double *restrict Q,  // q-by-n
  const int ldq,
  double *restrict x,
  const int incx,
  double *restrict workspace
) {

  int info=0;

  /* Test input parameters */
  if (p < 0) info = -1;
  if (q < 0) info = -2;
  if (n < 0) info = -3;
  if (U==NULL) info = -4;
  if (ldu < p) info = -5;
  if (V==NULL) info = -6;
  if (ldv < p) info = -7;
  if (P==NULL) info = -8;
  if (ldp < q) info = -9;
  if (Q==NULL) info = -10;
  if (ldq < q) info = -11;
  if (x==NULL) info = -12;
  if (incx == 0) info = -13;
  if (workspace==NULL) info = -14;
  if (info != 0) {
    const char *func = __func__;
    egrss_err_param(func, info);
    return info;
  }

  /* Initialize workspace */
  for (int k=0;k<p;k++) { workspace[k] = 0.0; }
  for (int k=0;k<q;k++) { workspace[p+k] = 0.0; }
  for (int i=0;i<n;i++) {
    for (int k=0;k<q;k++) {
      workspace[p+k] += Q[i*ldq+k]*x[i*incx];
    }
  }

  /* Compute EGRSS(U,V,P,Q)*x */
  for (int i=0;i<n;i++) {

    /* Update wv */
    for(int k=0;k<p;k++) {workspace[k] += V[k]*(*x);}

    /* Update wq */
    for(int k=0;k<q;k++) {workspace[p+k] -= Q[k]*(*x);}

    /* Overwrite x_i */
    *x = 0.0;
    for(int k=0;k<p;k++) {*x += U[k]*workspace[k];}
    for(int k=0;k<q;k++) {*x += P[k]*workspace[p+k];}

    U += ldu;
    V += ldv;
    P += ldp;
    Q += ldq;
    x += incx;

  }

  return info;
}
