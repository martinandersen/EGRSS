#include <stdlib.h>
#include <math.h>
#include "egrss.h"

/**
\author Martin S. Andersen
\copyright BSD-2-clause

\brief Computes Cholesky factorization of a symmetric positive definite extended generator representable p-semiseparable matrix.

The function computes the Cholesky factorization \f$A = LL^T\f$ or \f$A+\mathrm{diag}(d) = LL^T\f$ where \f[A = \mathrm{tril}(UV^T) + \mathrm{triu}(VU^T,1)\f]
with generators \f$U,V \in \mathbb{R}^{n \times p}\f$.

If the input \b d is \c NULL and \b incd is \c 0, the function computes the Cholesky factorization \f$A = LL^T\f$ where \f$L\f$ is given by
\f[ L = \mathrm{tril}(UW^T). \f]
Upon exit, the matrix \f$V\f$ is overwritten by \f$W\f$.

If the input \b d is not equal to \c NULL, the function computes the Cholesky factorization \f$A+\mathrm{diag}(d) = LL^T\f$
where \f$L\f$ is given by
\f[ L = \mathrm{tril}(UW^T,-1) + \mathrm{diag}(c). \f]
Upon exit, the matrix \f$V\f$ is overwritten by \f$W\f$, and \f$d\f$ is overwritten by \f$c\f$.

\note The input arrays \f$U\f$ and \f$V\f$ must be row-major arrays.

@param[in] p      number of columns in U and V
@param[in] n      number of rows in U and V
@param[in] U      row-major array of size n-by-p
@param[in] ldu    leading dimension of U
@param[in,out] V  row-major array of size n-by-p
@param[in] ldv    leading dimension of V
@param[in,out] d  array of length n
@param[in] incd   stride of d
@param[out] workspace   array of length at least p*p

@return  The function returns the value \c 0 if successful. A return value \c INFO < 0 indicates that input argument \c -INFO is invalid. A return value \c INFO > 0 means that the leading minor of order \c INFO is not positive definite, and hence the factorization could not be completed.

@see egrss_dtrmv egrss_dtrsv
*/
int egrss_dpotrf(
  const int p,
  const int n,
  const double *restrict U,
  const int ldu,
  double *restrict V,
  const int ldv,
  double *restrict d,
  const int incd,
  double *restrict workspace
) {

  int info=0;
  double tmp;

  /* Test input parameters */
  if (p < 0) info = -1;
  if (n < 0) info = -2;
  if (U==NULL) info = -3;
  if (ldu < p) info = -4;
  if (V==NULL) info = -5;
  if (ldv < p) info = -6;
  if (d == NULL && incd != 0) info = -8;
  if (d != NULL && incd == 0) info = -8;
  if (workspace==NULL) info = -9;
  if (info != 0) {
    const char *func = __func__;
    egrss_err_param(func, info);
    return info;
  }

  /* Initialize workspace */
  for (int k=0;k<p*p;k++) workspace[k] = 0.0;

  /* Cholesky factorization */
  if (d==NULL) {d = &tmp;}
  for (int i=0; i<n; i++) {

    /* v_i = v_i - P*u_i */
    for (int j=0;j<p;j++) {
      for (int k=0;k<p;k++) {
        V[k] -= workspace[p*j+k]*U[j];
      }
    }

    /* Scale v_i by 1/(u_i'*v_i) or 1/(u_i'*v_i + d_i) */
    tmp = 0.0;
    for (int k=0;k<p;k++) *d += U[k]*V[k];
    if (*d<=0.0) return i+1;
    *d = sqrt(*d);
    tmp = 1.0/(*d);
    for (int k=0;k<p;k++) V[k] *= tmp;

    /* P = P + v_i*v_i' */
    for (int j=0;j<p;j++) {
      for (int k=0;k<p;k++) {
        workspace[p*j+k] += V[j]*V[k];
      }
    }
    U += ldu;
    V += ldv;
    d += incd;
  }
  return info;
}
