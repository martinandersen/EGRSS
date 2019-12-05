#include <stdlib.h>
#include <math.h>
#include "egrss.h"

/**
\author Martin S. Andersen
\copyright BSD-2-clause

\brief Computes the squared column norms of a lower triangular extended generator representable p-quasiseparable matrix.

The function computes the squared column norms \f$x_i \gets \|Le_i\|_2^2\f$ of a matrix of the form \f$L = \mathrm{tril}(UW^T,-1) + \mathrm{diag}(c)\f$ where \f$U,W \in \mathbb{R}^{n \times p}\f$ and \f$c \in \mathbb{R}^n\f$.

\note The input arrays \f$U\f$ and \f$W\f$ must be row-major arrays.

@param[in] p      number of columns in U and W
@param[in] n      number of rows in U, W
@param[in] U      row-major array of size n-by-p
@param[in] ldu    leading dimension of U
@param[in] W      row-major array of size n-by-p
@param[in] ldw    leading dimension of W
@param[in] c      array of length n
@param[in] incc   stride of d
@param[out] x     array of length n
@param[in] incx   stride of x
@param[out] workspace   array of length at least p*p

@return  The function returns the value \c 0 if successful. A return value \c INFO < 0 indicates that input argument \c -INFO is invalid.

@see egrss_dpotrf
*/
int egrss_dtrnrms(
  const int p,
  const int n,
  const double *restrict U,
  const int ldu,
  const double *restrict W,
  const int ldw,
  const double *restrict c,
  const int incc,
  double *restrict x,
  const int incx,
  double *restrict workspace
) {

  int info=0;

  /* Test input parameters */
  if (p < 0) info = -1;
  if (n < 0) info = -2;
  if (U==NULL) info = -3;
  if (ldu < p) info = -4;
  if (W==NULL) info = -5;
  if (ldw < p) info = -6;
  if (c == NULL) info = -7;
  if (incc == 0) info = -8;
  if (workspace==NULL) info = -9;
  if (info != 0) {
    const char *func = __func__;
    egrss_err_param(func, info);
    return info;
  }

  /* Initialize workspace */
  for (int k=0;k<p*p;k++) workspace[k] = 0.0;

  U += (n-1)*ldu; W += (n-1)*ldw;
  c += (n-1)*incc; x += (n-1)*incx;
  for (int i=0;i<n;i++) {

    /* Compute x_i */
    if (*c == 0.0) return -(n-1-i);
    *x = (*c)*(*c);
    for (int k=0;k<p;k++){
      for (int j=0;j<p;j++) {
        *x += W[k]*W[j]*workspace[p*k+j];
      }
    }

    /* Update P */
    for (int k=0;k<p;k++){
      for (int j=0;j<p;j++) {
        workspace[p*k+j] += U[k]*U[j];
      }
    }

    U -= ldu; W -= ldw;
    c -= incc; x -= incx;
  }

  return info;
}
