#include <stdlib.h>
#include <math.h>
#include "egrss.h"

/**
\author Martin S. Andersen
\copyright BSD-2-clause

\brief Computes matrix-vector product with a lower/upper triangular extended generator representable p-quasiseparable matrix.

Computes the matrix-vector product \f$x \gets Lx\f$ (@a trans is @c N) or \f$x \gets L^Tx\f$ (@a trans is @c T) where \f$L\f$ is either of the form
\f[L = \mathrm{tril}(UW^T)\f]
(if @a c is @c NULL and @a incc is @c 0) or
\f[L = \mathrm{tril}(UW^T,-1) + \mathrm{diag}(c) \f]
with generators \f$U,W \in \mathbb{R}^{n \times p}\f$ and \f$c \in \mathbb{R}^n\f$.

\note The input arrays \f$U\f$ and \f$W\f$ must be row-major arrays.

@param[in] trans  \c N or \c T
@param[in] p      number of columns in U and W
@param[in] n      number of rows in U, W
@param[in] U      row-major array of size n-by-p
@param[in] ldu    leading dimension of U
@param[in] W      row-major array of size n-by-p
@param[in] ldw    leading dimension of W
@param[in] c      array of length n
@param[in] incc   stride of d
@param[in,out] x  array of length n
@param[in] incx   stride of x
@param[out] workspace   array of length at least p

@return  The function returns the value \c 0 if successful. A return value \c INFO < 0 indicates that input argument \c -INFO is invalid.

@see egrss_dpotrf egrss_dtrsv
*/
int egrss_dtrmv(
  const char trans,
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
  if (trans != 'N' && trans != 'T') info = -1;
  if (p < 0) info = -2;
  if (n < 0) info = -3;
  if (U==NULL) info = -4;
  if (ldu < p) info = -5;
  if (W==NULL) info = -6;
  if (ldw < p) info = -7;
  if (c == NULL && incc != 0) info = -9;
  if (x == NULL) info = -10;
  if (incx == 0) info = -11;
  if (workspace==NULL) info = -12;
  if (info != 0) {
    const char *func = __func__;
    egrss_err_param(func, info);
    return info;
  }

  /* Initialize workspace */
  double * restrict z = workspace;
  for (int k=0;k<p;k++) z[k] = 0.0;

  /* Matrix-vector product */
  if (c != NULL) {
    if (trans == 'N') {
      for (int i=0; i<n; i++) {
        /* Compute u_i'*z */
        double dot=0.0;
        for (int k=0;k<p;k++) {
          dot += U[k]*z[k];
        }

        /* Update z */
        for (int k=0;k<p;k++) {
          z[k] += W[k]*(*x);
        }

        /* Update x_i */
        *x = (*c)*(*x) + dot; /* x_i = d_i*x_i + u_i'*z */

        U += ldu; W += ldw;
        x += incx; c += incc;
      }
    } else {
      /* trans == 'T' */
      U += (n-1)*ldu; W += (n-1)*ldw;
      x += (n-1)*incx; c += (n-1)*incc;
      for (int i=0; i<n; i++) {
        /* Compute v_i'*z */
        double dot=0.0;
        for (int k=0;k<p;k++) {
          dot += W[k]*z[k];
        }

        /* Update z */
        for (int k=0;k<p;k++) {
          z[k] += U[k]*(*x);
        }

        /* Update x_i */
        *x = (*c)*(*x) + dot; /* x_i = d_i*x_i + v_i'*z */

        U -= ldu; W -= ldw;
        x -= incx; c -= incc;
      }
    }
  } else {
    /* d == NULL */
    if (trans == 'N') {
      for (int i=0;i<n;i++) {
        /* Update z */
        for (int k=0;k<p;k++) {
          z[k] += W[k]*(*x);
        }

        /* Compute x_i = u_i'*z */
        *x = 0.0;
        for (int k=0;k<p;k++) {
          *x += U[k]*z[k];
        }

        U += ldu; W += ldw;
        x += incx;
      }
    } else {
      /* trans == 'T' */
      U += (n-1)*ldu; W += (n-1)*ldw; x += (n-1)*incx;
      for (int i=0;i<n;i++) {
        /* Update z */
        for (int k=0;k<p;k++) {
          z[k] += U[k]*(*x);
        }

        /* Compute x_i = v_i'*z */
        *x = 0.0;
        for (int k=0;k<p;k++) {
          *x += W[k]*z[k];
        }

        U -= ldu; W -= ldw;
        x -= incx;
      }
    }
  }

  return info;
}
