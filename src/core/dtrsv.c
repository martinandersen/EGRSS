#include <stdlib.h>
#include <math.h>
#include "egrss.h"

/**
\author Martin S. Andersen
\copyright BSD-2-clause

\brief Solves the a system of equations with a lower/upper triangular extended generator representable p-quasiseparable matrix.

The function computes the matrix-vector product \f$x \gets L^{-1}x\f$ (if \b trans is \c N) or \f$x \gets L^{-T}x\f$ (if \b trans is \c T) where \f$L\f$ is either of the form
\f[L = \mathrm{tril}(UW^T)\f]
(if \b c is \c NULL and \b incc is \c 0) or
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
@param[in,out] b  array of length n
@param[in] incb   stride of b
@param[out] workspace   array of length at least p

@return  The function returns the value \c 0 if successful. A return value \c INFO < 0 indicates that input argument \c -INFO is invalid.

@see egrss_dpotrf egrss_dtrmv
*/
int egrss_dtrsv(
  const char trans,
  const int p,
  const int n,
  const double *restrict U,
  const int ldu,
  const double *restrict W,
  const int ldw,
  const double *restrict c,
  const int incc,
  double *restrict b,
  const int incb,
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
    if (b == NULL) info = -10;
    if (incb == 0) info = -11;
    if (workspace==NULL) info = -12;
    if (info != 0) {
      const char *func = __func__;
      egrss_err_param(func, info);
      return info;
    }

    /* Initialize workspace */
    for (int k=0;k<p;k++) workspace[k] = 0.0;
    double * z = workspace;

    /* Solve L*x = b or L'*x = b */
    if (c != NULL) {
      if (trans == 'N') {
        for (int i=0; i<n; i++) {
          /* Compute u_i'*z */
          double dot=0.0;
          for (int k=0;k<p;k++) {
            dot += U[k]*z[k];
          }

          /* Update b_i := (b_i-u_i'*z)/c_i */
          *b = ((*b)-dot)/(*c);

          /* Update z */
          for (int k=0;k<p;k++) {
            z[k] += W[k]*(*b);
          }

          U += ldu; W += ldw;
          b += incb; c += incc;
        }
      } else {
        /* trans == 'T' */
        U += (n-1)*ldu; W += (n-1)*ldw;
        b += (n-1)*incb; c += (n-1)*incc;
        for (int i=0; i<n; i++) {
          /* Compute v_i'*z */
          double dot=0.0;
          for (int k=0;k<p;k++) {
            dot += W[k]*z[k];
          }

          /* Update b_i */
          *b = ((*b) - dot)/(*c);

          /* Update z */
          for (int k=0;k<p;k++) {
            z[k] += U[k]*(*b);
          }

          U -= ldu; W -= ldw;
          b -= incb; c -= incc;
        }
      }
    } else {
      /* d == NULL */
      if (trans == 'N') {
        for (int i=0;i<n;i++) {

          /* Compute b_i := (b_i - u_i'*z)/(v_i'*u_i) */
          double dot = 0.0;
          for (int k=0;k<p;k++) {
            *b -= U[k]*z[k];
            dot += U[k]*W[k];
          }
          *b /= dot;

          /* Update z */
          for (int k=0;k<p;k++) {
            z[k] += W[k]*(*b);
          }

          U += ldu; W += ldw;
          b += incb;
        }
      } else {
        /* trans == 'T' */
        U += (n-1)*ldu; W += (n-1)*ldw;
        b += (n-1)*incb;
        for (int i=0;i<n;i++) {

          /* Compute b_i := (b_i - u_i'*z)/(v_i'*u_i) */
          double dot = 0.0;
          for (int k=0;k<p;k++) {
            *b -= W[k]*z[k];
            dot += U[k]*W[k];
          }
          *b /= dot;

          /* Update z */
          for (int k=0;k<p;k++) {
            z[k] += U[k]*(*b);
          }

          U -= ldu; W -= ldw;
          b -= incb;
        }
      }
    }

    return info;
  }
