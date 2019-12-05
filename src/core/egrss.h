/**
  \file egrss.h
  \author Martin S. Andersen
  \copyright BSD-2-clause
*/

#ifndef __egrss_h__
#define __egrss_h__

void egrss_err_param(const char * func, int info);

int egrss_dsplkgr(
    const int p,
    const int n,
    double *restrict U,
    const int ldu,
    double *restrict V,
    const int ldv,
    const double *restrict t,
    const int inct
);

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
);

int egrss_dgemv(
  const int p,
  const int q,
  const int n,
  const double *restrict U,
  const int ldu,
  const double *restrict V,
  const int ldv,
  const double *restrict P,
  const int ldp,
  const double *restrict Q,
  const int ldq,
  double *restrict x,
  const int incx,
  double *restrict workspace
);

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
);

int egrss_dtrmv(
  const char trans,
  const int p,
  const int n,
  const double *restrict U,
  const int ldu,
  const double *restrict W,
  const int ldw,
  const double *restrict d,
  const int incd,
  double *restrict x,
  const int incx,
  double *restrict workspace
);

int egrss_dtrsv(
  const char trans,
  const int p,
  const int n,
  const double *restrict U,
  const int ldu,
  const double *restrict W,
  const int ldw,
  const double *restrict d,
  const int incd,
  double *restrict b,
  const int incb,
  double *restrict workspace
);

int egrss_dtrnrms(
  const int p,
  const int n,
  const double *restrict U,
  const int ldu,
  const double *restrict W,
  const int ldw,
  const double *restrict d,
  const int incd,
  double *restrict x,
  const int incx,
  double *restrict workspace
);

#endif
