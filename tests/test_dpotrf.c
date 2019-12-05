#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "egrss.h"

#ifndef P
#define P 3
#endif
#ifndef N
#define N 100
#endif

int main(void) {

  int ret_val = 0, info;
  int p = P, n = N;
  double * t = malloc(n*sizeof(*t));
  double * U = malloc(p*n*sizeof(*U));
  double * V = malloc(p*n*sizeof(*V));
  double * W = malloc(p*n*sizeof(*W));
  double * d = malloc(n*sizeof(*d));
  double * ws = malloc(p*p*sizeof(*ws));
  if (t == NULL || U == NULL || V == NULL || W == NULL || ws == NULL) {
    ret_val = -1;
    goto cleanup;
  }

  /* Compute spline kernel generators */
  for (int i=0;i<n;i++) { t[i] = (i+1.0)/n;}
  info = egrss_dsplkgr(p,n,U,p,V,p,t,1);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  /* Test 1 (verify success) */
  memcpy(W,V,p*n*sizeof(*V));
  info = egrss_dpotrf(p,n,U,p,W,p,NULL,0,ws);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  /* Test 2 (verify success) */
  for (int i=0;i<n;i++) {d[i] = 1e-6;}
  memcpy(W,V,p*n*sizeof(*V));
  info = egrss_dpotrf(p,n,U,p,W,p,d,1,ws);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };


  /* Compute new spline kernel generators (singular kernel matrix) */
  for (int i=0;i<n;i++) { t[i] = (double)i/(n-1); }
  info = egrss_dsplkgr(p,n,U,p,V,p,t,1);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  /* Test 3 (verify failure) */
  memcpy(W,V,p*n*sizeof(*V));
  info = egrss_dpotrf(p,n,U,p,W,p,NULL,0,ws);
  if (!info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  /* Test 4 (verify success) */
  for (int i=0;i<n;i++) {d[i] = 1e-6;}
  memcpy(W,V,p*n*sizeof(*V));
  info = egrss_dpotrf(p,n,U,p,W,p,d,1,ws);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  /* Test 5 (verify success) */
  memcpy(W,V,p*n*sizeof(*V));
  info = egrss_dpotrf(p,n-1,U+p,p,W+p,p,NULL,0,ws);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  /* Test 6 (verify success) */
  for (int i=0;i<n;i++) {d[i] = 1e-6;}
  memcpy(W,V,p*n*sizeof(*V));
  info = egrss_dpotrf(p,n/2,U,2*p,W,2*p,d,2,ws);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };


cleanup:
  free(t);
  free(U);
  free(V);
  free(W);
  free(d);
  free(ws);
  return ret_val;

}
