#include <stdio.h>
#include <stdlib.h>
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
  if (t == NULL || U == NULL || V == NULL) { ret_val = -1; goto cleanup;}
  for (int i=0;i<n;i++) { t[i] = (i+1.0)/n;}

  info = egrss_dsplkgr(p,n,U,p,V,p,t,1);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  info = egrss_dsplkgr(p,n,U,p,V,p,t+n-1,-1);
  if (info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  /* Check for failure when t is not monotonic */
  t[0] = 1.0;
  info = egrss_dsplkgr(p,n,U,p,V,p,t,1);
  if (!info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

  info = egrss_dsplkgr(p,n,U,p,V,p,t+n-1,-1);
  if (!info) {
    printf("%s: failed on line %d\n",__FILE__,__LINE__-2);
    printf("info = %d\n",info);
    ret_val = -1;
    goto cleanup;
  };

cleanup:
  free(t);
  free(U);
  free(V);
  return ret_val;
}
