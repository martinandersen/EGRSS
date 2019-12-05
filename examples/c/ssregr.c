#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "egrss.h"

/* BLAS/LAPACK prototypes */
void dtrsv_(char *uplo,char *trans,char *diag,int *n,double *A,int *lda,double *x,int *incx);
void dgelqf_(int *m, int *n, double *A, int *lda, double *tau, double *work, int *lwork, int *info);
void dormlq_(char *side, char *trans, int *m, int *n, int *k, double *A, int *lda, double *tau, double *C, int *ldc, double *work, int *lwork, int *info);

int main(int argc, char const *argv[]) {

  if (argc != 3) {
    fprintf(stderr,"Usage: %s p lambda\n",argv[0]);
    return EXIT_FAILURE;
  }

  int n=200;            // Number of observations

  int p=0;                 // Spline kernel order
  p = atoi(argv[1]);
  if (p<1 || p >= n) {
    fprintf(stderr, "p must be greater than or equal to 1 and less than n\n");
    return EXIT_FAILURE;
  }

  double lambda=0.0;       // Regularization parameter
  lambda = atof(argv[2]);
  if (lambda<=0.0) {
    fprintf(stderr, "lambda must be positive\n");
    return EXIT_FAILURE;
  }


  int retval = EXIT_SUCCESS;
  double *t=NULL, *y=NULL, *dcoef=NULL, *ccoef=NULL;
  double *Ut=NULL, *Wt=NULL, *ct=NULL, *Bt=NULL, *ws=NULL;

  /* Additional variables used for BLAS/LAPACK calls */
  int info;
  int iOne=1;
  double dOne=1.0, dNegOne=-1.0, dZero=0.0;
  char cL='L',cN='N',cT='T';

  /* Initialize random number generator */
  srand(time(NULL));

  /* Allocate arrays */
  Ut = malloc(sizeof(*Ut)*n*p);
  Wt = malloc(sizeof(*Wt)*n*p);
  Bt = malloc(sizeof(*Bt)*n*p);
  ct = malloc(sizeof(*ct)*n);
  t = malloc(sizeof(*t)*n);
  y = malloc(sizeof(*y)*n);
  ccoef = malloc(sizeof(*ccoef)*n);
  dcoef = malloc(sizeof(*dcoef)*p);
  ws = malloc(sizeof(*ws)*p*(p+1));
  if (Ut==NULL||Wt==NULL||Bt==NULL||ct==NULL||t==NULL||y==NULL||dcoef==NULL||ccoef==NULL||ws==NULL) {
    fprintf(stderr,"Error: memory allocation failed.\n");
    retval = -1;
    goto cleanup;
  }

  /* Initialize t and c */
  for (int k=0;k<n;k++) t[k] = (double)k/(n-1);
  for (int k=0;k<n;k++) ct[k] = lambda*n;

  /* Generate noisy measurements (uniform noise distr.) */
  for (int k=0;k<n;k++) {
    y[k] = cos(2.*M_PI*t[k]) + 0.3*sin(10.*M_PI*t[k]) - 0.5*t[k];
    y[k] += 0.2*((double)rand()/RAND_MAX-0.5);
  }

  /* Compute generators */
  if(egrss_dsplkgr(p,n,Ut,p,Wt,p,t,1)) {
    retval = EXIT_FAILURE;
    goto cleanup;
  }

  /* Compute Cholesky factorization (overwrites W) */
  if (egrss_dpotrf(p,n,Ut,p,Wt,p,ct,1,ws)) {
    retval = EXIT_FAILURE;
    goto cleanup;
  }

  /* Initialize and solve for Bt */
  memcpy(Bt,Ut,n*p*sizeof(*Ut));
  for (int k=0;k<p;k++) {
    egrss_dtrsv(cN,p,n,Ut,p,Wt,p,ct,1,Bt+k,p,ws);
  }

  /* LQ factorization: Bt = Lb*Q
     The last p elements of ws are used to store the scalar
     factors of the elementary reflectors */
  int lwork=p*p;
  dgelqf_(&p, &n, Bt, &p, ws+lwork, ws, &lwork, &info);
  if (info) {
    fprintf(stderr,"Error: LQ factorization failed.\n");
    retval = EXIT_FAILURE;
    goto cleanup;
  }

  /* Solve L*ccoef = y */
  memcpy(ccoef,y,n*sizeof(*y));
  egrss_dtrsv(cN,p,n,Ut,p,Wt,p,ct,1,ccoef,1,ws);

  /* Compute Q*ccoef */
  dormlq_(&cL, &cN, &n, &iOne, &p, Bt, &p, ws+lwork, ccoef, &n, ws, &lwork, &info);
  if (info) {
    fprintf(stderr,"Error: orthogonal transformation failed.\n");
    retval = EXIT_FAILURE;
    goto cleanup;
  }

  /* Copy first p elements of ccoef to dcoef,
     and set first p elements of ccoef to zero */
  memcpy(dcoef,ccoef,p*sizeof(*ccoef));
  for (int k=0;k<p;k++) ccoef[k]=0.0;

  /* Compute Q'*ccoef */
  dormlq_(&cL, &cT, &n, &iOne, &p, Bt, &p, ws+lwork, ccoef, &n, ws, &lwork, &info);
  if (info) {
    fprintf(stderr,"Error: orthogonal transformation failed.\n");
    retval = EXIT_FAILURE;
    goto cleanup;
  }

  /* Solve L'*x = ccoef */
  egrss_dtrsv(cT,p,n,Ut,p,Wt,p,ct,1,ccoef,1,ws);

  /* Solve Lb'*x = dcoef */
  dtrsv_(&cL,&cT,&cN,&p,Bt,&p,dcoef,&iOne);

  /* Compute log. GML and estimate of noise variance */
  double log_gml = 0.0;
  double nvar = 0.0;
  for (int k=0;k<n;k++) {
    nvar += ccoef[k]*y[k];
  }
  log_gml += log(nvar);
  nvar *= n*lambda/(n-p);
  for (int k=0;k<p;k++) log_gml += 2.0/(n-p)*log(fabs(Bt[k*(p+1)]));
  for (int k=0;k<n;k++) log_gml += 2.0/(n-p)*log(ct[k]);

  /* Print result */
  printf("Log. GML (up to const): %.2e\n",log_gml);
  printf("GML est. of noise std.: %.2e\n\n",sqrt(nvar));

  //printf("%12s %12s %12s\n","t","y","yhat");
  //for (int k=0;k<n;k++) {
  //  printf("% 12.4e % 12.4e % 12.4e\n",t[k],y[k],y[k]-n*lambda*ccoef[k]);
  //}

cleanup:
  free(Ut);
  free(Wt);
  free(Bt);
  free(ct);
  free(t);
  free(ws);
  free(dcoef);
  free(ccoef);
  free(y);

  return retval;
}
