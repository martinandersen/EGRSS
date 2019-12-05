#include "mex.h"
#include "egrss.h"
#include <math.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

     /* Check for proper number of arguments. */
     if ( nrhs > 5 ) {
        mexErrMsgIdAndTxt("egrss:gemv:invalidNumInputs","Too many input arguments.");
    	}
    	else if (nrhs < 5) {
    		mexErrMsgIdAndTxt("egrss:gemv:invalidNumInputs","Not enough input arguments.");
    	}
    	if (nlhs > 1) {
    		mexErrMsgIdAndTxt("egrss:gemv:invalidNumOutputs","Invalid number of output arguments.");
    	}

      /* Check input types */
      for (int i=0;i<5;i++) {
          if (!mxIsDouble(prhs[i]))
            mexErrMsgIdAndTxt("egrss:gemv:invalidInput","Inputs must be real-valued numeric arrays.");
      }

    	mwSize p = mxGetM(prhs[0]);
    	mwSize m = mxGetN(prhs[0]);
     mwSize q = mxGetM(prhs[3]);
    	mwSize n = mxGetN(prhs[3]);

     mwSize min_mn = m>n ? n : m;

      /* Validate input arguments */
      if (p != mxGetM(prhs[1]) || (mxGetN(prhs[1]) != min_mn && mxGetN(prhs[1]) != n)) {
        mexErrMsgIdAndTxt("egrss:gemv:matchdims","Dimensions of Ut and Vt do not match.");
      }
      if (q != mxGetM(prhs[2]) || (mxGetN(prhs[2]) != min_mn && mxGetN(prhs[2]) != m)) {
        mexErrMsgIdAndTxt("egrss:gemv:matchdims","Dimensions of Pt and Qt do not match.");
      }

      if (!(mxGetM(prhs[4]) == n || mxGetN(prhs[4]) == 1)) {
        mexErrMsgIdAndTxt("egrss:gemv:matchdims","Dimensions of x are invalid.");
    	 }

    	/* Allocate/initialize output array b */
      double * b;
      if (m == n) {
        plhs[0] = mxDuplicateArray(prhs[4]);
        b = mxGetPr(plhs[0]);
      }
      else {
        plhs[0] = mxCreateDoubleMatrix(m,1,mxREAL);
        b = mxGetPr(plhs[0]);
        double *x = mxGetPr(prhs[4]);
        for (int i=0;i<min_mn;i++) {b[i] = x[i];}
        for (int i=min_mn;i<m;i++) {b[i] = 0.0;}
      }

    	/* Allocate workspace */
    	mxArray * work = mxCreateDoubleMatrix(p+q,1,mxREAL);
     double * ws = mxGetPr(work);

    	/* Compute matrix-vector product and clean up */
      double * U = mxGetPr(prhs[0]);
      double * V = mxGetPr(prhs[1]);
      double * P = mxGetPr(prhs[2]);
      double * Q = mxGetPr(prhs[3]);
    	int info = egrss_dgemv(p,q,min_mn,U,p,V,p,P,q,Q,q,b,1,ws);
      /* If m > n: reuse first p elements of workspace from egrss_gemv() */
      for (int i=n;i<m;i++) {
          for (int k=0;k<p;k++) {
            b[i] += U[p*i+k]*ws[k];
          }
      }
      /* Update b with low rank term */
      if (m < n) {
        double * x = mxGetPr(prhs[4]);
        for (int k=0;k<q;k++) ws[k] = 0.0;
        /* Compute ws = Qt(:,m+1:end)*x(m+1:end) */
        for (int i=m;i<n;i++) {
          for (int k=0;k<q;k++) {
            ws[k] += Q[i*q+k]*x[i];
          }
        }
        /* Compute b += Pt'*ws */
        for (int i=0;i<m;i++) {
          for (int k=0;k<q;k++) {
            b[i] += P[i*q+k]*ws[k];
          }
        }
      }
    	mxDestroyArray(work);
    	if (info) {
    		mexErrMsgIdAndTxt("egrss:gemv:failure","Unknown failure.");
    	}
}
