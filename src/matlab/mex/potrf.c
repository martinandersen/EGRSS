#include "mex.h"
#include "egrss.h"
#include <math.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  /* Check for proper number of arguments. */
  if ( nrhs > 3 ) {
    mexErrMsgIdAndTxt("egrss:potrf:invalidNumInputs","Too many input arguments.");
	}
	else if (nrhs < 2) {
		mexErrMsgIdAndTxt("egrss:potrf:invalidNumInputs","Not enough input arguments.");
	}
	if ((nrhs == 3 && nlhs != 2) || nlhs > 2) {
			mexErrMsgIdAndTxt("egrss:potrf:invalidNumOutputs","Invalid number of output arguments.");
	}

  /* Check input types */
	if ( !mxIsDouble(prhs[0]) ||
    	 !mxIsDouble(prhs[1]) ||
			 (nrhs == 3 && !mxIsDouble(prhs[2])) ) {
		mexErrMsgIdAndTxt("egrss:potrf:invalidInput","Inputs must be real-valued numeric arrays.");
	}

	mwSize p = mxGetM(prhs[0]);
	mwSize n = mxGetN(prhs[0]);

  /* Validate input arguments */
  if (p != mxGetM(prhs[1]) || n != mxGetN(prhs[1])) {
    mexErrMsgIdAndTxt("egrss:potrf:matchdims","Dimensions of Ut and Vt do not match.");
  }
	if (nrhs == 3 && !(mxGetM(prhs[2]) == 1 || mxGetN(prhs[2]) == 1)) {
    mexErrMsgIdAndTxt("egrss:potrf:matchdims","Size of d is invalid.");
	}
  if (nrhs == 3 && (mxGetNumberOfElements(prhs[2]) != n && mxGetNumberOfElements(prhs[2]) != 1)) {
    mexErrMsgIdAndTxt("egrss:potrf:matchdims","Length of d is invalid.");
  }

	/* Allocate/initialize output array W */
  plhs[0] = mxDuplicateArray(prhs[1]);
	double * c = NULL;
	int inc = 0;
  if (nlhs == 2) {
		/* Allocate/initialize output array c */
		if (mxGetNumberOfElements(prhs[2]) == n) {
			plhs[1] = mxDuplicateArray(prhs[2]);
			c = mxGetPr(plhs[1]);
		}
		else {
			/* Input is a scalar */
			plhs[1] = mxCreateDoubleMatrix(n,1,mxREAL);
			c = mxGetPr(plhs[1]);
			double val = mxGetScalar(prhs[2]);
			for (int i=0;i<n;i++) c[i] = val;
		}
		inc = 1;
	}

	/* Allocate workspace */
	mxArray * work = mxCreateDoubleMatrix(p,p,mxREAL);
  double * ws = mxGetPr(work);

	/* Compute factorization and clean up */
	int info = egrss_dpotrf(p,n,mxGetPr(prhs[0]),p,mxGetPr(plhs[0]),p,c,inc,ws);
	mxDestroyArray(work);
	if (info) {
		mexErrMsgIdAndTxt("egrss:potrf:failure","Factorization failed.");
	}
}
