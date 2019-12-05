#include "mex.h"
#include "egrss.h"
#include <math.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

   /* Check for proper number of arguments. */
   if ( nrhs > 3 ) {
      mexErrMsgIdAndTxt("egrss:symv:invalidNumInputs","Too many input arguments.");
  	}
  	else if (nrhs < 3) {
  		mexErrMsgIdAndTxt("egrss:symv:invalidNumInputs","Not enough input arguments.");
  	}
  	if (nlhs != 1) {
  		mexErrMsgIdAndTxt("egrss:symv:invalidNumOutputs","Invalid number of output arguments.");
  	}

    /* Check input types */
  	if ( !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) ) {
  		mexErrMsgIdAndTxt("egrss:symv:invalidInput","Inputs must be real-valued numeric arrays.");
  	}

  	mwSize p = mxGetM(prhs[0]);
  	mwSize n = mxGetN(prhs[0]);

    /* Validate input arguments */
    if (p != mxGetM(prhs[1]) || n != mxGetN(prhs[1])) {
      mexErrMsgIdAndTxt("egrss:symv:matchdims","Dimensions of Ut and Vt do not match.");
    }
    if (!(mxGetM(prhs[2]) == n || mxGetN(prhs[2]) == 1)) {
      mexErrMsgIdAndTxt("egrss:symv:matchdims","Dimensions of x are invalid.");
  	 }

  	/* Allocate/initialize output array B */
   plhs[0] = mxDuplicateArray(prhs[2]);
  	double * b = mxGetPr(plhs[0]);

  	/* Allocate workspace */
  	mxArray * work = mxCreateDoubleMatrix(2*p,1,mxREAL);
   double * ws = mxGetPr(work);

  	/* Compute matrix-vector product and clean up */
  	int info = egrss_dsymv(p,n,mxGetPr(prhs[0]),p,mxGetPr(prhs[1]),p,b,1,ws);
  	mxDestroyArray(work);
  	if (info) {
  		mexErrMsgIdAndTxt("egrss:symv:failure","Unknown failure.");
  	}
}
