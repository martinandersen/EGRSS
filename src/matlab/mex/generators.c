#include "mex.h"
#include "egrss.h"
#include <math.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

   /* Check for proper number of arguments. */
	if ( nrhs > 2 ) {
 		mexErrMsgIdAndTxt("egrss:generators:invalidNumInputs","Too many input arguments.");
	}
	else if (nrhs < 2) {
		mexErrMsgIdAndTxt("egrss:generators:invalidNumInputs","Not enough input arguments.");
	}
	if (nlhs != 2) {
		mexErrMsgIdAndTxt("egrss:potrf:invalidNumOutputs","Invalid number of output arguments.");
	}

	/* Validate input arguments */
   if (!mxIsDouble(prhs[0]) || !(mxGetM(prhs[0]) == 1 || mxGetN(prhs[0]) == 1)) {
	 	mexErrMsgIdAndTxt("egrss:generators:matchdims","Input t must be a vector.");
	}
   if (!mxIsScalar(prhs[1]) || mxGetScalar(prhs[1]) < 1 || (mxGetScalar(prhs[1]) != (int)mxGetScalar(prhs[1]))) {
		mexErrMsgIdAndTxt("egrss:generators:invalidInput","Input p must be a positive integer.");
	}

	mwSize n = mxGetM(prhs[0]) > mxGetN(prhs[0]) ? mxGetM(prhs[0]) : mxGetN(prhs[0]);
	mwSize p = (mwSize) mxGetScalar(prhs[1]);
  double * t = mxGetPr(prhs[0]);

  /* Allocate output arrays Ut and Vt */
 	plhs[0] = mxCreateDoubleMatrix(p,n,mxREAL);
 	plhs[1] = mxCreateDoubleMatrix(p,n,mxREAL);

	/* Compute SS generators */
	int info;
	info = egrss_dsplkgr(p,n,mxGetPr(plhs[0]),p,mxGetPr(plhs[1]),p,t,1);
	if (info > 0) {
		mexErrMsgIdAndTxt("egrss:generators:failure","Input t must be monotonic.");
	}

}
