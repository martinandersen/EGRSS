#include "mex.h"
#include "egrss.h"
#include <math.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

  /* Check for proper number of arguments. */
  if ( nrhs > 3 ) {
    mexErrMsgIdAndTxt("egrss:trnrms:invalidNumInputs","Too many input arguments.");
	}
	else if (nrhs < 3) {
		mexErrMsgIdAndTxt("egrss:trnrms:invalidNumInputs","Not enough input arguments.");
	}
	if (nlhs != 1) {
			mexErrMsgIdAndTxt("egrss:trnrms:invalidNumOutputs","Invalid number of output arguments.");
	}

  /* Check input types */
	if ( !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) ) {
		mexErrMsgIdAndTxt("egrss:trnrms:invalidInput","Inputs must be real-valued numeric arrays.");
	}

	mwSize p = mxGetM(prhs[0]);
	mwSize n = mxGetN(prhs[0]);

  /* Validate input arguments */
  if (p != mxGetM(prhs[1]) || n != mxGetN(prhs[1])) {
    mexErrMsgIdAndTxt("egrss:trnrms:matchdims","Dimensions of Ut and Vt do not match.");
  }
	if (!(mxGetM(prhs[2]) == 1 || mxGetN(prhs[2]) == 1) || mxGetNumberOfElements(prhs[2]) != n) {
    mexErrMsgIdAndTxt("egrss:trnrms:matchdims","Size of c is invalid.");
	}


	/* Allocate/initialize output array b */
  plhs[0] = mxCreateDoubleMatrix(n,1,mxREAL);

	/* Allocate workspace */
	mxArray * work = mxCreateDoubleMatrix(p,p,mxREAL);
  double * ws = mxGetPr(work);

	/* Compute row norms and clean up */
	int info = egrss_dtrnrms(p,n,mxGetPr(prhs[0]),p,mxGetPr(prhs[1]),p,mxGetPr(prhs[2]),1,mxGetPr(plhs[0]),1,ws);
	mxDestroyArray(work);
	if (info) {
		mexErrMsgIdAndTxt("egrss:trnrms:failure","Unknown failure.");
	}
}
