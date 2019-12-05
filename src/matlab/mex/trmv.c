#include "mex.h"
#include "egrss.h"
#include <math.h>
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{

   /* Check for proper number of arguments. */
   if ( nrhs > 5 ) {
      mexErrMsgIdAndTxt("egrss:trmv:invalidNumInputs","Too many input arguments.");
  	}
  	else if (nrhs < 3) {
  		mexErrMsgIdAndTxt("egrss:trmv:invalidNumInputs","Not enough input arguments.");
  	}
  	if (nlhs > 1) {
  		mexErrMsgIdAndTxt("egrss:trmv:invalidNumOutputs","Invalid number of output arguments.");
  	}

    /* Check input types */
  	if ( !mxIsDouble(prhs[0]) || !mxIsDouble(prhs[1]) || !mxIsDouble(prhs[2]) ) {
  		mexErrMsgIdAndTxt("egrss:trmv:invalidInput","Inputs must be real-valued numeric arrays.");
  	}

  	mwSize p = mxGetM(prhs[0]);
  	mwSize n = mxGetN(prhs[0]);

    /* Validate input arguments */
    if (p != mxGetM(prhs[1]) || n != mxGetN(prhs[1])) {
      mexErrMsgIdAndTxt("egrss:trmv:matchdims","Dimensions of Ut and Wt do not match.");
    }
    char trans = 'N';
    double * c = NULL;
    int inc = 0;
    if (nrhs == 3) {
      if (!(mxGetM(prhs[2]) == n && mxGetN(prhs[2]) == 1)) {
        mexErrMsgIdAndTxt("egrss:trmv:matchdims","Dimensions of x are invalid.");
    	 }
    }
    else if (nrhs == 4) {
      if (mxIsChar(prhs[3])){
        trans = *mxGetChars(prhs[3]);
        if (mxGetNumberOfElements(prhs[3]) != 1 || (trans != 'N' && trans != 'T')) {
          mexErrMsgIdAndTxt("egrss:trmv:invalidInput","Invalid input: expected 'N' or 'T'.");
        }
      }
      else if (mxIsDouble(prhs[3])) {
        if (!(mxGetM(prhs[3]) == n && mxGetN(prhs[3]) == 1)) {
          mexErrMsgIdAndTxt("egrss:trmv:matchdims","Dimensions of x are invalid.");
        }
        c = mxGetPr(prhs[2]);
        inc = 1;
      }
      else {
        mexErrMsgIdAndTxt("egrss:trmv:invalidInput","Invalid input: expected real-valued numeric array or a charater.");
      }
    }
    else if (nrhs == 5) {
      if (mxIsChar(prhs[4])){
         trans = *mxGetChars(prhs[4]);
         if (mxGetNumberOfElements(prhs[4]) != 1 || (trans != 'N' && trans != 'T')) {
           mexErrMsgIdAndTxt("egrss:trmv:invalidInput","Invalid input: expected 'N' or 'T'.");
         }
      }
      else {
        mexErrMsgIdAndTxt("egrss:trmv:invalidInput","Invalid input: expected 'N' or 'T'.");
      }
      if (mxIsDouble(prhs[3])) {
        if (!(mxGetM(prhs[3]) == n && mxGetN(prhs[3]) == 1)) {
          mexErrMsgIdAndTxt("egrss:trmv:matchdims","Dimensions of x are invalid.");
        }
        c = mxGetPr(prhs[2]);
        inc = 1;
      }
      else {
        mexErrMsgIdAndTxt("egrss:trmv:invalidInput","Invalid input: expected real-valued numeric array.");
      }
    }

  	/* Allocate/initialize output array b */
   plhs[0] = mxDuplicateArray(c == NULL ? prhs[2] : prhs[3]);
   double * b = mxGetPr(plhs[0]);

  	/* Allocate workspace */
  	mxArray * work = mxCreateDoubleMatrix(p,1,mxREAL);
   double * ws = mxGetPr(work);

  	/* Compute matrix-vector product and clean up */
  	int info = egrss_dtrmv(trans,p,n,mxGetPr(prhs[0]),p,mxGetPr(prhs[1]),p,c,inc,b,1,ws);
  	mxDestroyArray(work);
  	if (info) {
  		mexErrMsgIdAndTxt("egrss:trmv:failure","Factorization failed.");
  	}
}
