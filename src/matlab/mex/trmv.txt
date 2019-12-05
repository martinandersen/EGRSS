% EGRSS_TRMV   Computes matrix-vector product L*x or L'*x where L is a lower
% triangular extended generator representable semiseparable or quasi-separable
% matrix defined in terms of matrices Ut and Wt, which are of size p-by-n
% (with p > 0), and in the quasi-separable case, a vector c of length n.
%
% b = EGRSS_TRMV(Ut,Wt,x) computes b = L*x with L = tril(Ut'*Wt).
%
% b = EGRSS_TRMV(Ut,Wt,x,'N') is the same as b = EGRSS_TRMV(Ut,Wt,x).
%
% b = EGRSS_TRMV(Ut,Wt,x,'T') computes b = L'*x with L = tril(Ut'*Wt).
%
% b = EGRSS_TRMV(Ut,Wt,c,x) computes b = L*x with L = tril(Ut'*Wt,-1) + diag(c).
%
% b = EGRSS_TRMV(Ut,Wt,c,x,'N') is the same as b = EGRSS_TRMV(Ut,Wt,c,x).
%
% b = EGRSS_TRMV(Ut,Wt,c,x,'T') computes b = L'*x with L = tril(Ut'*Wt,-1) + diag(c).
%
% See also: EGRSS_POTRF, EGRSS_TRSV
