% EGRSS_TRSV   Solves the equation L*x = b or L'*x = b where L is a lower
% triangular extended generator representable semiseparable or quasi-separable
% matrix defined in terms of matrices Ut and Wt, which are of size p-by-n
% (with p > 0), and in the quasi-separable case, a vector c of length n.
%
% x = EGRSS_TRSV(Ut,Wt,b) solves L*x = b with L = tril(Ut'*Wt).
%
% x = EGRSS_TRSV(Ut,Wt,b,'N') is the same as x = EGRSS_TRSV(Ut,Wt,b).
%
% x = EGRSS_TRSV(Ut,Wt,b,'T') solves L'*x = b with L = tril(Ut'*Wt).
%
% x = EGRSS_TRSV(Ut,Wt,c,b) solves L*x = b with L = tril(Ut'*Wt,-1) + diag(c).
%
% x = EGRSS_TRSV(Ut,Wt,c,b,'N') is the same as x = EGRSS_TRSV(Ut,Wt,c,b).
%
% x = EGRSS_TRSV(Ut,Wt,c,b,'T') solves L'*x = b with L = tril(Ut'*Wt,-1) + diag(c).
%
% See also: EGRSS_POTRF, EGRSS_TRMV
