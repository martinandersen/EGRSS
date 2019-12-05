% EGRSS_POTRF   Computes implicit Cholesky factorization of the sum of a
% symmetric extended generator representable semiseparable matrix and a
% diagonal matrix.
%
% Wt = EGRSS_POTRF(Ut,Vt) computes a matrix Wt such that L = tril(Ut'*Wt) is
% the Cholesky factor of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1),
% i.e., A = L*L'.
%
% [Wt,c] = EGRSS_POTRF(Ut,Vt,d) computes a matrix Wt and a vector c such
% that L = tril(Ut'*Wt,-1) + diag(c) is the Cholesky factor of the symmetric
% matrix A = tril(Ut'*Vt) + triu(Vt'*Ut,1) + diag(d), i.e., A = L*L'. The input
% d must either be a vector of length size(Ut,2) or a scalar; in the latter case
% diag(d) is interpreted as the identity matrix scaled by d.
%
% See also: EGRSS_TRMV, EGRSS_TRSV
