% EGRSS_SYMV   Computes matrix-vector product A*x where A is a symmetric
% and extended generator representable semiseparable matrix.
%
% b = EGRSS_SYMV(Ut,Vt,x) computes b = A*x where A is given by tril(Ut'*Vt) +
%     triu(Vt'*Ut,1).
%
% See also: EGRSS_GENERATORS
