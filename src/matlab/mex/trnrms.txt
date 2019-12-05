% EGRSS_TRNRMS   Computes the squared column norms of a lower triangular
% matrix of the form L = tril(Ut'*Wt,-1) + diag(c) where Ut and Wt are of size
% p-by-n (with p > 0), and c is a positive vector of length n.
%
% b = EGRSS_TRNRMS(Ut,Wt,c) computes a vector b and with entries b(i) =
% norm(L(:,i))^2 where L = tril(Ut'*Wt,-1) + diag(c).
%
% See also: EGRSS_POTRF, EGRSS_TRTRI
