function [b] = egrss_trnrms(Ut,Wt,c)
% EGRSS_TRNRMS   Computes the squared column norms of a lower triangular
% matrix of the form L = tril(Ut'*Wt,-1) + diag(c) where Ut and Wt are of size
% p-by-n (with p > 0), and c is a positive vector of length n.
%
% b = EGRSS_TRNRMS(Ut,Wt,c) computes a vector b and with entries b(i) =
% norm(L(:,i))^2 where L = tril(Ut'*Wt,-1) + diag(c).
%
% See also: EGRSS_POTRF, EGRSS_TRTRI


assert(all(size(Ut) == size(Wt)),'Dimension mismatch: U and W must be of the same size.')
assert(length(c) == size(Ut,2) && isvector(c),'Dimension mismatch: c must be a vector of length size(U,2)')

[p,n] = size(Ut);

P = zeros(p);
b = zeros(n,1);
for k = n:-1:1
    b(k) = c(k)^2 + Wt(:,k)'*P*Wt(:,k);
    P = P + Ut(:,k)*Ut(:,k)';
end
