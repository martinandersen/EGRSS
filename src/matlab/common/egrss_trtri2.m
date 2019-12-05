function [Linv] = egrss_trtri2(Ut,Wt,c)
% EGRSS_TRTRI2   Computes an explicit inverse of a lower triangular matrix
% of the form L = tril(Ut'*Wt,-1) + diag(c).
%
% Linv = EGRSS_TRTRI2(Ut,Wt,c) forms the dense matrix inv(L) from Ut, Wt, and c.
%
% See also: EGRSS_POTRF

assert(all(size(Ut) == size(Wt)),'Dimension mismatch: Ut and Wt must be of the same size.')
assert(length(c) == size(Ut,2) && isvector(c),'Dimension mismatch: c must be a vector of length size(Ut,2)')

[p,n] = size(Ut);
Linv = zeros(n,n);
for k = 1:n
    Linv(k:end,k) = egrss_trsv(Ut(:,k:end),Wt(:,k:end),c(k:end),[1;zeros(n-k,1)]);
end

end
