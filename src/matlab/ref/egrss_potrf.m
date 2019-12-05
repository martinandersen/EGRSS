function [Wt,c] = egrss_potrf(Ut,Vt,d)
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

assert(all(size(Ut) == size(Vt)),'Dimension mismatch: U and V must be of the same size.')

[p,n] = size(Ut);
Wt = Vt;
P = zeros(p);

if nargin == 3
    assert((length(d) == size(Ut,2) && isvector(d)) || isscalar(d),'Dimension mismatch: d must be a vector of length size(Ut,2) or a scalar.')
    if isscalar(d)
        c = d*ones(size(Ut,2),1);
    else
        c = d;
    end
    for k = 1:n
        Wt(:,k) = Wt(:,k)-P*Ut(:,k);
        c(k) = realsqrt(Ut(:,k)'*Wt(:,k) + c(k));
        Wt(:,k) = Wt(:,k)/c(k);
        P = P + Wt(:,k)*Wt(:,k)';
    end
else
    for k = 1:n
        Wt(:,k) = Wt(:,k)-P*Ut(:,k);
        Wt(:,k) = Wt(:,k)/realsqrt(Ut(:,k)'*Wt(:,k));
        P = P + Wt(:,k)*Wt(:,k)';
    end

end
end
