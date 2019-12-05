function [Wt,c] = egrss_ldl(Ut,Vt,d)
% EGRSS_POTRF   Computes implicit LDL factorization of the sum of a
% symmetric extended generator representable semiseparable matrix and a
% diagonal matrix.
%
% [Wt,c] = EGRSS_LDL(Ut,Vt) computes a matrix Wt and a vector c such that the LDL
% factorization of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1) + diag(d)
% is given by L = tril(Ut'*Wt,-1)+I and D = diag(c), i.e., A = L*D*L'.
%
% [Wt,c] = EGRSS_LDL(Ut,Vt,d) computes a matrix Wt and a vector c such that the
% LDL factorization of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1) is
% given by L = tril(Ut'*Wt,-1)+eye(size(Ut,2)) and D = diag(c), i.e., A = L*D*L'.
% The input d must either be a vector of length size(Ut,2) or a scalar; in the
% latter case diag(d) is interpreted as the identity matrix scaled by d.
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
else
    c = zeros(n,1);
end

for k = 1:n
    Wt(:,k) = Wt(:,k)-P*Ut(:,k);
    c(k) = Ut(:,k)'*Wt(:,k) + c(k);
    P = P + (Wt(:,k)*Wt(:,k)')/c(k);
    Wt(:,k) = Wt(:,k)/c(k);
end

end
