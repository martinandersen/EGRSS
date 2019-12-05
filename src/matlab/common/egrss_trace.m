function b = egrss_trace(Ut,Vt,d,Yt,Zt,c)
% EGRSS_TRACE   Computes the trace of inv(K+diag(d))*K where K is a
% symmetric extended generator representable semiseparable matrix.
%
% b = EGRSS_TRACE(Ut,Vt,Yt,Zt,c) computes trace(inv(K+diag(d))*K) where K is
% symmetric and satisfies tril(K) = tril(Ut'*Vt), and K+diag(d) = L*L' where
% L = diag(Ut'*Wt,-1) + diag(c), and inv(L) = tril(Yt'*Zt,-1) + diag(1./c).
%
% See also: EGRSS_POTRF, EGRSS_TRTRI

assert(all(size(Ut) == size(Vt)),'Dimension mismatch: Ut and Vt must be of the same size.')
assert(all(size(Ut) == size(Yt)),'Dimension mismatch: Ut and Yt must be of the same size.')
assert(all(size(Ut) == size(Zt)),'Dimension mismatch: Ut and Zt must be of the same size.')
assert(length(c) == size(Ut,2) && isvector(c),'Dimension mismatch: c must be a vector of length size(Ut,2)')
assert(length(d) == size(Ut,2) && isvector(d),'Dimension mismatch: d must be a vector of length size(Ut,2)')

[p,n] = size(Ut);

P = zeros(p);
R = zeros(p);
b = 0;
for k = 1:n
    b = b + Yt(:,k)'*P*Yt(:,k) + 2*(Yt(:,k)'*R*Ut(:,k))/c(k) + (Ut(:,k)'*Vt(:,k) + d(k))/c(k)^2;
    P = P + (Ut(:,k)'*Vt(:,k) + d(k))*(Zt(:,k)*Zt(:,k)') + Zt(:,k)*(R*Ut(:,k))' + (R*Ut(:,k))*Zt(:,k)';
    R = R + Zt(:,k)*Vt(:,k)';
end

end
