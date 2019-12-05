function b = egrss_gemv(Ut,Vt,Pt,Qt,x)
% EGRSS_GEMV   Computes matrix-vector product A*x where A is an
% extended generator representable semiseparable matrix given
% by A = tril(Ut'*Vt) + triu(Pt'*Qt,1).
%
% b = EGRSS_GEMV(Ut,Vt,Pt,Qt,x) computes b = A*x.
%

assert(size(Ut,1) == size(Vt,1),'Dimension mismatch: Ut and Vt must have the same number of rows.')
assert(size(Pt,1) == size(Qt,1),'Dimension mismatch: Pt and Qt must have the same number of rows.')
assert(isvector(x) && length(x) == size(Vt,2),'Dimension mismatch: x must be a vector of length size(V,2)')

m = size(Ut,2);
n = size(Qt,2);
min_mn = min(m,n);

assert(size(Vt,2) == min_mn || size(Vt,2) == n,'Dimension mismatch: Vt has incompatible size.')
assert(size(Pt,2) == min_mn || size(Pt,2) == m,'Dimension mismatch: Pt has incompatible size.')

b = [x(1:min_mn); zeros(max(0,m-n),1)];
v = zeros(size(Ut,1),1);
q = Qt(:,1:min_mn)*x(1:min_mn);

for k = 1:min_mn
    v = v + Vt(:,k)*b(k);
    q = q - Qt(:,k)*b(k);
    b(k) = Ut(:,k)'*v + Pt(:,k)'*q;
end

if n > m
    b = b + Pt'*(Qt(:,min_mn+1:end)*x(min_mn+1:end));
elseif m > n
    b(min_mn+1:end) = Ut(:,min_mn+1:end)'*(Vt*x);
end

end
