function [Yt,Zt] = egrss_trtri(Ut,Wt,c)
% EGRSS_TRTRI   Computes an implicit inverse of a lower triangular matrix
% of the form L = tril(Ut'*Wt,-1) + diag(c).
%
% [Yt,Zt] = EGRSS_TRTRI(Ut,Wt,c) computes Yt and Zt such that
% inv(L) = tril(Yt'*Zt,-1) + diag(1./c).
%
% See also: EGRSS_POTRF

assert(all(size(Ut) == size(Wt)),'Dimension mismatch: Ut and Wt must be of the same size.')
assert(length(c) == size(Ut,2) && isvector(c),'Dimension mismatch: c must be a vector of length size(Ut,2)')

[p,n] = size(Ut);

% Solve L*Yt' = Ut'
Yt = zeros(size(Ut));
for k = 1:p
    Yt(k,:) = egrss_trsv(Ut,Wt,c,Ut(k,:)')';
end

% Solve L'*Zt = Wt and scale Zt := (Ut*Zt' - I)'\Zt
Zt = Wt;
Zt(:,end) = 0;
for k = 1:p
    Zt(k,:) = egrss_trsv(Ut,Wt,c,Zt(k,:)','T')';
end
Zt = (Zt(:,1:end-1)*Ut(:,1:end-1)' - eye(p))\Zt;

end
