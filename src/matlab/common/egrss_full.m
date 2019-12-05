function K = egrss_full(Ut,Vt,varargin)
% EGRSS_FULL   Forms dense symmetric matrix from generator representation.
%
% K = EGRSS_FULL(Ut,Vt) forms the symmetric matrix K = tril(Ut'*Vt) +
% triu(Vt'*Ut,1).
%
% K = EGRSS_FULL(Ut,Vt,d) forms the symmetric matrix K = tril(Ut'*Vt) +
% triu(Vt'*Ut,1) + diag(d).

if nargin > 3
  error('Too many input arguments.');
elseif nargin < 2
  error('Not enough input arguments.');
end
K = tril(Ut'*Vt,-1);
K = K + K' + diag(sum(Ut.*Vt,1));
if nargin == 3
    d = varargin{1};
    if isscalar(d)
      d = d*ones(size(K,1),1);
    elseif ~isvector(d) || length(d) ~= size(K,1)
      error('Dimension of d does not match K.')
    end
    K = K + diag(d);
end

end
