function L = egrss_full_tril(Ut,Wt,varargin)
% EGRSS_FULL_TRIL   Forms dense lower triangular matrix from generator
% representation.
%
% L = EGRSS_FULL_TRIL(Ut,Wt) forms the lower triangular matrix L = tril(Ut'*Wt).
%
% L = EGRSS_FULL_TRIL(Ut,Wt,c) forms the lower triangular matrix L =
%     tril(Ut'*Wt,-1) + diag(c).

if nargin > 3
  error('Too many input arguments.');
elseif nargin < 2
  error('Not enough input arguments.');
end
if nargin == 2
  L = tril(Ut'*Wt);
else
  c = varargin{1};
  if isscalar(c)
    c = c*ones(size(Ut,2),1);
  elseif ~isvector(c) || length(c) ~= size(Ut,2)
    error('Dimension of c does not match Ut.')
  end
  L = tril(Ut'*Wt,-1) + diag(c);
end
end
