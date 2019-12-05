function b = egrss_trmv(Ut,Wt,varargin)
% EGRSS_TRMV   Computes matrix-vector product L*x or L'*x where L is a lower
% triangular extended generator representable semiseparable or quasi-separable
% matrix defined in terms of matrices Ut and Wt, which are of size p-by-n
% (with p > 0), and in the quasi-separable case, a vector c of length n.
%
% b = EGRSS_TRMV(Ut,Wt,x) computes b = L*x with L = tril(Ut'*Wt).
%
% b = EGRSS_TRMV(Ut,Wt,x,'N') is the same as b = EGRSS_TRMV(Ut,Wt,x).
%
% b = EGRSS_TRMV(Ut,Wt,x,'T') computes b = L'*x with L = tril(Ut'*Wt).
%
% b = EGRSS_TRMV(Ut,Wt,c,x) computes b = L*x with L = tril(Ut'*Wt,-1) + diag(c).
%
% b = EGRSS_TRMV(Ut,Wt,c,x,'N') is the same as b = EGRSS_TRMV(Ut,Wt,c,x).
%
% b = EGRSS_TRMV(Ut,Wt,c,x,'T') computes b = L'*x with L = tril(Ut'*Wt,-1) + diag(c).
%
% See also: EGRSS_POTRF, EGRSS_TRSV


assert(all(size(Ut) == size(Wt)),'Dimension mismatch: Ut and Wt must be of the same size.')
if nargin == 3
    x = varargin{1};
    trans = 'N';
elseif nargin == 4
    if ischar(varargin{2})
        x = varargin{1};
        trans = varargin{2};
    elseif isvector(varargin{2})
        c = varargin{1};
        x = varargin{2};
        trans = 'N';
    else
        error('Invalid input.');
    end
elseif nargin == 5
    c = varargin{1};
    x = varargin{2};
    trans = varargin{3};
else
    error('Invalid number of inputs.');
end

assert(ischar(trans) && length(trans) == 1,'Expected character ''N'' or ''T''.')
assert(length(x) == size(Ut,2) && isvector(x),'Dimension mismatch: x must be a vector of length size(Ut,2)')

[p,n] = size(Ut);
b = x;
z = zeros(p,1);

if exist('c')
    assert(length(c) == size(Ut,2) && isvector(c),'Dimension mismatch: c must be a vector of length size(Ut,2)')
    switch lower(trans)
        case 'n'
            for k = 1:n
                b(k) = c(k)*b(k) + Ut(:,k)'*z;
                z = z + Wt(:,k)*x(k);
            end
        case 't'
            for k = n:-1:1
                b(k) = c(k)*b(k) + Wt(:,k)'*z;
                z = z + Ut(:,k)*x(k);
            end
        otherwise
            error('Expected character ''N'' or ''T''.')
    end
else
    switch lower(trans)
        case 'n'
            for k = 1:n
                z = z + Wt(:,k)*b(k);
                b(k) = Ut(:,k)'*z;
            end
        case 't'
            for k = n:-1:1
                z = z + Ut(:,k)*b(k);
                b(k) = Wt(:,k)'*z;
            end
        otherwise
            error('Expected character ''N'' or ''T''.')
    end
end
end
