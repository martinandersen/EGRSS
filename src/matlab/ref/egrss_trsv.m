function x = egrss_trsv(Ut,Wt,varargin)
% EGRSS_TRSV   Solves the equation L*x = b or L'*x = b where L is a lower
% triangular extended generator representable semiseparable or quasi-separable
% matrix defined in terms of matrices Ut and Wt, which are of size p-by-n
% (with p > 0), and in the quasi-separable case, a vector c of length n.
%
% x = EGRSS_TRSV(Ut,Wt,b) solves L*x = b with L = tril(Ut'*Wt).
%
% x = EGRSS_TRSV(Ut,Wt,b,'N') is the same as x = EGRSS_TRSV(Ut,Wt,b).
%
% x = EGRSS_TRSV(Ut,Wt,b,'T') solves L'*x = b with L = tril(Ut'*Wt).
%
% x = EGRSS_TRSV(Ut,Wt,c,b) solves L*x = b with L = tril(Ut'*Wt,-1) + diag(c).
%
% x = EGRSS_TRSV(Ut,Wt,c,b,'N') is the same as x = EGRSS_TRSV(Ut,Wt,c,b).
%
% x = EGRSS_TRSV(Ut,Wt,c,b,'T') solves L'*x = b with L = tril(Ut'*Wt,-1) + diag(c).
%
% See also: EGRSS_POTRF, EGRSS_TRMV


assert(all(size(Ut) == size(Wt)),'Dimension mismatch: Ut and Wt must be of the same size.')
if nargin == 3
    b = varargin{1};
    trans = 'N';
elseif nargin == 4
    if ischar(varargin{2})
        b = varargin{1};
        trans = varargin{2};
    elseif isvector(varargin{2})
        c = varargin{1};
        b = varargin{2};
        trans = 'N';
    else
        error('Invalid input.');
    end
elseif nargin == 5
    c = varargin{1};
    b = varargin{2};
    trans = varargin{3};
else
    error('Invalid number of inputs.');
end

assert(ischar(trans) && length(trans) == 1,'Expected character ''N'' or ''T''.')
assert(length(b) == size(Ut,2) && isvector(b),'Dimension mismatch: b must be a vector of length size(Ut,2)')

[p,n] = size(Ut);
x = b;
z = zeros(p,1);

if exist('c')
    assert(length(c) == size(Ut,2) && isvector(c),'Dimension mismatch: c must be a vector of length size(Ut,2)')
    switch lower(trans)
        case 'n'
            for k = 1:n
                x(k) = (x(k)-Ut(:,k)'*z)/c(k);
                z = z + Wt(:,k)*x(k);
            end
        case 't'
            for k = n:-1:1
                x(k) = (x(k)-Wt(:,k)'*z)/c(k);
                z = z + Ut(:,k)*x(k);
            end
        otherwise
            error('Expected character ''N'' or ''T''.')
    end
else
    switch lower(trans)
        case 'n'
            for k = 1:n
                x(k) = (x(k)-Ut(:,k)'*z)/(Wt(:,k)'*Ut(:,k));
                z = z + Wt(:,k)*x(k);
            end
        case 't'
            for k = n:-1:1
                x(k) = (x(k)-Wt(:,k)'*z)/(Wt(:,k)'*Ut(:,k));
                z = z + Ut(:,k)*x(k);
            end
        otherwise
            error('Expected character ''N'' or ''T''.')
    end
end
end
