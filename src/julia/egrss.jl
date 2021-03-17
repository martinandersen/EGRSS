module egrss
using LinearAlgebra

export generators, symv, gemv, potrf, ldl, trmv, trsv, trtri, trtri2, trnrms, full, tril

"""
    generators(t, p)

Computes generator representation of p'th order spline kernel matrix generated
by a strictly monotonic vector t of length n.

Ut,Vt = generators(t,p) returns two matrices Ut and Vt of size p-by-n
(with p > 0) such that K = tril(Ut'*Vt) + triu(Vt'*Ut,1) is the kernel
matrix with elements

    K[i,j] = sum_{k=0}^{p-1} (-1)^k/(factorial(p-1-k)*factorial(p+k)
                *(t[i]*t[j])^(p-1-k)*min(t[i],t[j])^(2*k+1)

where t is a nonnegative vector.

# Example

```
t = range(0,1,101)
Ut, Vt = generators(t, 2)
```
"""
function generators(t, p)
    Ut = repeat(t',p,1).^collect(p-1:-1:0)./map(factorial,p-1:-1:0)
    Vt = ((-1).^(0:p-1)).*(repeat(t',p,1).^collect(p:2*p-1)./map(factorial,p:2*p-1))
    return Ut,Vt
end

"""
    full(Ut, Vt[, d])

Forms dense symmetric matrix from generator representation.

K = full(Ut,Vt) forms the symmetric matrix

    K = tril(Ut'*Vt) + triu(Vt'*Ut,1).

K = full(Ut,Vt,d) forms the symmetric matrix

    K = tril(Ut'*Vt) + triu(Vt'*Ut,1) + diag(d).
"""
function full(Ut, Vt, d=Nothing)
    K = tril(Ut'*Vt,-1)
    K = K + K' + Diagonal(sum(Ut.*Vt,dims=1)[:])
    if d !== Nothing
        if isa(d,Number)
            K = K + Diagonal(d*ones(eltype(d),size(Ut,2)))
        else
            K = K + Diagonal(d[:])
        end
    end
    return Symmetric(K)
end

"""
    full_tril(Ut, Wt[, d])

Forms dense lower triangular matrix from generator representation.

L = full_tril(Ut,Wt) forms the lower triangular matrix

    L = tril(Ut'*Wt)

L = full_tril(Ut,Wt,d) forms the lower triangular matrix

    L = tril(Ut'*Wt,-1) + diag(d).
"""
function full_tril(Ut, Wt, c=Nothing)
    if c === Nothing
        return LowerTriangular(tril(Ut'*Wt))
    else
        if isa(c,Number)
            c = c*ones(eltype(c),size(Ut,2))
        end
        return LowerTriangular(tril(Ut'*Wt,-1) + Diagonal(c[:]))
    end
end

"""
    gemv(Ut, Vt, Pt, Qt, x)

Computes matrix-vector product A*x where A is an extended generator
representable semiseparable matrix given by

   A = tril(Ut'*Vt) + triu(Pt'*Qt,1).

# Example

```
b = gemv(Ut,Vt,Pt,Qt,x)
```
"""
function gemv(Ut,Vt,Pt,Qt,x)
    p, m = size(Ut)
    q, n = size(Qt)
    min_mn = min(m,n)

    b = [x[1:min_mn]; zeros(eltype(Ut),max(0,m-n))]
    v = zeros(eltype(Ut),size(Ut,1))
    q = Qt[:,1:min_mn]*x[1:min_mn]

    for k = 1:min_mn
        v = v + Vt[:,k]*b[k]
        q = q - Qt[:,k]*b[k]
        b[k] = Ut[:,k]'*v + Pt[:,k]'*q
    end

    if n > m
        b = b + Pt'*(Qt[:,min_mn+1:end]*x[min_mn+1:end])
    elseif m > n
        b[min_mn+1:end] = Ut[:,min_mn+1:end]'*(Vt*x)
    end

    return b
end

"""

Computes matrix-vector product A*x where A is a symmetric and extended generator
representable semiseparable matrix given by

    A = tril(Ut'*Vt) + triu(Vt'*Ut,1)

# Example

```
b = symv(Ut,Vt,x)
```
"""
function symv(Ut,Vt,x)
    p, n = size(Ut)
    b = copy(x)
    z = zeros(eltype(Ut),p)
    y = Ut*x
    for k = 1:n
        z = z + Vt[:,k]*b[k]
        y = y - Ut[:,k]*b[k]
        b[k] = Ut[:,k]'*z + Vt[:,k]'*y
    end
    return b
end


"""
    potrf(Ut, Vt[, d])

Computes implicit Cholesky factorization of the sum of a symmetric extended
generator representable semiseparable matrix and a diagonal matrix.

Wt = potrf(Ut,Vt) computes a matrix Wt such that L = tril(Ut'*Wt) is
the Cholesky factor of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1),
i.e., A = L*L'.

Wt,c = potrf(Ut,Vt,d) computes a matrix Wt and a vector c such that
L = tril(Ut'*Wt,-1) + diag(c) is the Cholesky factor of the symmetric matrix
A = tril(Ut'*Vt) + triu(Vt'*Ut,1) + diag(d), i.e., A = L*L'. The input d must
either be a vector of length size(Ut,2) or a scalar; in the latter case diag(d)
is interpreted as the identity matrix scaled by d.
"""
function potrf(Ut,Vt,d=Nothing)
    p, n = size(Ut)
    Wt = copy(Vt)
    P = zeros(eltype(Ut),p,p)
    if d === Nothing
        for k = 1:n
            Wt[:,k] = Wt[:,k] - P*Ut[:,k]
            Wt[:,k] = Wt[:,k]/sqrt(Ut[:,k]'*Wt[:,k])
            P = P + Wt[:,k]*Wt[:,k]'
        end
        return Wt
    else
        if isa(d,Number)
            c = d*ones(eltype(d),n)
        else
            c = copy(d)
        end
        for k = 1:n
            Wt[:,k] = Wt[:,k] - P*Ut[:,k]
            c[k] = sqrt(Ut[:,k]'*Wt[:,k] + c[k])
            Wt[:,k] = Wt[:,k]/c[k]
            P = P + Wt[:,k]*Wt[:,k]'
        end
        return Wt, c
    end
end

"""
    ldl(Ut, Vt[, d])

Computes implicit LDL factorization of the sum of a symmetric extended
generator representable semiseparable matrix and a diagonal matrix.

Wt,c = ldl(Ut,Vt) computes a matrix Wt and a vector c such that the LDL
factorization of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1) + diag(d)
is given by L = tril(Ut'*Wt,-1)+I and D = diag(c), i.e., A = L*D*L'.

Wt,c = ldl(Ut,Vt,d) computes a matrix Wt and a vector c such that the LDL
factorization of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1) is given
by L = tril(Ut'*Wt,-1)+I and D = diag(c), i.e., A = L*D*L'.
"""
function ldl(Ut,Vt,d=Nothing)
    p, n = size(Ut)
    Wt = copy(Vt)
    P = zeros(eltype(Ut),p,p)
    if d === Nothing
        c = zeros(eltype(Ut),n);
        for k = 1:n
            Wt[:,k] = Wt[:,k] - P*Ut[:,k]
            c[k] = Ut[:,k]'*Wt[:,k]
            P = P + (Wt[:,k]*Wt[:,k]')/c[k]
            Wt[:,k] = Wt[:,k]/c[k]
        end
    else
        if isa(d,Number)
            c = d*ones(eltype(d),n)
        else
            c = copy(d)
        end
        for k = 1:n
            Wt[:,k] = Wt[:,k] - P*Ut[:,k]
            c[k] = Ut[:,k]'*Wt[:,k] + c[k]
            P = P + (Wt[:,k]*Wt[:,k]')/c[k]
            Wt[:,k] = Wt[:,k]/c[k]
        end
    end
    return Wt, c
end

"""
Computes matrix-vector product L*x or L'*x where L is a lower triangular
extended generator representable semiseparable or quasi-separable matrix
defined in terms of matrices Ut and Wt, which are of size p-by-n (with p > 0),
and in the quasi-separable case, a vector c of length n.

b = trmv(Ut,Wt,x) computes b = L*x with L = tril(Ut'*Wt).

b = trmv(Ut,Wt,x,'N') is the same as b = trmv(Ut,Wt,x).

b = trmv(Ut,Wt,x,'T') computes b = L'*x with L = tril(Ut'*Wt).

b = trmv(Ut,Wt,c,x) computes b = L*x with L = tril(Ut'*Wt,-1) + diag(c).

b = trmv(Ut,Wt,c,x,'N') is the same as b = trmv(Ut,Wt,c,x).

b = trmv(Ut,Wt,c,x,'T') computes b = L'*x with L = tril(Ut'*Wt,-1) + diag(c).
"""
function trmv(Ut,Wt,args...)
    p, n = size(Ut)

    if length(args) == 3
        c = args[1]
        x = args[2]
        trans = args[3]
    elseif length(args) == 2
        if args[2] isa Char
            c = Nothing
            x = args[1]
            trans = args[2]
        else
            c = args[1]
            x = args[2]
            trans = 'N'
        end
    elseif length(args) == 1
        c = Nothing
        x = args[1]
        trans = 'N'
    else
        error("Invalid number of input arguments.")
    end

    b = copy(x)
    z = zeros(eltype(Ut),p)
    if c === Nothing
        if trans == 'N'
            for k = 1:n
                z = z + Wt[:,k]*x[k]
                b[k] = Ut[:,k]'*z
            end
        elseif trans == 'T'
            for k = n:-1:1
                z = z + Ut[:,k]*x[k]
                b[k] = Wt[:,k]'*z
            end
        else
            error("Expected character 'N or 'T'.")
        end
    else
        if trans == 'N'
            for k = 1:n
                b[k] = c[k]*b[k] + Ut[:,k]'*z
                z = z + Wt[:,k]*x[k]
            end
        elseif trans == 'T'
            for k = n:-1:1
                b[k] = c[k]*b[k] + Wt[:,k]'*z
                z = z + Ut[:,k]*x[k]
            end
        else
            error("Expected character 'N or 'T'.")
        end
    end

    return b
end

"""
Solves the equation L*x = b or L'*x = b where L is a lower triangular extended
generator representable semiseparable or quasi-separable matrix defined in terms
of matrices Ut and Wt, which are of size p-by-n (with p > 0), and in the
quasi-separable case, a vector c of length n.

x = trsv(Ut,Wt,b) solves L*x = b with L = tril(Ut'*Wt).

x = trsv(Ut,Wt,b,'N') is the same as x = trsv(Ut,Wt,b).

x = trsv(Ut,Wt,b,'T') solves L'*x = b with L = tril(Ut'*Wt).

x = trsv(Ut,Wt,c,b) solves L*x = b with L = tril(Ut'*Wt,-1) + diag(c).

x = trsv(Ut,Wt,c,b,'N') is the same as x = trsv(Ut,Wt,c,b).

x = trsv(Ut,Wt,c,b,'T') solves L'*x = b with L = tril(Ut'*Wt,-1) + diag(c).
"""
function trsv(Ut,Wt,args...)
    p, n = size(Ut)

    if length(args) == 3
        c = args[1]
        b = args[2]
        trans = args[3]
    elseif length(args) == 2
        if args[2] isa Char
            c = Nothing
            b = args[1]
            trans = args[2]
        else
            c = args[1]
            b = args[2]
            trans = 'N'
        end
    elseif length(args) == 1
        c = Nothing
        b = args[1]
        trans = 'N'
    else
        error("Invalid number of input arguments.")
    end

    x = copy(b)
    z = zeros(eltype(Ut),p);
    if c === Nothing
        if trans == 'N'
            for k = 1:n
                x[k] = (x[k]-Ut[:,k]'*z)/(Wt[:,k]'*Ut[:,k])
                z = z + Wt[:,k]*x[k]
            end
        elseif trans == 'T'
            for k = n:-1:1
                x[k] = (x[k]-Wt[:,k]'*z)/(Wt[:,k]'*Ut[:,k])
                z = z + Ut[:,k]*x[k]
            end
        else
            error("Expected character 'N or 'T'.")
        end
    else
        if trans == 'N'
            for k = 1:n
                x[k] = (x[k]-Ut[:,k]'*z)/c[k]
                z = z + Wt[:,k]*x[k]
            end
        elseif trans == 'T'
            for k = n:-1:1
                x[k] = (x[k]-Wt[:,k]'*z)/c[k]
                z = z + Ut[:,k]*x[k]
            end
        else
            error("Expected character 'N or 'T'.")
        end
    end

    return x
end

"""
    trtri(Ut, Wt, c)

Computes an implicit inverse of a lower triangular matrix of the form

    L = tril(Ut'*Wt,-1) + diag(c).

Yt,Zt = trtri(Ut,Wt,c) computes Yt and Zt such that

    L^{-1} = tril(Yt'*Zt,-1) + diag(1./c).
"""
function trtri(Ut,Wt,c)

    p, n = size(Ut)
    Yt = copy(Ut)
    Zt = copy(Wt)

    for k = 1:p
        Yt[k,:] = trsv(Ut,Wt,c,Yt[k,:]')';
    end

    Zt[:,end] .*= 0.0;
    for k = 1:p
        Zt[k,:] = trsv(Ut,Wt,c,Zt[k,:]','T')';
    end
    Zt = (Zt[:,1:end-1]*Ut[:,1:end-1]' - I)\Zt;

    return Yt,Zt
end

"""
    trtri2(Ut, Wt[, c])

Computes the explicit inverse of a lower triangular matrix of either the form
L = tril(Ut'*Wt) or L = tril(Ut'*Wt,-1) + diag(c).

"""
function trtri2(Ut,Wt,c=Nothing)
    p, n = size(Ut)
    Linv = zeros(eltype(Ut),n,n)
    e = zeros(eltype(Ut),n)
    e[1] = 1
    if c === Nothing
        for k = 1:n
            Linv[k:end,:] = trsv(Ut,Wt,e[1:end+1-k])
        end
    else
        for k = 1:n
            Linv[k:end,k] = trsv(Ut[:,k:end],Wt[:,k:end],c[k:end],e[1:end+1-k])
        end
    end
    return LowerTriangular(Linv)
end

"""
    trnrms(Ut, Wt, c)

Computes the squared column norms of a lower triangular matrix of the form
L = tril(Ut'*Wt,-1) + diag(c) where Ut and Wt are of size p-by-n (with p > 0),
and c is a positive vector of length n.

b = trnrms(Ut,Wt,c) computes a vector b and with entries
b(i) = norm(L(:,i))^2 where L = tril(Ut'*Wt,-1) + diag(c).
"""
function trnrms(Ut,Wt,c)
    p, n = size(Ut)
    colnrms = zeros(eltype(Ut),n)
    P = zeros(eltype(Ut),p,p);
    for k = n:-1:1
        colnrms[k] = c[k]^2 + Wt[:,k]'*P*Wt[:,k];
        P = P + Ut[:,k]*Ut[:,k]';
    end
    return colnrms
end

end
