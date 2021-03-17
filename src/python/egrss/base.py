# -*- coding: utf-8 -*-
import numpy as np

def generators(t,p):
    """
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
    p      = 4
    t      = np.linspace(1e-2,1,10)
    Ut, Vt = generators(t,p)
    ```
    """
    if t.ndim != 1: raise ValueError('t must be a one-dimensional array')

    if np.all(np.diff(t) > 0):
        monotonic = 1
    elif np.all(np.diff(t) < 0):
        monotonic = -1
    else:
        raise ValueError('t must be monotonic')

    T = np.vander(t, 2*p)/np.flip(np.cumprod(np.hstack([1,np.arange(1,2*p)])))
    Ut = T[:,p:].T
    Vt = (np.fliplr(T[:,:p])*((-1)**np.arange(0,p))).T
    if monotonic == 1:
        return Ut,Vt
    else:
        return Vt,Ut


def full(Ut, Vt, d = None):
    """
    Forms dense symmetric matrix from generator representation.

    K = full(Ut,Vt) forms the symmetric matrix
        K = tril(Ut'*Vt) + triu(Vt'*Ut,1).

    K = full(Ut,Vt,d) forms the symmetric matrix
        K = tril(Ut'*Vt) + triu(Vt'*Ut,1) + diag(d).
    """
    K = np.tril(Ut.T@Vt,-1)
    K = K + K.T + np.diag(np.sum(Ut*Vt,axis=0).flatten())
    if d is not None:
        if np.isscalar(d):
            K = K + np.diag(d*np.ones(Ut.shape[1]))
        else:
            K = K + np.diag(d)

    return K

def potrf(Ut,Vt, d = None):
    """
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
    p,n = Ut.shape
    Wt  = Vt.copy()
    P   = np.zeros((p,p))
    if d is None:
        for k in range(n):
            Wt[:,k] = Wt[:,k] - P@Ut[:,k]
            Wt[:,k] = Wt[:,k] / np.sqrt(np.dot(Ut[:,k],Wt[:,k]))
            P       = P + np.outer(Wt[:,k],Wt[:,k])

        return Wt
    else:
        if np.isscalar(d):
            c = d*np.ones(n)
        else:
            c = d.copy()

        for k in range(n):
            Wt[:,k] = Wt[:,k] - P@Ut[:,k]
            c[k]    = np.sqrt(np.dot(Ut[:,k],Wt[:,k])+c[k])
            Wt[:,k] = Wt[:,k] / c[k]
            P       = P + np.outer(Wt[:,k],Wt[:,k])

        return Wt, c


def ldl(Ut,Vt, d = None):
    """
    Computes implicit LDL factorization of the sum of a symmetric extended
    generator representable semiseparable matrix and a diagonal matrix.

    Wt,c = ldl(Ut,Vt) computes a matrix Wt and a vector c such that the LDL
    factorization of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1) + diag(d)
    is given by L = tril(Ut'*Wt,-1)+I and D = diag(c), i.e., A = L*D*L'.

    Wt,c = ldl(Ut,Vt,d) computes a matrix Wt and a vector c such that the LDL
    factorization of the symmetric matrix A = tril(Ut'*Vt)+triu(Vt'*Ut,1) is given
    by L = tril(Ut'*Wt,-1)+I and D = diag(c), i.e., A = L*D*L'.
    """
    p,n = Ut.shape
    Wt  = Vt.copy()
    P   = np.zeros((p,p))
    if d is None:
        c = np.zeros(n)
        for k in range(n):
            Wt[:,k] = Wt[:,k] - P@Ut[:,k]
            c[k]    = np.dot(Ut[:,k],Wt[:,k])
            P       = P + np.outer(Wt[:,k],Wt[:,k]) / c[k]
            Wt[:,k] = Wt[:,k] / c[k]
    else:
        if np.isscalar(d):
            c = d*np.ones(n)
        else:
            c = d.copy()

        for k in range(n):
            Wt[:,k] = Wt[:,k] - P@Ut[:,k]
            c[k]    = np.dot(Ut[:,k],Wt[:,k]) + c[k]
            P       = P + np.outer(Wt[:,k],Wt[:,k]) / c[k]
            Wt[:,k] = Wt[:,k] / c[k]
    return Wt, c


def full_tril(Ut, Wt, d = None):
    """
    Forms dense lower triangular matrix from generator representation.

    L = full_tril(Ut,Wt) forms the lower triangular matrix L = tril(Ut'*Wt).

    L = full_tril(Ut,Wt,d) forms the lower triangular matrix L = tril(Ut'*Wt,-1) + diag(d).
    """
    if d is None:
        return np.tril(Ut.T@Wt)
    else:
        if np.isscalar(d):
            return np.tril(Ut.T@Wt,-1) + np.diag(d*np.ones(Ut.shape[1]))
        else:
            return np.tril(Ut.T@Wt,-1) + np.diag(d)


def trmv(Ut, Wt, x, c = None, trans = 'N'):
    """
    Computes matrix-vector product L*x or L'*x where L is a lower
    triangular extended generator representable semiseparable or quasi-separable
    matrix defined in terms of matrices Ut and Wt, which are of size p-by-n
    (with p > 0), and in the quasi-separable case, a vector c of length n.

    b = trmv(Ut,Wt,x) computes b = L*x with L = tril(Ut'*Wt).

    b = trmv(Ut,Wt,x,trans='N') is the same as b = trmv(Ut,Wt,x).

    b = trmv(Ut,Wt,x,trans='T') computes b = L'*x with L = tril(Ut'*Wt).

    b = trmv(Ut,Wt,x,c) computes b = L*x with L = tril(Ut'*Wt,-1) + diag(c).

    b = trmv(Ut,Wt,x,c,trans='N') is the same as b = trmv(Ut,Wt,c,x).

    b = trmv(Ut,Wt,x,c,trans='T') computes b = L'*x with L = tril(Ut'*Wt,-1) + diag(c).
    """
    p,n = Ut.shape
    if trans not in ['N','T']:
        raise ValueError("trans must be 'N' or 'T'")

    b = x.copy()
    z = np.zeros(p)

    if c is None:
        if trans == 'N':
            for k in range(n):
                z    = z + Wt[:,k]*b[k]
                b[k] = np.dot(Ut[:,k],z)
        elif trans == 'T':
            for k in range(n-1,-1,-1):
                z    = z + Ut[:,k]*b[k]
                b[k] = np.dot(Wt[:,k],z)
    else:
        if trans == 'N':
            for k in range(n):
                b[k] = c[k]*b[k]+np.dot(Ut[:,k],z)
                z    = z + Wt[:,k]*x[k]
        elif trans == 'T':
            for k in range(n-1,-1,-1):
                b[k] = c[k]*b[k] + np.dot(Wt[:,k],z)
                z    = z + Ut[:,k]*x[k]

    return b



def trsv(Ut, Wt, b, c = None, trans = 'N'):
    """
    Solves the equation L*x = b or L'*x = b where L is a lower triangular extended
    generator representable semiseparable or quasi-separable matrix defined in terms
    of matrices Ut and Wt, which are of size p-by-n (with p > 0), and in the
    quasi-separable case, a vector c of length n.

    x = trsv(Ut,Wt,b) solves L*x = b with L = tril(Ut'*Wt).

    x = trsv(Ut,Wt,b,trans='N') is the same as x = trsv(Ut,Wt,b).

    x = trsv(Ut,Wt,b,trans='T') solves L'*x = b with L = tril(Ut'*Wt).

    x = trsv(Ut,Wt,b,c) solves L*x = b with L = tril(Ut'*Wt,-1) + diag(c).

    x = trsv(Ut,Wt,b,c,trans='N') is the same as x = trsv(Ut,Wt,c,b).

    x = trsv(Ut,Wt,b,c,trans='T') solves L'*x = b with L = tril(Ut'*Wt,-1) + diag(c).
    """
    p,n = Ut.shape
    if trans not in ['N','T']:
        raise ValueError("trans must be 'N' or 'T'")

    x = b.copy()
    z = np.zeros(p)

    if c is None:
        if trans == 'N':
            for k in range(n):
                x[k] = (x[k]-np.dot(Ut[:,k],z)) / np.dot(Wt[:,k],Ut[:,k])
                z    = z + Wt[:,k] * x[k]
        elif trans == 'T':
            for k in range(n-1,-1,-1):
                x[k] = (x[k]-np.dot(Wt[:,k],z)) / np.dot(Wt[:,k],Ut[:,k])
                z    = z + Ut[:,k] * x[k]
    else:
        if trans == 'N':
            for k in range(n):
                x[k] = (x[k]-np.dot(Ut[:,k],z)) / c[k]
                z    = z + Wt[:,k] * x[k]
        elif trans == 'T':
            for k in range(n-1,-1,-1):
                x[k] = (x[k]-np.dot(Wt[:,k],z)) / c[k]
                z    = z + Ut[:,k] * x[k]

    return x


def symv(Ut,Vt,x):
    """
    Computes matrix-vector product A*x where A is a symmetric and extended generator
    representable semiseparable matrix given by

        A = tril(Ut'*Vt) + triu(Vt'*Ut,1)

    # Example

    ```
    b = symv(Ut,Vt,x)
    ```
    """
    p,n = Ut.shape
    b   = x.copy()
    z   = np.zeros(p)
    y   = np.reshape(np.dot(Ut,x),(p,))

    for k in range(n):
        z = z + Vt[:,k]*b[k]
        y = y - Ut[:,k]*b[k]
        b[k] = np.dot(Ut[:,k],z) + np.dot(Vt[:,k],y)
    return b


def gemv(Ut, Vt, Pt, Qt, x):
    """
    Computes matrix-vector product A*x where A is an extended generator
    representable semiseparable matrix given by

       A = tril(Ut'*Vt) + triu(Pt'*Qt,1).

    # Example

    ```
    b = gemv(Ut,Vt,Pt,Qt,x)
    ```
    """
    p, m = Ut.shape
    q, n = Qt.shape
    min_mn = min(m,n)

    b = np.concatenate([x[:min_mn], np.zeros(max(0,m-n))])
    v = np.zeros(p)
    q = Qt[:,:min_mn]@x[:min_mn]

    for k in range(min_mn):
        v = v + Vt[:,k]*b[k]
        q = q - Qt[:,k]*b[k]
        b[k] = np.dot(Ut[:,k],v) + np.dot(Pt[:,k],q)

    if n > m:
        b = b + Pt.T@(Qt[:,min_mn:]@x[min_mn:])
    elif m > n:
        b[min_mn:] = Ut[:,min_mn:].T@(Vt@x)

    return b

def trtri(Ut,Wt,c):
    """
    Computes an implicit inverse of a lower triangular matrix of the form

        L = tril(Ut'*Wt,-1) + diag(c).

    Yt,Zt = trtri(Ut,Wt,c) computes Yt and Zt such that

        L^{-1} = tril(Yt'*Zt,-1) + diag(1./c).
    """
    p, n = Ut.shape
    Yt = Ut.copy()
    Zt = Wt.copy()

    for k in range(p):
        Yt[k,:] = trsv(Ut,Wt,Yt[k,:],c,trans='N')

    Zt[:,-1] = 0.0
    for k in range(p):
        Zt[k,:] = trsv(Ut,Wt,Zt[k,:],c,trans='T')
    Zt = np.linalg.solve(Zt[:,:-1]@Ut[:,:-1].T - np.eye(p),Zt)

    return Yt,Zt

def trtri2(Ut,Wt,c=None):
    """
    Computes the explicit inverse of a lower triangular matrix of either the form
    L = tril(Ut'*Wt) or L = tril(Ut'*Wt,-1) + diag(c).
    """
    p,n = Ut.shape
    Linv = np.zeros((n,n))
    e = np.zeros(n)
    e[0] = 1
    if c is None:
        for k in range(n):
            Linv[k:,:] = trsv(Ut,Wt,e[:n-k])
    else:
        for k in range(n):
            Linv[k:,k] = trsv(Ut[:,k:],Wt[:,k:],e[:n-k],c[k:])
    return Linv


def trnrms(Ut,Wt,c):
    """
    Computes the squared column norms of a lower triangular matrix of the form
    L = tril(Ut'*Wt,-1) + diag(c) where Ut and Wt are of size p-by-n (with p > 0),
    and c is a positive vector of length n.

    b = trnrms(Ut,Wt,c) computes a vector b and with entries
    b(i) = norm(L(:,i))^2 where L = tril(Ut'*Wt,-1) + diag(c).
    """
    p, n = Ut.shape
    colnrms = np.zeros(n)
    P = np.zeros((p,p))

    for k in range(n-1,-1,-1):
        colnrms[k] = c[k]**2 + np.dot(Wt[:,k],P@Wt[:,k])
        P = P + np.outer(Ut[:,k],Ut[:,k])

    return colnrms
