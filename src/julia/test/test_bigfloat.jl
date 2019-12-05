let
    setprecision(BigFloat, 256)

    tol = 1e-20
    p = 3
    n = 20
    t = map(BigFloat, (1:n)//n)
    Ut,Vt = egrss.generators(t,p);
    d = BigFloat("1e-8")*ones(BigFloat,n)
    Wt,c = egrss.potrf(Ut,Vt,d)
    Yt,Zt = egrss.trtri(Ut,Wt,c)

    K = egrss.full(Ut,Vt,d)
    Lref = cholesky(K).L
    L = egrss.full_tril(Ut,Wt,c)
    L_inv = egrss.full_tril(Yt,Zt,1.0./c)

    @test ( norm(I-L*L_inv) < n*tol )
    @test ( norm(L-Lref) < norm(Lref)*tol )
end
