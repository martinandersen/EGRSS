let
    n = 50
    p = 3
    t = (1:n)/n
    Ut,Vt = egrss.generators(t,p);
    Wt,c = egrss.potrf(Ut,Vt,1e-2*ones(n))
    Yt,Zt = egrss.trtri(Ut,Wt,c)

    Lref = cholesky(egrss.full(Ut,Vt) + Diagonal(1e-2*ones(n))).L
    Lref_inv = LowerTriangular(tril(inv(Lref)));

    L_inv = egrss.full_tril(Yt,Zt,1.0./c)
    @test ( isapprox(L_inv,Lref_inv) )

    L_inv = egrss.trtri2(Ut,Wt,c)
    @test ( isapprox(L_inv,Lref_inv) )
end
