let
    n = 50
    p = 3
    t = (1:n)/n
    Ut,Vt = egrss.generators(t,p);
    K = egrss.full(Ut,Vt)

    Wt = egrss.potrf(Ut,Vt)
    Lref = cholesky(K).L
    L = egrss.full_tril(Ut,Wt)
    @test( isapprox(L,Lref) )

    alpha = 1e-6
    Wt,c = egrss.potrf(Ut,Vt,alpha)
    Lref = cholesky(K + Diagonal(alpha*ones(n))).L
    L = egrss.full_tril(Ut,Wt,c)
    @test( isapprox(L,Lref) )

    d = collect((1:n)/n) .+ 1e-3
    Wt,c = egrss.potrf(Ut,Vt,d)
    Lref = cholesky(K + Diagonal(d)).L
    L = egrss.full_tril(Ut,Wt,c)
    @test( isapprox(L,Lref) )

end
