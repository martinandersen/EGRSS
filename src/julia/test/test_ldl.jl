let
    n = 50
    p = 2
    t = (1:n)/n
    Ut,Vt = egrss.generators(t,p);
    K = egrss.full(Ut,Vt)

    Wt,c = egrss.ldl(Ut,Vt)
    Lref = cholesky(K).L
    L = egrss.full_tril(Ut,Wt,1.0)
    @test( isapprox(L,Lref./diag(Lref)') )
    @test( isapprox(map(sqrt,c),diag(Lref)) )

    alpha = 1e-6
    Wt,c = egrss.ldl(Ut,Vt,alpha)
    Lref = cholesky(egrss.full(Ut,Vt,alpha)).L
    L = egrss.full_tril(Ut,Wt,1.0)
    @test( isapprox(L,Lref./diag(Lref)') )
    @test( isapprox(map(sqrt,c),diag(Lref)) )

    d = collect((1:n)/n) .+ 1e-3
    Wt,c = egrss.ldl(Ut,Vt,d)
    Lref = cholesky(K + Diagonal(d)).L
    L = egrss.full_tril(Ut,Wt,1.0)
    @test( isapprox(L,Lref./diag(Lref)') )
    @test( isapprox(map(sqrt,c),diag(Lref)) )

end
