let
    using Random
    Random.seed!(0)

    n = 50
    p = 2
    t = (1:n)/n
    b = randn(n)
    Ut,Vt = egrss.generators(t,p);
    K = egrss.full(Ut,Vt)

    Lref = cholesky(K).L
    Wt = egrss.potrf(Ut,Vt)
    xref = Lref\b
    x = egrss.trsv(Ut,Wt,b)
    @test ( isapprox(x,xref) )
    x = egrss.trsv(Ut,Wt,b,'N')
    @test ( isapprox(x,xref) )
    xref = Lref'\b
    x = egrss.trsv(Ut,Wt,b,'T')
    @test ( isapprox(x,xref) )

    d = collect((1:n)/n) .+ 1e-3
    Lref = cholesky(egrss.full(Ut,Vt,d)).L;
    Wt,c = egrss.potrf(Ut,Vt,d)
    xref = Lref\b
    x = egrss.trsv(Ut,Wt,c,b)
    @test ( isapprox(x,xref) )
    x = egrss.trsv(Ut,Wt,c,b,'N');
    @test ( isapprox(x,xref) )
    xref = Lref'\b;
    x = egrss.trsv(Ut,Wt,c,b,'T');
    @test ( isapprox(x,xref) )
end
