let
    using Random
    Random.seed!(0)

    n = 50
    p = 2
    t = (1:n)/n
    x = randn(n)
    Ut,Vt = egrss.generators(t,p);
    K = egrss.full(Ut,Vt)

    Lref = cholesky(K).L
    Wt = egrss.potrf(Ut,Vt)
    bref = Lref*x
    b = egrss.trmv(Ut,Wt,x)
    @test ( isapprox(b,bref) )
    b = egrss.trmv(Ut,Wt,x,'N')
    @test ( isapprox(b,bref) )
    bref = Lref'*x
    b = egrss.trmv(Ut,Wt,x,'T')
    @test ( isapprox(b,bref) )

    d = collect((1:n)/n) .+ 1e-3
    Lref = cholesky(egrss.full(Ut,Vt,d)).L;
    Wt,c = egrss.potrf(Ut,Vt,d)
    bref = Lref*x
    b = egrss.trmv(Ut,Wt,c,x)
    @test ( isapprox(b,bref) )
    b = egrss.trmv(Ut,Wt,c,x,'N');
    @test ( isapprox(b,bref) )
    bref = Lref'*x;
    b = egrss.trmv(Ut,Wt,c,x,'T');
    @test ( isapprox(b,bref) )
end
