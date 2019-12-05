let
    using Random
    Random.seed!(0)
    p = 3
    n = 12
    Ut = randn(p,n)
    Vt = randn(p,n)
    x = randn(n)
    A = egrss.full(Ut,Vt)
    yref = A*x
    y = egrss.symv(Ut,Vt,x);
    @test ( isapprox(y,yref) )
end
