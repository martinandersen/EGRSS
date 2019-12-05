let
    n = 50
    p = 3
    t = (1:n)/n
    Ut,Vt = egrss.generators(t,p);

    K = Symmetric(tril(Ut'*Vt) + triu(Vt'*Ut,1))
    @test ( isapprox(egrss.full(Ut,Vt),K) )
    @test ( isapprox(egrss.full(Ut,Vt,1e-3),K+Diagonal(1e-3*ones(n))) )
    @test ( isapprox(egrss.full(Ut,Vt,1e-3*ones(n)),K+Diagonal(1e-3*ones(n))) )
end
