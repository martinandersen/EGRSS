let
    n = 50
    p = 3
    t = (1:n)/n
    Ut,Vt = egrss.generators(t,p);

    @test ( size(Ut) == (p,n) )
    @test ( size(Vt) == (p,n) )

    K = Symmetric(tril(Ut'*Vt) + triu(Vt'*Ut,1))
    cholesky(K)
end
