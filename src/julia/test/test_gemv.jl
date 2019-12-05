let
    using Random

    # Square matrix
    Random.seed!(0)
    p = 4; q = 5; m = 8; n = m;
    Ut = randn(p,m)
    Vt = randn(p,n)
    Pt = randn(q,m)
    Qt = randn(q,n)
    x = randn(n,1)
    A = tril(Ut'*Vt) + triu(Pt'*Qt,1)
    yref = A*x
    y = egrss.gemv(Ut,Vt,Pt,Qt,x)
    @test ( isapprox(y,yref) )

    # Fat matrix
    Random.seed!(0)
    p = 3; q = 4; m = 8; n = m+3;
    Ut = randn(p,m)
    Vt = randn(p,n)
    Pt = randn(q,m)
    Qt = randn(q,n)
    x = randn(n,1)
    A = tril(Ut'*Vt) + triu(Pt'*Qt,1)
    yref = A*x
    y = egrss.gemv(Ut,Vt,Pt,Qt,x)
    @test ( isapprox(y,yref) )
    y = egrss.gemv(Ut[:,1:m],Vt,Pt,Qt,x)
    @test ( isapprox(y,yref) )

    # Tall matrix
    Random.seed!(0)
    p = 3; q = 2; m = 8; n = m-2;
    Ut = randn(p,m)
    Vt = randn(p,n)
    Pt = randn(q,m)
    Qt = randn(q,n)
    x = randn(n,1)
    A = tril(Ut'*Vt) + triu(Pt'*Qt,1)
    yref = A*x
    y = egrss.gemv(Ut,Vt,Pt,Qt,x)
    @test ( isapprox(y,yref) )
    y = egrss.gemv(Ut,Vt,Pt[:,1:n],Qt,x);
    @test ( isapprox(y,yref) )

end
