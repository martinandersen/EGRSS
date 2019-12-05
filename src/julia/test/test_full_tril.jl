let
    n = 50
    p = 3
    t = (1:n)/n
    Ut,Vt = egrss.generators(t,p);

    Wt = egrss.potrf(Ut,Vt)
    L = LowerTriangular(tril(Ut'*Wt))
    @test ( isapprox(egrss.full_tril(Ut,Wt),L) )

    Wt,c = egrss.potrf(Ut,Vt,1e-3*ones(n))
    L = LowerTriangular(tril(Ut'*Wt,-1)+Diagonal(c[:]))
    @test ( isapprox(egrss.full_tril(Ut,Wt,c),L) )
end
