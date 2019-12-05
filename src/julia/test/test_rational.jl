let
    p = 3
    n = BigInt(8)
    t = (1:n)//n
    Ut,Vt = egrss.generators(t,p)

    K = egrss.full(Ut,Vt)
    xt = copy(t)
    @test ( K*xt == egrss.symv(Ut,Vt,xt) )

    Wt,c = egrss.ldl(Ut,Vt)
    L = egrss.full_tril(Ut,Wt,ones(eltype(Ut),n))
    D = Diagonal(c)
    @test ( K == L*D*L' )

    b = K*xt
    z = egrss.trsv(Ut,Wt,ones(eltype(Ut),n),b,'N')
    z = z ./ c
    x = egrss.trsv(Ut,Wt,ones(eltype(Ut),n),z,'T')
    @test ( x == xt )

end
