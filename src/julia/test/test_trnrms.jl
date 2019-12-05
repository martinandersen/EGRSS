let
    n = 50
    p = 3
    t = (1:n)/n
    Ut,Vt = egrss.generators(t,p);
    Wt,c = egrss.potrf(Ut,Vt,1e-4*ones(n))

    Lref = egrss.full_tril(Ut,Wt,c)
    nrmref = sum(Lref.^2,dims=1)'
    nrm = egrss.trnrms(Ut,Wt,c)
    @test ( isapprox(nrm,nrmref) )
end
