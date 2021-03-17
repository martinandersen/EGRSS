import unittest
import numpy as np
import numpy.testing as npt

class TestEGRSS(unittest.TestCase):

    def test_generators(self):
        import egrss
        n,p = (50,3)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p);
        self.assertEqual(Ut.shape, (p,n))
        self.assertEqual(Vt.shape, (p,n))

    def test_full(self):
        import egrss
        n,p = (50,3)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p)

        K = np.tril(Ut.T@Vt) + np.triu(Vt.T@Ut,1)
        npt.assert_almost_equal(egrss.full(Ut,Vt),K)
        npt.assert_almost_equal(egrss.full(Ut,Vt,1e-3),K+np.diag(1e-3*np.ones(n)))
        npt.assert_almost_equal(egrss.full(Ut,Vt,1e-3*np.ones(n)),K+np.diag(1e-3*np.ones(n)))

    def test_symv(self):
        import egrss
        p,n = (3,12)
        Ut = np.random.randn(p,n)
        Vt = np.random.randn(p,n)
        x = np.random.randn(n)
        A = egrss.full(Ut,Vt)
        yref = A@x
        y = egrss.symv(Ut,Vt,x)
        npt.assert_almost_equal(y,yref)

    def test_gemv(self):
        import egrss
        p,q,m,n = (4,5,8,8)
        np.random.seed(0)
        Ut = np.random.randn(p,m)
        Vt = np.random.randn(p,n)
        Pt = np.random.randn(q,m)
        Qt = np.random.randn(q,n)
        x = np.random.randn(n)
        A = np.tril(Ut.T@Vt) + np.triu(Pt.T@Qt,1)
        yref = A@x
        y = egrss.gemv(Ut,Vt,Pt,Qt,x)
        npt.assert_almost_equal(y,yref)

        # Fat matrix
        p,q,m,n = (3,4,8,11)
        np.random.seed(0)
        Ut = np.random.randn(p,m)
        Vt = np.random.randn(p,n)
        Pt = np.random.randn(q,m)
        Qt = np.random.randn(q,n)
        x = np.random.randn(n)
        A = np.tril(Ut.T@Vt) + np.triu(Pt.T@Qt,1)
        yref = A@x
        y = egrss.gemv(Ut,Vt,Pt,Qt,x)
        npt.assert_almost_equal(y,yref)
        y = egrss.gemv(Ut[:,:m],Vt,Pt,Qt,x)
        npt.assert_almost_equal(y,yref)

        # Tall matrix
        p,q,m,n = (3,2,8,6)
        np.random.seed(0)
        Ut = np.random.randn(p,m)
        Vt = np.random.randn(p,n)
        Pt = np.random.randn(q,m)
        Qt = np.random.randn(q,n)
        x = np.random.randn(n)
        A = np.tril(Ut.T@Vt) + np.triu(Pt.T@Qt,1)
        yref = A@x
        y = egrss.gemv(Ut,Vt,Pt,Qt,x)
        npt.assert_almost_equal(y,yref)
        y = egrss.gemv(Ut,Vt,Pt[:,:n],Qt,x);
        npt.assert_almost_equal(y,yref)

    def test_tril(self):
        import egrss
        n,p = (50,3)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p)

        Wt = egrss.potrf(Ut,Vt)
        L = np.tril(Ut.T@Wt)
        npt.assert_almost_equal(egrss.full_tril(Ut,Wt),L)

        Wt,c = egrss.potrf(Ut,Vt,1e-3)
        L = np.tril(Ut.T@Wt,-1) + np.diag(c)
        npt.assert_almost_equal(egrss.full_tril(Ut,Wt,c),L)

    def test_potrf(self):
        import egrss
        n,p = (50,3)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p)

        K = egrss.full(Ut,Vt)

        Wt = egrss.potrf(Ut,Vt)
        Lref = np.linalg.cholesky(K)
        L = egrss.full_tril(Ut,Wt)
        npt.assert_almost_equal(L,Lref)

        alpha = 1e-6
        Wt,c = egrss.potrf(Ut,Vt,alpha)
        Lref = np.linalg.cholesky(K + np.diag(alpha*np.ones(n)))
        L = egrss.full_tril(Ut,Wt,c)
        npt.assert_almost_equal(L,Lref)

        d = t + 1e-3
        Wt,c = egrss.potrf(Ut,Vt,d)
        Lref = np.linalg.cholesky(K + np.diag(d))
        L = egrss.full_tril(Ut,Wt,c)
        npt.assert_almost_equal(L,Lref)

    def test_ldl(self):
        import egrss
        n,p = (50,2)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p)

        K = egrss.full(Ut,Vt)

        Wt,c = egrss.ldl(Ut,Vt)
        Lref = np.linalg.cholesky(K)
        L = egrss.full_tril(Ut,Wt,1.0)
        npt.assert_almost_equal(np.sqrt(c),np.diag(Lref))
        npt.assert_almost_equal(L,Lref/np.diag(Lref).T)

        alpha = 1e-6
        Wt,c = egrss.ldl(Ut,Vt,alpha)
        Lref = np.linalg.cholesky(egrss.full(Ut,Vt,alpha))
        L = egrss.full_tril(Ut,Wt,1.0)
        npt.assert_almost_equal(np.sqrt(c),np.diag(Lref))
        npt.assert_almost_equal(L,Lref/np.diag(Lref).T)

        d = t + 1e-3
        Wt,c = egrss.ldl(Ut,Vt,d)
        Lref = np.linalg.cholesky(egrss.full(Ut,Vt,d))
        L = egrss.full_tril(Ut,Wt,1.0)
        npt.assert_almost_equal(np.sqrt(c),np.diag(Lref))
        npt.assert_almost_equal(L,Lref/np.diag(Lref).T)


    def test_trmv(self):
        import egrss
        n,p = (50,3)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p)

        np.random.seed(0)
        x = np.random.randn(n)
        K = egrss.full(Ut,Vt)

        Lref = np.linalg.cholesky(K)
        Wt = egrss.potrf(Ut,Vt)
        bref = Lref@x
        b = egrss.trmv(Ut,Wt,x)
        npt.assert_almost_equal(b,bref)
        b = egrss.trmv(Ut,Wt,x,trans='N')
        npt.assert_almost_equal(b,bref)
        bref = Lref.T@x
        b = egrss.trmv(Ut,Wt,x,trans='T')
        npt.assert_almost_equal(b,bref)

        d = t + 1e-3
        Lref = np.linalg.cholesky(egrss.full(Ut,Vt,d))
        Wt,c = egrss.potrf(Ut,Vt,d)
        bref = Lref@x
        b = egrss.trmv(Ut,Wt,x,c)
        npt.assert_almost_equal(b,bref)
        b = egrss.trmv(Ut,Wt,x,c,trans='N');
        npt.assert_almost_equal(b,bref)
        bref = Lref.T@x;
        b = egrss.trmv(Ut,Wt,x,c,trans='T');
        npt.assert_almost_equal(b,bref)


    def test_trsv(self):
        import egrss
        n,p = (50,2)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p)

        np.random.seed(0)
        b = np.random.randn(n)
        K = egrss.full(Ut,Vt)

        Lref = np.linalg.cholesky(K)
        Wt = egrss.potrf(Ut,Vt)
        xref = np.linalg.solve(Lref,b)
        x = egrss.trsv(Ut,Wt,b)
        npt.assert_almost_equal(x,xref,decimal=5)
        x = egrss.trsv(Ut,Wt,b,trans='N')
        npt.assert_almost_equal(x,xref,decimal=5)
        xref = np.linalg.solve(Lref.T,b)
        x = egrss.trsv(Ut,Wt,b,trans='T')
        npt.assert_almost_equal(x,xref,decimal=5)

        d = t + 1e-3
        Lref = np.linalg.cholesky(egrss.full(Ut,Vt,d))
        Wt,c = egrss.potrf(Ut,Vt,d)
        xref = np.linalg.solve(Lref,b)
        x = egrss.trsv(Ut,Wt,b,c)
        npt.assert_almost_equal(x,xref)
        x = egrss.trsv(Ut,Wt,b,c,trans='N');
        npt.assert_almost_equal(x,xref)
        xref = np.linalg.solve(Lref.T,b)
        x = egrss.trsv(Ut,Wt,b,c,trans='T');
        npt.assert_almost_equal(x,xref)


    def test_trtri(self):
        import egrss
        n,p = (21,3)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p)

        Wt,c = egrss.potrf(Ut,Vt,1e-2)
        Yt,Zt = egrss.trtri(Ut,Wt,c)
        L = egrss.full_tril(Ut,Wt,c)
        Linv_ref = np.tril(np.linalg.inv(L))

        invL = egrss.full_tril(Yt,Zt,1.0/c)
        npt.assert_almost_equal(invL,Linv_ref)

        invL = egrss.trtri2(Ut,Wt,c);
        npt.assert_almost_equal(invL,Linv_ref)

    def test_trnrms(self):
        import egrss
        n,p = (50,3)
        t = np.linspace(0.02,1.0,n)
        Ut,Vt = egrss.generators(t,p)

        Wt,c = egrss.potrf(Ut,Vt,1e-4*np.ones(n))
        Lref = egrss.full_tril(Ut,Wt,c)
        nrmref = np.sum(Lref**2,axis=0).flatten()
        nrm = egrss.trnrms(Ut,Wt,c)
        npt.assert_almost_equal(nrm,nrmref)

if __name__ == '__main__':
    unittest.main()
