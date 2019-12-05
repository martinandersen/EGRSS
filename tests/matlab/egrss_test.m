function tests = egrss_test
tests = functiontests(localfunctions);
end

function setup(testCase)
N = 21;
p = 3;
t = linspace(1e-2,1,N);
testCase.TestData.t = t;
testCase.TestData.p = p;
[Ut,Vt] = splinekernel_ref(t,p);
testCase.TestData.Ut_ref = Ut;
testCase.TestData.Vt_ref = Vt;
end

function test_splinekernel(testCase)

% Test: monotonically increasing sequence
t = testCase.TestData.t;
p = testCase.TestData.p;
[Ut,Vt] = egrss_generators(t,p);
verifyEqual(testCase,Ut,testCase.TestData.Ut_ref,'RelTol',1e-12);
verifyEqual(testCase,Vt,testCase.TestData.Vt_ref,'RelTol',1e-12);

% Test: monotonically decreasing sequence
[Ut_ref,Vt_ref] = splinekernel_ref(fliplr(t),p);
[Ut,Vt] = egrss_generators(fliplr(t),p);
verifyEqual(testCase,Ut,Ut_ref,'RelTol',1e-12);
verifyEqual(testCase,Vt,Vt_ref,'RelTol',1e-12);
end

function test_full(testCase)
Ut = testCase.TestData.Ut_ref;
Vt = testCase.TestData.Vt_ref;

K = egrss_full(Ut,Vt);
Kref = tril(Ut'*Vt) + triu(Vt'*Ut,1);
verifyEqual(testCase,K,Kref,'RelTol',1e-12);

d = linspace(0.1,2.0,size(K,1));
K = egrss_full(Ut,Vt,d);
Kref = tril(Ut'*Vt) + triu(Vt'*Ut,1) + diag(d);
verifyEqual(testCase,K,Kref,'RelTol',1e-12);

d = 2.0;
K = egrss_full(Ut,Vt,d);
Kref = tril(Ut'*Vt) + triu(Vt'*Ut,1) + diag(2.0*ones(size(K,1),1));
verifyEqual(testCase,K,Kref,'RelTol',1e-12);
end

function test_gemv(testCase)

% Square matrix
rng(0);
p = 4; q = 5; m = 8; n = m;
Ut = randn(p,m); Vt = randn(p,n);
Pt = randn(q,m); Qt = randn(q,n);
x = randn(n,1);
A = tril(Ut'*Vt) + triu(Pt'*Qt,1);
yref = A*x;
y = egrss_gemv(Ut,Vt,Pt,Qt,x);
verifyEqual(testCase,y,yref,'RelTol',1e-12);

% Fat matrix
rng(0);
p = 3; q = 4; m = 8; n = m+3;
Ut = randn(p,m); Vt = randn(p,n);
Pt = randn(q,m); Qt = randn(q,n);
x = randn(n,1);
A = tril(Ut'*Vt) + triu(Pt'*Qt,1);
yref = A*x;
y = egrss_gemv(Ut,Vt,Pt,Qt,x);
verifyEqual(testCase,y,yref,'RelTol',1e-12);
y = egrss_gemv(Ut(:,1:m),Vt,Pt,Qt,x);
verifyEqual(testCase,y,yref,'RelTol',1e-12);

% Tall matrix
rng(0);
p = 3; q = 2; m = 8; n = m-2;
Ut = randn(p,m); Vt = randn(p,n);
Pt = randn(q,m); Qt = randn(q,n);
x = randn(n,1);
A = tril(Ut'*Vt) + triu(Pt'*Qt,1);
yref = A*x;
y = egrss_gemv(Ut,Vt,Pt,Qt,x);
verifyEqual(testCase,y,yref,'RelTol',1e-12);
y = egrss_gemv(Ut,Vt,Pt(:,1:n),Qt,x);
verifyEqual(testCase,y,yref,'RelTol',1e-12);
end

function test_symv(testCase)
rng(0);
p = 3; n = 12;
Ut = randn(p,n); Vt = randn(p,n);
x = randn(n,1);
A = tril(Ut'*Vt) + triu(Vt'*Ut,1);
yref = A*x;
y = egrss_symv(Ut,Vt,x);
verifyEqual(testCase,y,yref,'RelTol',1e-12);
end

function test_potrf(testCase)
t = testCase.TestData.t;
p = testCase.TestData.p;
Ut = testCase.TestData.Ut_ref;
Vt = testCase.TestData.Vt_ref;
K = tril(Ut'*Vt) + triu(Vt'*Ut,1);
disp(sprintf('\ncond(K) = %.2e\n',cond(K)))

Wt = egrss_potrf(Ut,Vt);
L = tril(Ut'*Wt);
Lref = chol(K)';
verifyEqual(testCase,L,Lref,'RelTol',1e-5);

alpha = 0.1;
[Wt,c] = egrss_potrf(Ut,Vt,alpha);
L = tril(Ut'*Wt,-1) + diag(c);
Lref = chol(K + diag(alpha*ones(size(K,1),1)))';
verifyEqual(testCase,L,Lref,'RelTol',1e-12);

d = linspace(0.1,2.0,size(K,1));
[Wt,c] = egrss_potrf(Ut,Vt,d);
L = tril(Ut'*Wt,-1) + diag(c);
Lref = chol(K + diag(d))';
verifyEqual(testCase,L,Lref,'RelTol',1e-12);
end

function test_ldl(testCase)
t = testCase.TestData.t;
p = testCase.TestData.p;
Ut = testCase.TestData.Ut_ref;
Vt = testCase.TestData.Vt_ref;
K = egrss_full(Ut,Vt);
disp(sprintf('\ncond(K) = %.2e\n',cond(K)))

[Wt,c] = egrss_ldl(Ut,Vt);
L = egrss_full_tril(Ut,Wt,1.0);
Lref = chol(K)';
verifyEqual(testCase,L,Lref./diag(Lref)','RelTol',1e-5);
verifyEqual(testCase,sqrt(c),diag(Lref),'RelTol',1e-5);

alpha = 0.1;
[Wt,c] = egrss_ldl(Ut,Vt,alpha);
L = egrss_full_tril(Ut,Wt,1.0);
Lref = chol(egrss_full(Ut,Vt,alpha))';
verifyEqual(testCase,L,Lref./diag(Lref)','RelTol',1e-12);
verifyEqual(testCase,sqrt(c),diag(Lref),'RelTol',1e-12);

d = linspace(0.1,2.0,size(K,1))';
[Wt,c] = egrss_ldl(Ut,Vt,d);
L = egrss_full_tril(Ut,Wt,1.0);
Lref = chol(egrss_full(Ut,Vt,d))';
verifyEqual(testCase,L,Lref./diag(Lref)','RelTol',1e-12);
verifyEqual(testCase,sqrt(c),diag(Lref),'RelTol',1e-12);
end


function test_trmv(testCase)
t = testCase.TestData.t;
p = testCase.TestData.p;
Ut = testCase.TestData.Ut_ref;
Vt = testCase.TestData.Vt_ref;

rng(0);
x = randn(length(t),1);

K = tril(Ut'*Vt) + triu(Vt'*Ut,1);
Wt = egrss_potrf(Ut,Vt);
Lref = tril(Ut'*Wt);
bref = Lref*x;
b = egrss_trmv(Ut,Wt,x);
verifyEqual(testCase,b,bref,'RelTol',1e-8);
b = egrss_trmv(Ut,Wt,x,'N');
verifyEqual(testCase,b,bref,'RelTol',1e-8);
bref = Lref'*x;
b = egrss_trmv(Ut,Wt,x,'T');
verifyEqual(testCase,b,bref,'RelTol',1e-8);

d = linspace(0.1,2.0,length(t));
Lref = chol(K + diag(d))';
[Wt,c] = egrss_potrf(Ut,Vt,d);
bref = Lref*x;
b = egrss_trmv(Ut,Wt,c,x);
verifyEqual(testCase,b,bref,'RelTol',1e-12);
b = egrss_trmv(Ut,Wt,c,x,'N');
verifyEqual(testCase,b,bref,'RelTol',1e-12);
bref = Lref'*x;
b = egrss_trmv(Ut,Wt,c,x,'T');
verifyEqual(testCase,b,bref,'RelTol',1e-12);
end

function test_trsv(testCase)
t = testCase.TestData.t;
p = testCase.TestData.p;
Ut = testCase.TestData.Ut_ref;
Vt = testCase.TestData.Vt_ref;

rng(0);
b = randn(length(t),1);

K = tril(Ut'*Vt) + triu(Vt'*Ut,1);
Wt = egrss_potrf(Ut,Vt);
Lref = tril(Ut'*Wt);
xref = Lref\b;
x = egrss_trsv(Ut,Wt,b);
verifyEqual(testCase,x,xref,'RelTol',1e-8);
x = egrss_trsv(Ut,Wt,b,'N');
verifyEqual(testCase,x,xref,'RelTol',1e-8);
xref = Lref'\b;
x = egrss_trsv(Ut,Wt,b,'T');
verifyEqual(testCase,x,xref,'RelTol',1e-8);

d = linspace(0.1,2.0,length(t));
[Wt,c] = egrss_potrf(Ut,Vt,d);
Lref = tril(Ut'*Wt,-1) + diag(c);
xref = Lref\b;
x = egrss_trsv(Ut,Wt,c,b);
verifyEqual(testCase,x,xref,'RelTol',1e-12);
x = egrss_trsv(Ut,Wt,c,b,'N');
verifyEqual(testCase,x,xref,'RelTol',1e-12);
xref = Lref'\b;
x = egrss_trsv(Ut,Wt,c,b,'T');
verifyEqual(testCase,x,xref,'RelTol',1e-12);
end

function test_trnrms(testCase)
t = testCase.TestData.t;
p = testCase.TestData.p;
Ut = testCase.TestData.Ut_ref;
Vt = testCase.TestData.Vt_ref;
[Wt,c] = egrss_potrf(Ut,Vt,1e-4);

Lref = tril(Ut'*Wt,-1) + diag(c);
nrmref = sum(Lref.^2,1)';
nrm = egrss_trnrms(Ut,Wt,c);
verifyEqual(testCase,nrm,nrmref,'RelTol',1e-12);
end

function test_trtri(testCase)
t = testCase.TestData.t;
p = testCase.TestData.p;
Ut = testCase.TestData.Ut_ref;
Vt = testCase.TestData.Vt_ref;
[Wt,c] = egrss_potrf(Ut,Vt,1e-2);
[Yt,Zt] = egrss_trtri(Ut,Wt,c);
L = tril(Ut'*Wt,-1)+diag(c);
Linv_ref = tril(inv(L));
% Implicit inverse
invL = tril(Yt'*Zt,-1)+diag(1./c);
verifyEqual(testCase,invL,Linv_ref,'RelTol',1e-8);
% Explicit inverse
invL = egrss_trtri2(Ut,Wt,c);
verifyEqual(testCase,invL,Linv_ref,'RelTol',1e-12);
end

function test_trace(testCase)
t = testCase.TestData.t;
p = testCase.TestData.p;
Ut = testCase.TestData.Ut_ref;
Vt = testCase.TestData.Vt_ref;
d = 1e-2*ones(length(t),1);
[Wt,c] = egrss_potrf(Ut,Vt,d);
[Yt,Zt] = egrss_trtri(Ut,Wt,c);
Kd = egrss_full(Ut,Vt,d);
L = tril(Ut'*Wt,-1)+diag(c);
tr_ref = trace(L\Kd/L');
tr = egrss_trace(Ut,Vt,d,Yt,Zt,c);
verifyEqual(testCase,tr,tr_ref,'RelTol',1e-8);
end

function [Ut,Vt] = splinekernel_ref(t,p)
Ut = (repmat(t,p,1).^([p-1:-1:0]'))./factorial([p-1:-1:0]');
Vt = (repmat(t,p,1).^([p:2*p-1]')).*((-1).^[0:p-1]')./factorial([p:2*p-1]');
if t(1) > t(end)
  % Swap Ut and Vt if t is increasing
  tmp = Ut;
  Ut = Vt;
  Vt = tmp;
end
end
