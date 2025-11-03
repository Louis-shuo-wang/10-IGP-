function ns_criticality_demo()
% NS_CRITICALITY_DEMO
% Refines a Neimark–Sacker point for the Euler map Phi(X;gamma)=X+dt*f(X;gamma)
% of the nondimensional Leslie–Gower IGP model, and computes the
% first Lyapunov coefficient l1 for maps (criticality: l1<0 supercritical).
%
% Run: ns_criticality_demo

% -------- Parameters (tune as desired) --------
par.alpha   = 1.0;
par.beta    = 2.0;
par.delta   = 0.5;
par.epsilon = 0.2;

dt     = 0.10;             % Euler step
gamma0 = 0.260;            % initial guess near an NS point
X0     = [0.32;0.19;0.45]; % initial fixed-point guess

% Newton & refinement options
newton.maxit   = 30;
newton.tol     = 1e-12;
newton.verbose = false;

optsNS.maxit   = 40;
optsNS.tol_fun  = 1e-10; % target |mu|-1
optsNS.tol_gamma = 1e-10;
optsNS.imagtol    = 1e-8;
optsNS.bracket0   = 1e-3;
optsNS.bracketMax = 0.2;

% ---- 1) fixed point near gamma0
[Xstar, ok] = newton_fp(X0, gamma0, par, dt, newton);
assert(ok, 'Initial Newton failed; adjust X0/gamma0.');

% ---- 2) refine gamma s.t. |mu|=1
[gammaNS, XNS, mu, q, p, info] = refine_NS(gamma0, Xstar, par, dt, newton, optsNS);
fprintf('\nRefined NS: gamma = %.12g\n', gammaNS);
theta = angle(mu);
fprintf('  mu = exp(i*theta), theta = %.8f rad (%.6f deg)\n', theta, theta*180/pi);
fprintf('  nonresonance k=1..4: %s\n', mat2str(info.nonres_ok));

% ---- 3) compute first Lyapunov coefficient l1
A    = DPhi(XNS, gammaNS, par, dt);
ell1 = first_Lyapunov_coeff(XNS, gammaNS, par, dt, A, mu, q, p);
fprintf('  first Lyapunov coefficient l1 = %.6e  =>  %s NS\n', ell1, ...
    ternary(ell1<0,'supercritical (stable circle)','subcritical (unstable circle)'));

end

% ================= fixed-point solve & NS refinement =================

function [X, ok] = newton_fp(X0, gamma, par, dt, newton)
X = X0; ok = false;
for it=1:newton.maxit
    [G, JG] = G_and_JG(X, gamma, par, dt);
    if norm(G,2) < newton.tol, ok = true; return; end
    X = X - JG\G;
end
ok = (norm(G_and_JG(X,gamma,par,dt)) < 10*newton.tol);
end

function [gammaNS, XNS, mu, q, p, info] = refine_NS(gamma0, X0, par, dt, newton, opts)
[X,ok] = newton_fp(X0, gamma0, par, dt, newton); assert(ok);
[mu0,~,~,~] = unitcircle_target(DPhi(X, gamma0, par, dt), opts.imagtol);
phi0 = abs(mu0)-1;

[gL, XL, ~, ~, ~, phiL, gR, XR, ~, ~, ~, phiR] = ...
    bracket_gamma_NS(gamma0, X, phi0, par, dt, newton, opts);

for it=1:opts.maxit
    gN = gR - phiR*(gR-gL)/(phiR-phiL);
    if ~isfinite(gN), gN = (gL+gR)/2; end
    [XN, okN] = newton_fp(XR, gN, par, dt, newton);
    if ~okN, gN=(gL+gR)/2; [XN,okN]=newton_fp(XR,gN,par,dt,newton); assert(okN); end
    [muN,qN,pN,~] = unitcircle_target(DPhi(XN,gN,par,dt), opts.imagtol);
    phiN = abs(muN)-1;
    if phiL*phiN<=0, gR=gN; XR=XN; phiR=phiN; else, gL=gN; XL=XN; phiL=phiN; end
    if abs(phiN)<opts.tol_fun || abs(gR-gL)<opts.tol_gamma
        gammaNS=gN; XNS=XN; mu=muN; q=qN; p=pN; info.nonres_ok = resonance_check(angle(mu),4); return
    end
end
error('NS refinement did not converge.');
end

function [gL, XL, muL, qL, pL, phiL, gR, XR, muR, qR, pR, phiR] = ...
    bracket_gamma_NS(g0, X0, phi0, par, dt, newton, opts)
half=opts.bracket0; maxH=opts.bracketMax;
gL=g0; XL=X0; muL=NaN; qL=[]; pL=[]; phiL=phi0;
gR=g0; XR=X0; muR=NaN; qR=[]; pR=[]; phiR=phi0;
for tries=1:40
    gLt=g0-half; [XLt,ok]=newton_fp(XL,gLt,par,dt,newton);
    if ok, [muLt, qLt, pLt, ~]=unitcircle_target(DPhi(XLt,gLt,par,dt),opts.imagtol);
        phiLt=abs(muLt)-1; if sign(phiLt)~=sign(phi0), gL=gLt; XL=XLt; muL=muLt; qL=qLt; pL=pLt; phiL=phiLt; end
    end
    gRt=g0+half; [XRt,ok]=newton_fp(XR,gRt,par,dt,newton);
    if ok, [muRt, qRt, pRt, ~]=unitcircle_target(DPhi(XRt,gRt,par,dt),opts.imagtol);
        phiRt=abs(muRt)-1; if sign(phiRt)~=sign(phi0), gR=gRt; XR=XRt; muR=muRt; qR=qRt; pR=pRt; phiR=phiRt; end
    end
    if isfinite(phiL) && isfinite(phiR) && sign(phiL)~=sign(phiR), return; end
    half=min(2*half,maxH);
end
error('Failed to bracket |mu|-1=0 around gamma0.');
end

function [mu, q, p, isComplex] = unitcircle_target(DP, imagtol)
[V,D]=eig(DP); lam=diag(D);
[~,idx]=sort(abs(abs(lam)-1),'ascend');
mu=NaN; q=[]; p=[]; isComplex=false;
for j=idx.'
    if abs(imag(lam(j)))>imagtol
        mu=lam(j); q=V(:,j);
        [VLt,DLt]=eig(DP.'); lamLt=diag(DLt);
        [~,jL]=min(abs(lamLt-mu)); p=conj(VLt(:,jL)); p=p/(p'*q);
        isComplex=true; if imag(mu)<0, mu=conj(mu); q=conj(q); p=conj(p); end; return
    end
end
j=idx(1); mu=lam(j); q=V(:,j);
[VLt,DLt]=eig(DP.'); lamLt=diag(DLt);
[~,jL]=min(abs(lamLt-mu)); p=conj(VLt(:,jL)); p=p/(p'*q);
end

function ok = resonance_check(theta,K)
tol=1e-6; ok=true(1,K);
for k=1:K, ok(k)=abs(exp(1i*k*theta)-1)>tol; end
end

% ================= l1: multilinear forms + resolvents =================

function l1 = first_Lyapunov_coeff(X, gamma, par, dt, A, mu, q, p)
B = @(u,v) B_map(X,gamma,par,dt,u,v);
C = @(u,v,w) C_map(X,gamma,par,dt,u,v,w);
Q = eye(3) - q*(p.');
M1 = eye(3) - conj(mu)*A; R1 = @(b) solve_on_Q(M1,q,p,Q*b);
M2 = 2*eye(3) - conj(mu)*A; R2 = @(b) solve_on_Q(M2,q,p,Q*b);
term1 = conj(mu) * (p.' * C(q,q,conj(q)));
term2 = conj(mu) * (p.' * B(q, R1(B(q,conj(q)))));
term3 = conj(mu) * (p.' * B(conj(q), R2(B(q,q))));
l1 = 0.5 * real(term1 - 2*term2 + term3);
end

function x = solve_on_Q(M,q,p,b)
K=[M,q; p.',0]; rhs=[b;0]; sol=K\rhs; x=sol(1:3);
end

function Buv = B_map(X,gamma,par,dt,u,v)
% Symmetric bilinear: B(u,v)=dt*D^2F[X][u,v] (FD directional)
eps0=1e-6; epsv=eps0*max(1,norm(v));
Jp=J_f(X+epsv*v,gamma,par); Jm=J_f(X-epsv*v,gamma,par); D2F_uv=(Jp*u-Jm*u)/(2*epsv);
epsu=eps0*max(1,norm(u));
Jp2=J_f(X+epsu*u,gamma,par); Jm2=J_f(X-epsu*u,gamma,par); D2F_vu=(Jp2*v-Jm2*v)/(2*epsu);
Buv=0.5*dt*(D2F_uv+D2F_vu);
end

function Cuvw = C_map(X,gamma,par,dt,u,v,w)
% Trilinear: C(u,v,w)=dt*D^3F[X][u,v,w] (FD of B)
eps0=1e-6;
Cuvw = ( ...
    (B_atX(X+eps0*w,gamma,par,dt,u,v)-B_atX(X-eps0*w,gamma,par,dt,u,v))/(2*eps0) + ...
    (B_atX(X+eps0*u,gamma,par,dt,v,w)-B_atX(X-eps0*u,gamma,par,dt,v,w))/(2*eps0) + ...
    (B_atX(X+eps0*v,gamma,par,dt,w,u)-B_atX(X-eps0*v,gamma,par,dt,w,u))/(2*eps0) )/3;
end

function Buv = B_atX(X,gamma,par,dt,u,v)
eps0=1e-6; epsv=eps0*max(1,norm(v));
Jp=J_f(X+epsv*v,gamma,par); Jm=J_f(X-epsv*v,gamma,par); D2F_uv=(Jp*u-Jm*u)/(2*epsv);
epsu=eps0*max(1,norm(u));
Jp2=J_f(X+epsu*u,gamma,par); Jm2=J_f(X-epsu*u,gamma,par); D2F_vu=(Jp2*v-Jm2*v)/(2*epsu);
Buv=0.5*dt*(D2F_uv+D2F_vu);
end

% ================= model & derivatives =================

function [G,JG]=G_and_JG(X,gamma,par,dt)
[f1,f2,f3]=f_rhs(X,gamma,par); Jf=J_f(X,gamma,par);
G = dt*[f1;f2;f3]; JG = dt*Jf;
end

function A=DPhi(X,gamma,par,dt)
A = eye(3)+dt*J_f(X,gamma,par);
end

function [f1,f2,f3]=f_rhs(X,gamma,par)
x=X(1); y=X(2); z=X(3); z=max(z,1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
f1 = x*(1 - x/z) - x*y;
f2 = a*y*(1 - b*y/z) + x*y;
f3 = z*(gamma - d*x - e*y);
end

function J=J_f(X,gamma,par)
x=X(1); y=X(2); z=X(3); z=max(z,1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
J = [ (1-2*x/z)-y , -x , (x^2)/(z^2) ;
    y, a*(1-2*b*y/z)+x , a*b*(y^2)/(z^2) ;
    -d*z, -e*z, gamma-d*x-e*y  ];
end

function s=ternary(c,a,b), if c, s=a; else, s=b; end, end