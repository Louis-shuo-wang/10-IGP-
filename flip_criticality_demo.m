function flip_criticality_demo()
% FLIP_CRITICALITY_DEMO
%   Refine a period-doubling (flip) point (mu=-1) and compute the cubic
%   normal-form coefficient sigma_flip on the 1D center manifold for the
%   Euler map Phi(X;gamma) = X + dt * f(X;gamma) of the nondimensional
%   Leslie–Gower IGP model (3D).
%
%   Classification: a stable 2-cycle is born if sigma_flip * beta < 0,
%   where beta = d(mu)/d(gamma) at the flip point.

%% ---------- Model parameters (tune) ----------
par.alpha   = 1.0;
par.beta    = 2.0;
par.delta   = 0.5;
par.epsilon = 0.2;

dt      = 0.10;            % Euler step
gamma0  = 0.295;           % initial guess near a flip (from your continuation)
X0guess = [0.31; 0.18; 0.44];

% Newton options
newton.maxit   = 30;
newton.tol     = 1e-12;
newton.verbose = false;

% Flip refinement options
opts.maxit     = 40;
opts.tol_fun   = 1e-11;     % target: |mu + 1| -> 0
opts.tol_gamma = 1e-11;
opts.imagtol   = 1e-8;      % enforce real eigenvalue near -1
opts.bracket0  = 1e-3;      % initial half-width for bracketing in gamma
opts.bracketMax= 0.2;

% Finite-difference step for multilinear derivatives
fd_eps = 1e-6;

fprintf('--- Refining flip point (mu = -1) and computing sigma_flip ---\n');

%% 1) Solve for fixed point near gamma0
[Xstar, ok] = newton_fp(X0guess, gamma0, par, dt, newton);
assert(ok, 'Initial Newton failed; adjust X0guess/gamma0.');

% 2) Refine gamma to mu = -1 (real)
[gF, XF, muF, v, w] = refine_flip(gamma0, Xstar, par, dt, newton, opts);
fprintf('Refined flip: gamma = %.12g, mu = %.12g\n', gF, real(muF));

% 3) Compute cubic normal-form coefficient on center manifold
A  = DPhi(XF, gF, par, dt);
% ---- h2 from (I - A)_Q h2 = Q B(v,v)
Q  = eye(3) - v*(w.');
Bf = @(u,vv) B_map(XF, gF, par, dt, u, vv, fd_eps);
Cf = @(u,vv,ww) C_map(XF, gF, par, dt, u, vv, ww, fd_eps);

rhs2 = Q * Bf(v, v);
h2   = solve_on_Q( (eye(3)-A), v, w, rhs2 );

% ---- a2 and a3 in projection coordinates
a2 = 0.5 * ( w.' * (A*h2) + w.' * Bf(v, v) );

% h3 from (I + A)_Q h3 = -6 a2 h2 - 6 Q B(v,h2) - Q C(v,v,v)
rhs3 = -6*a2*h2 - 6*(Q*Bf(v, h2)) - (Q*Cf(v, v, v));
h3   = solve_on_Q( (eye(3)+A), v, w, rhs3 );

a3 = w.' * Bf(v, h2) + (1/6) * (w.' * Cf(v, v, v));

% ---- cubic coefficient in (odd) normal form (after removing quadratic)
sigma_flip = real( a3 + (3/2)*(a2^2) );

% 4) Crossing speed beta = d(mu)/d(gamma) at flip (finite-difference)
beta = mu_slope_gamma(XF, gF, par, dt, newton);

% 5) Report & classify
fprintf('  a2 = %.6e, a3 = %.6e\n', a2, a3);
fprintf('  sigma_flip = %.6e\n', sigma_flip);
fprintf('  beta (d mu / d gamma) = %.6e\n', beta);
if sigma_flip*beta < 0
    verdict = 'SUPERCRITICAL (stable 2-cycle is born)';
else
    verdict = 'SUBCRITICAL (unstable 2-cycle; possible bistability)';
end
fprintf('  -> Flip criticality: %s\n', verdict);
fprintf('--- Done ---\n');

end
%% ================== Flip refinement (mu = -1) ==================

function [gF, XF, muF, v, w] = refine_flip(gamma0, X0, par, dt, newton, opts)
% Root-find for phi(gamma) = mu(gamma) + 1 = 0 with real mu near -1

% Solve FP at initial gamma
[X, ok] = newton_fp(X0, gamma0, par, dt, newton);
assert(ok, 'Newton failed at initial gamma0.');
[mu0, v0, w0, isReal0] = real_eig_target(DPhi(X, gamma0, par, dt), opts.imagtol, -1);
phi0 = real(mu0) + 1;
if ~isReal0
    warning('Initial target eigenvalue not real; proceeding with bracketing.');
end

% Bracket gamma where phi changes sign
[gL, XL, muL, ~, ~, phiL, gR, XR, muR, ~, ~, phiR] = ...
    bracket_gamma_flip(gamma0, X, phi0, par, dt, newton, opts);

% Secant/bisection
for it = 1:opts.maxit
    gN = gR - phiR*(gR - gL)/(phiR - phiL);
    if ~isfinite(gN), gN = 0.5*(gL + gR); end

    [XN, okN] = newton_fp(XR, gN, par, dt, newton);
    if ~okN
        gN = 0.5*(gL + gR);
        [XN, okN] = newton_fp(XR, gN, par, dt, newton);
        assert(okN, 'Newton failed during flip refinement.');
    end

    [muN, vN, wN, isRealN] = real_eig_target(DPhi(XN, gN, par, dt), opts.imagtol, -1);
    phiN = real(muN) + 1;

    if phiL*phiN <= 0
        gR=gN; XR=XN; muR=muN; phiR=phiN;
    else
        gL=gN; XL=XN; muL=muN; phiL=phiN;
    end

    if abs(phiN) < opts.tol_fun || abs(gR-gL) < opts.tol_gamma
        gF = gN; XF = XN; muF = muN; v=vN; w=wN;
        % normalize left/right: w^T v = 1
        s = (w.'*v); w = w / s; % v unchanged
        % enforce mu exactly real if tiny imag
        muF = real(muF);
        return
    end
    if ~isRealN
        warning('Non-real eigenvalue encountered during refinement; continue bracketing.');
    end
end
error('Flip refinement did not converge.');
end

function [gL, XL, muL, vL, wL, phiL, gR, XR, muR, vR, wR, phiR] = ...
    bracket_gamma_flip(g0, X0, phi0, par, dt, newton, opts)
half = opts.bracket0; maxH = opts.bracketMax;

gL=g0; XL=X0; muL=NaN; vL=[]; wL=[]; phiL=phi0;
gR=g0; XR=X0; muR=NaN; vR=[]; wR=[]; phiR=phi0;

for tries=1:40
    gLt = g0 - half;
    [XLt, ok] = newton_fp(XL, gLt, par, dt, newton);
    if ok
        [muLt, vLt, wLt, isRealLt] = real_eig_target(DPhi(XLt, gLt, par, dt), opts.imagtol, -1);
        phiLt = real(muLt) + 1;
        if isRealLt && sign(phiLt) ~= sign(phi0)
            gL=gLt; XL=XLt; muL=muLt; vL=vLt; wL=wLt; phiL=phiLt;
        end
    end

    gRt = g0 + half;
    [XRt, ok] = newton_fp(XR, gRt, par, dt, newton);
    if ok
        [muRt, vRt, wRt, isRealRt] = real_eig_target(DPhi(XRt, gRt, par, dt), opts.imagtol, -1);
        phiRt = real(muRt) + 1;
        if isRealRt && sign(phiRt) ~= sign(phi0)
            gR=gRt; XR=XRt; muR=muRt; vR=vRt; wR=wRt; phiR=phiRt;
        end
    end

    if isfinite(phiL) && isfinite(phiR) && sign(phiL) ~= sign(phiR)
        return
    end
    half = min(2*half, maxH);
end
error('Failed to bracket mu=-1 around gamma0; adjust guesses.');
end

function [mu, v, w, isReal] = real_eig_target(DP, imagtol, target)
% Pick the eigenvalue closest (in absolute distance) to 'target' (default -1)
if nargin<3, target = -1; end
[V,D] = eig(DP); lam = diag(D);
[~,ix] = min(abs(lam - target));
mu = lam(ix); v = V(:,ix);

% left eigenvector from transpose
[VLt,DLt] = eig(DP.'); lamLt = diag(DLt);
[~,jx] = min(abs(lamLt - mu));
w = conj(VLt(:,jx));
% normalize w^T v = 1
s = (w.'*v); w = w / s;

isReal = (abs(imag(mu)) < imagtol);
mu = real(mu) + 1i*0; % force real if tiny imag
end

%% ============== Normal-form building blocks (B, C, resolvents) ==============

function x = solve_on_Q(M, v, w, b)
% Solve on Q-subspace (Q = I - v w^T): find x s.t. M x = b, w^T x = 0.
K = [M, v; w.', 0];
rhs = [b; 0];
sol = K \ rhs;
x = sol(1:3);
end

function Buv = B_map(X, gamma, par, dt, u, v, eps)
% Symmetric bilinear: B(u,v) = D^2 Phi[X][u,v] = dt * D^2 F[X][u,v]
if nargin<7, eps = 1e-6; end
epsv = eps*max(1,norm(v));
Jp = J_f(X + epsv*v, gamma, par);
Jm = J_f(X - epsv*v, gamma, par);
D2F_uv = (Jp*u - Jm*u)/(2*epsv);

epsu = eps*max(1,norm(u));
Jp2 = J_f(X + epsu*u, gamma, par);
Jm2 = J_f(X - epsu*u, gamma, par);
D2F_vu = (Jp2*v - Jm2*v)/(2*epsu);

Buv = 0.5 * dt * (D2F_uv + D2F_vu);
end

function Cuvw = C_map(X, gamma, par, dt, u, v, w, eps)
% Trilinear: C(u,v,w) = dt * D^3F[X][u,v,w] via FD of B
if nargin<8, eps = 1e-6; end
Bw_p = B_atX(X + eps*w, gamma, par, dt, u, v);
Bw_m = B_atX(X - eps*w, gamma, par, dt, u, v);
Cw   = (Bw_p - Bw_m)/(2*eps);

Bu_p = B_atX(X + eps*u, gamma, par, dt, v, w);
Bu_m = B_atX(X - eps*u, gamma, par, dt, v, w);
Cu   = (Bu_p - Bu_m)/(2*eps);

Bv_p = B_atX(X + eps*v, gamma, par, dt, w, u);
Bv_m = B_atX(X - eps*v, gamma, par, dt, w, u);
Cv   = (Bv_p - Bv_m)/(2*eps);

Cuvw = (Cw + Cu + Cv)/3;
end

function Buv = B_atX(X, gamma, par, dt, u, v)
eps = 1e-6;
epsv = eps*max(1,norm(v));
Jp = J_f(X + epsv*v, gamma, par);
Jm = J_f(X - epsv*v, gamma, par);
D2F_uv = (Jp*u - Jm*u)/(2*epsv);

epsu = eps*max(1,norm(u));
Jp2 = J_f(X + epsu*u, gamma, par);
Jm2 = J_f(X - epsu*u, gamma, par);
D2F_vu = (Jp2*v - Jm2*v)/(2*epsu);

Buv = 0.5 * dt * (D2F_uv + D2F_vu);
end

%% ===================== Beta (d mu / d gamma) =====================

function beta = mu_slope_gamma(XF, gF, par, dt, newton)
% Finite-difference derivative of the flip multiplier wrt gamma
dg = 1e-6 * max(1,abs(gF));
% right step
[XR, ok] = newton_fp(XF, gF+dg, par, dt, newton); assert(ok);
muR = target_mu(DPhi(XR, gF+dg, par, dt));
% left step
[XL, ok] = newton_fp(XF, gF-dg, par, dt, newton); assert(ok);
muL = target_mu(DPhi(XL, gF-dg, par, dt));
beta = real(muR - muL)/(2*dg);
end

function mu = target_mu(DP)
% eigenvalue closest to -1 (assumed real near flip)
lam = eig(DP);
[~,ix] = min(abs(lam + 1));
mu = lam(ix);
end

%% ===================== Fixed-point & derivatives =====================

function [X, ok] = newton_fp(X0, gamma, par, dt, newton)
X = X0; ok = false;
for it=1:newton.maxit
    [G, JG] = G_and_JG(X, gamma, par, dt);
    if norm(G,2) < newton.tol, ok = true; return; end
    X = X - JG\G;
end
ok = (norm(G_and_JG(X,gamma,par,dt)) < 10*newton.tol);
end

function [G,JG] = G_and_JG(X, gamma, par, dt)
[f1,f2,f3] = f_rhs(X, gamma, par);
G  = dt*[f1; f2; f3];
JG = dt*J_f(X, gamma, par);
end

function A = DPhi(X, gamma, par, dt)
A = eye(3) + dt * J_f(X, gamma, par);
end

function [f1,f2,f3] = f_rhs(X, gamma, par)
x=X(1); y=X(2); z=X(3); z = max(z,1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
f1 = x*(1 - x/z) - x*y;
f2 = a*y*(1 - b*y/z) + x*y;
f3 = z*(gamma - d*x - e*y);
end

function J = J_f(X, gamma, par)
x=X(1); y=X(2); z=X(3); z=max(z,1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
J = [ (1-2*x/z)-y , -x , (x^2)/(z^2) ;
    y, a*(1-2*b*y/z)+x , a*b*(y^2)/(z^2) ;
    -d*z, -e*z, gamma-d*x-e*y ];
end