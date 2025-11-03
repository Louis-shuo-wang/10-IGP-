function ns_refine_and_l1()
% NS_REFINE_AND_L1
%   Refine a Neimark–Sacker (discrete Hopf) point for the Euler map
%   Phi(X;gamma) = X + dt * f(X;gamma) of the nondimensional Leslie–Gower IGP,
%   and compute the first Lyapunov coefficient l1 for maps.
%
% What it does
%   1) Given an initial guess (X0,gamma0) near NS, repeatedly solves G(X,g)=0
%      (fixed-point equation) and uses a bracket+secant on phi(g)=|mu(g)|-1
%      (mu = complex multiplier with Im>0) to solve phi(g)=0.
%   2) At the refined point, computes l1 using multilinear forms:
%      l1 = 0.5 * Re[  conj(mu) p^T C(q,q,conj(q))
%                      - 2 conj(mu) p^T B(q, (I - conj(mu)A)^{-1}_Q B(q,conj(q)))
%                      + conj(mu) p^T B(conj(q), (2I - conj(mu)A)^{-1}_Q B(q,q)) ].
%
% Notes
%   • Finite-difference directional derivatives are used for B (2nd) and C (3rd).
%   • Projected resolvents are solved with a bordered system enforcing p^T x = 0.
%   • Nonresonance (k=1..4) is checked; l1 is reported with sign (super/subcritical).
%
% -------------------------------------------------------------------------

%% Model parameters (tune these as you like)
par.alpha  = 2.0;
par.beta   = 1.0;
par.delta  = 0.5;
par.epsilon = 1;

dt     = 0.1;     % Euler step
gamma0 = 0.2;  % initial gamma guess (should be near an NS flagged by your continuation)
X0     = [1; 1; 1];  % initial fixed-point guess (positive)

% Newton settings for fixed-point solves
newton.maxit  = 30;
newton.tol    = 1e-12;
newton.verbose = false;

% Refinement settings
optsNS.maxit   = 40;
optsNS.tol_fun   = 1e-10;  % for |mu|-1
optsNS.tol_gamma = 1e-10;
optsNS.imagtol   = 1e-8;   % to treat a multiplier as genuinely complex
optsNS.bracket0  = 1e-3;    % initial gamma bracketing half-width
optsNS.bracketMax= 0.2;     % max half-width expansion for bracketing

fprintf('--- Refining NS point and computing l1 ---\n');

% 1) Find the fixed point near gamma0
[Xstar, ok] = newton_fp(X0, gamma0, par, dt, newton);
assert(ok, 'Initial Newton failed. Adjust X0/gamma0.');

% 2) Refine gamma to |mu|=1 via bracket+secant (resolving FP each time)
[gammaNS, XNS, mu, q, p, info] = refine_NS(gamma0, Xstar, par, dt, newton, optsNS);

fprintf('\n[Refined NS] gamma = %.12g\n', gammaNS);
theta = angle(mu);   % principal angle in (-pi, pi]
fprintf('  mu = e^{i*theta} with theta = %.8f rad (%.6f deg)\n', theta, theta*180/pi);
fprintf('  nonresonance checks (k=1..4): %s\n', mat2str(info.nonres_ok));

% 3) Compute first Lyapunov coefficient l1 at the refined NS
A = DPhi(XNS, gammaNS, par, dt);
ell1 = first_Lyapunov_coeff(XNS, gammaNS, par, dt, A, mu, q, p);

fprintf('  first Lyapunov coefficient l1 = %.6e  ->  %s NS\n', ell1, ...
    ternary(ell1<0, 'supercritical (stable circle)', 'subcritical (unstable circle)'));

fprintf('--- Done ---\n');

end
% ======================= CORE ROUTINES =======================

function [gammaNS, XNS, mu, q, p, info] = refine_NS(gamma0, X0, par, dt, newton, opts)
% Refine gamma so that |mu(gamma)| = 1, where mu is the complex multiplier (Im>0)
% closest to the unit circle. At each gamma, solve G(X,gamma)=0 via Newton.

% Start by evaluating at gamma0
[X, ok] = newton_fp(X0, gamma0, par, dt, newton);
assert(ok, 'Newton failed at initial gamma0.');
[mu0, q0, p0, isComplex] = unitcircle_target(DPhi(X, gamma0, par, dt), opts.imagtol);
phi0 = abs(mu0) - 1;  % function to root-find

% Bracket in gamma
[gL, XL, muL, ~, ~, phiL, gR, XR, muR, ~, ~, phiR] = ...
    bracket_gamma_NS(gamma0, X, phi0, par, dt, newton, opts);

% Secant on phi(g)=|mu|-1
for it = 1:opts.maxit
    % Secant update
    gN = gR - phiR*(gR - gL)/(phiR - phiL);
    if ~isfinite(gN), gN = (gL + gR)/2; end

    [XN, okN] = newton_fp(XR, gN, par, dt, newton);    % use last solution as init
    if ~okN
        % fallback to bisection style
        gN = (gL + gR)/2;
        [XN, okN] = newton_fp(XR, gN, par, dt, newton);
        assert(okN, 'Newton failed during NS refinement.');
    end

    [muN, qN, pN, isComplexN] = unitcircle_target(DPhi(XN, gN, par, dt), opts.imagtol);
    phiN = abs(muN) - 1;

    % Update bracket
    if phiL*phiN <= 0
        gR = gN; XR = XN; muR = muN; phiR = phiN;
    else
        gL = gN; XL = XN; muL = muN; phiL = phiN;
    end

    % Convergence?
    if abs(phiN) < opts.tol_fun || abs(gR - gL) < opts.tol_gamma
        gammaNS = gN; XNS = XN; mu = muN; q = qN; p = pN;
        info.nonres_ok = resonance_check(angle(mu), 4);
        return
    end
end

error('NS refinement did not converge. Consider a better initial guess or loosening tolerances.');
end

function [gL, XL, muL, qL, pL, phiL, gR, XR, muR, qR, pR, phiR] = ...
    bracket_gamma_NS(g0, X0, phi0, par, dt, newton, opts)
% Expand a bracket [gL,gR] with opposite signs of phi(g)=|mu|-1.
half = opts.bracket0;
maxH = opts.bracketMax;

gL = g0;  XL = X0;  muL = NaN; qL = []; pL = []; phiL = phi0;
gR = g0;  XR = X0;  muR = NaN; qR = []; pR = []; phiR = phi0;

for tries = 1:40
    % Left
    gLtry = g0 - half;
    [XLtry, ok] = newton_fp(XL, gLtry, par, dt, newton);
    if ok
        [muLtry, qLtry, pLtry, ~] = unitcircle_target(DPhi(XLtry, gLtry, par, dt), opts.imagtol);
        phiLtry = abs(muLtry) - 1;
        if sign(phiLtry) ~= sign(phi0)
            gL = gLtry; XL = XLtry; muL = muLtry; qL = qLtry; pL = pLtry; phiL = phiLtry;
        end
    end

    % Right
    gRtry = g0 + half;
    [XRtry, ok] = newton_fp(XR, gRtry, par, dt, newton);
    if ok
        [muRtry, qRtry, pRtry, ~] = unitcircle_target(DPhi(XRtry, gRtry, par, dt), opts.imagtol);
        phiRtry = abs(muRtry) - 1;
        if sign(phiRtry) ~= sign(phi0)
            gR = gRtry; XR = XRtry; muR = muRtry; qR = qRtry; pR = pRtry; phiR = phiRtry;
        end
    end

    if isfinite(phiL) && isfinite(phiR) && sign(phiL) ~= sign(phiR)
        return
    end

    half = min(2*half, maxH);
end

error('Failed to bracket |mu|-1 = 0 around gamma0; widen search or adjust initial guess.');
end

function [mu, q, p, isComplex] = unitcircle_target(DP, imagtol)
% Pick the complex multiplier (Im>0) closest to the unit circle, with right/left eigenvectors.
[V,D] = eig(DP);
lam = diag(D);
[~, idx] = sort(abs(abs(lam)-1),'ascend');

mu = NaN; q = []; p = []; isComplex = false;
for j = idx.'
    if abs(imag(lam(j))) > imagtol
        mu = lam(j);
        q  = V(:,j);
        % left eigenvector from transpose
        [VLt, DLt] = eig(DP.');
        lamLt = diag(DLt);
        [~, jL] = min(abs(lamLt - mu));    % find closest
        p = conj(VLt(:,jL));         % left eigenvector (conjugate of right of A^T)
        % normalise: p^T q = 1
        s = p'*q;  p = p / s;  q = q;
        isComplex = true;
        if imag(mu) < 0, mu = conj(mu); q = conj(q); p = conj(p); end
        return
    end
end

% Fall back: no complex multiplier found — pick the one closest to unit circle
j = idx(1);
mu = lam(j); q = V(:,j);
[VLt, DLt] = eig(DP.');
lamLt = diag(DLt);
[~, jL] = min(abs(lamLt - mu));
p = conj(VLt(:,jL));  s = p'*q; p = p / s;
isComplex = false;
end

function ok = resonance_check(theta, K)
% Check e^{ik theta} ~= 1 for k=1..K (within tight tolerance)
tol = 1e-6;
ok = true(1,K);
for k=1:K
    ok(k) = abs(exp(1i*k*theta) - 1) > tol;
end
end

% ================= Multilinear forms & resolvents =================

function l1 = first_Lyapunov_coeff(X, gamma, par, dt, A, mu, q, p)
% Compute first Lyapunov coefficient for NS of a 3D map Phi using:
% l1 = 0.5 * Re[ conj(mu) p^T C(q,q,conj(q))
%                 - 2 conj(mu) p^T B(q, (I - conj(mu)A)^{-1}_Q B(q,conj(q)))
%                 + conj(mu) p^T B(conj(q), (2I - conj(mu)A)^{-1}_Q B(q,q)) ]

% Build function handles for B and C at (X,gamma)
B = @(u,v) B_map(X, gamma, par, dt, u, v);
C = @(u,v,w) C_map(X, gamma, par, dt, u, v, w);

% Projector onto complement of span{q} along p: Q = I - q p^T
Q = eye(3) - q * (p.');

% Projected resolvents via bordered linear solves
M1 = eye(3) - conj(mu) * A;
M2 = 2*eye(3) - conj(mu) * A;

R1 = @(b) solve_on_Q(M1, q, p, Q*b);  % (I - conj(mu)A)^{-1} on Q
R2 = @(b) solve_on_Q(M2, q, p, Q*b);  % (2I - conj(mu)A)^{-1} on Q

term1 = conj(mu) * (p.' * C(q, q, conj(q)));
term2 = conj(mu) * (p.' * B(q, R1(B(q, conj(q)))));
term3 = conj(mu) * (p.' * B(conj(q), R2(B(q, q))));

l1 = 0.5 * real( term1 - 2*term2 + term3 );
end

function x = solve_on_Q(M, q, p, b)
% Solve on Q-subspace: find x s.t. M x = b, p^T x = 0.
% Use bordered system: [ M  q ][ x ] = [ b ]
%                      [ p^T 0 ][ s ]   [ 0 ]
K = [M, q; p.', 0];
rhs = [b; 0];
sol = K \ rhs;
x = sol(1:3);
end

function Buv = B_map(X, gamma, par, dt, u, v)
% Symmetric bilinear: B(u,v) = D^2 Phi[X][u,v] = dt * D^2 F[X][u,v]
% Numerically: D^2F[u,v] = d/dt ( J_f(X + t v) u )|_{t=0} (central diff)
eps0 = 1e-6;
epsv = eps0 * max(1, norm(v));
Jp = J_f(X + epsv*v, gamma, par);
Jm = J_f(X - epsv*v, gamma, par);
D2F_uv = (Jp*u - Jm*u) / (2*epsv);
% symmetrise with swapped arguments
Jp2 = J_f(X + epsv*u, gamma, par);
Jm2 = J_f(X - epsv*u, gamma, par);
D2F_vu = (Jp2*v - Jm2*v) / (2*epsv);
Buv = 0.5 * dt * (D2F_uv + D2F_vu);
end

function Cuvw = C_map(X, gamma, par, dt, u, v, w)
% Trilinear: C(u,v,w) = dt * D^3F[X][u,v,w]
% Use symmetric averaging of directional differences of B:
eps0 = 1e-6;
epsw = eps0 * max(1, norm(w));
epsu = eps0 * max(1, norm(u));
epsv = eps0 * max(1, norm(v));

% C_w(u,v) = d/dt B_{X + t w}(u,v)|_{t=0}
Bw_p = B_atX(X + epsw*w, gamma, par, dt, u, v);
Bw_m = B_atX(X - epsw*w, gamma, par, dt, u, v);
Cw = (Bw_p - Bw_m) / (2*epsw);

% C_u(v,w)
Bu_p = B_atX(X + epsu*u, gamma, par, dt, v, w);
Bu_m = B_atX(X - epsu*u, gamma, par, dt, v, w);
Cu = (Bu_p - Bu_m) / (2*epsu);

% C_v(w,u)
Bv_p = B_atX(X + epsv*v, gamma, par, dt, w, u);
Bv_m = B_atX(X - epsv*v, gamma, par, dt, w, u);
Cv = (Bv_p - Bv_m) / (2*epsv);

Cuvw = (Cw + Cu + Cv) / 3;
end

function Buv = B_atX(X, gamma, par, dt, u, v)
% B at a shifted base point X: same numerical B-map but evaluated at X.
eps0 = 1e-6;
epsv = eps0 * max(1, norm(v));
Jp = J_f(X + epsv*v, gamma, par);
Jm = J_f(X - epsv*v, gamma, par);
D2F_uv = (Jp*u - Jm*u) / (2*epsv);

epsu = eps0 * max(1, norm(u));
Jp2 = J_f(X + epsu*u, gamma, par);
Jm2 = J_f(X - epsu*u, gamma, par);
D2F_vu = (Jp2*v - Jm2*v) / (2*epsu);

Buv = 0.5 * dt * (D2F_uv + D2F_vu);
end

% ================= Fixed-point residuals/Jacobians =================

function [X, ok] = newton_fp(X0, gamma, par, dt, newton)
% Solve G(X,gamma) = Phi(X,gamma) - X = 0  (3x3) via Newton
X = X0;
ok = false;
for it=1:newton.maxit
    [G, JG] = G_and_JG(X, gamma, par, dt);
    nrm = norm(G,2);
    if nrm < newton.tol, ok = true; return; end
    DX = -JG \ G;
    X  = X + DX;
end
ok = (norm(G_and_JG(X,gamma,par,dt)) < 10*newton.tol);
end

function [G, JG] = G_and_JG(X, gamma, par, dt)
% G(X,g) = Phi(X,g) - X = dt * f(X,g)
% JG     = dG/dX       = dt * J_f(X,g)
[f1, f2, f3] = f_rhs(X, gamma, par);
G  = dt * [f1; f2; f3];
Jf = J_f(X, gamma, par);
JG = dt * Jf;
end

function A = DPhi(X, gamma, par, dt)
Jf = J_f(X, gamma, par);
A  = eye(3) + dt * Jf;
end

% ================= Model f and J_f (analytic) =================

function [f1, f2, f3] = f_rhs(X, gamma, par)
x = X(1); y = X(2); z = X(3);
alpha = par.alpha; beta = par.beta; delta = par.delta; epsilon = par.epsilon;

z = max(z, 1e-12);  % guard

f1 = x*(1 - x/z) - x*y;
f2 = alpha*y*(1 - beta*y/z) + x*y;
f3 = z*(gamma - delta*x - epsilon*y);
end

function J = J_f(X, gamma, par)
x = X(1); y = X(2); z = X(3);
alpha = par.alpha; beta = par.beta; delta = par.delta; epsilon = par.epsilon;

z = max(z, 1e-12);

df1dx = (1 - 2*x/z) - y;
df1dy = -x;
df1dz = (x^2)/(z^2);

df2dx = y;
df2dy = alpha*(1 - 2*beta*y/z) + x;
df2dz = alpha*beta*(y^2)/(z^2);

df3dx = -delta*z;
df3dy = -epsilon*z;
df3dz = gamma - delta*x - epsilon*y;

J = [ df1dx, df1dy, df1dz;
    df2dx, df2dy, df2dz;
    df3dx, df3dy, df3dz ];
end

% ================= Utilities =================
function s = ternary(cond, a, b), if cond, s=a; else, s=b; end, end