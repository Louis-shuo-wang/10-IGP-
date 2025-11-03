function fourier_circle_continuation()
% FOURIER_CIRCLE_CONTINUATION
% Fourier-collocation continuation of invariant circles for the Euler map
% Phi(X;gamma) = X + dt * f(X;gamma) of the nondimensional Leslie–Gower IGP.
%
% Unknowns: Fourier coeffs of X(θ) in R^3, rotation number rho, and parameter gamma.
% Invariance: X(θ + 2π rho) = Phi(X(θ); gamma) at collocation nodes.
% Phase fixing: first-harmonic sine coefficient of x-component set to 0 (b1x=0).
% Continuation: pseudo-arclength constraint.
%
% --- You can integrate this with your NS-refine script: use (XNS,gammaNS,mu,q)
%     and set rho0 = angle(mu)/(2π), then call solve/corrector below.

%% Model / numerics parameters
par.alpha   = 2.0;
par.beta    = 1.0;
par.delta   = 0.5;
par.epsilon = 1;

dt       = 0.10;         % Euler step
K        = 6;       % # Fourier harmonics per coordinate
M        = 2*K + 1; % # collocation nodes (M must equal 2K+1)
amp0     = 2e-3;    % initial torus amplitude (small)
ds       = 2.5e-3;  % arclength step
Nsteps   = 120;     % continuation steps
fd_eps   = 1e-6;    % finite-diff step for Jacobian

% Newton tolerances
newton.maxit   = 30;
newton.tol     = 1e-11;
newton.verbose = false;

%% 0) Find a coexistence fixed point near an NS candidate and seed the torus
% Choose a gamma near where your continuation flagged an NS
gamma0 = 0.2;       % <- pick from your detection phase
X0guess = [1; 1; 1];

[Xstar, ok] = newton_fp(X0guess, gamma0, par, dt, newton);
assert(ok, 'Could not find fixed point at gamma0. Try another guess/parameter.');

% Linear data at the fixed point
A0 = DPhi(Xstar, gamma0, par, dt);
[V,D] = eig(A0);
lam = diag(D);

% Pick the complex pair closest to unit circle (Im>0)
[~,ix] = min(abs(abs(lam)-1)); mu = lam(ix);
if abs(imag(mu)) < 1e-9
    error('No complex multiplier near unit circle at gamma0; choose another gamma.');
end
if imag(mu) < 0   % enforce Im(mu)>0
    mu = conj(mu);
    v = conj(V(:,ix));
else
    v = V(:,ix);
end
rho0 = angle(mu)/(2*pi);  % rotation number seed in (0,1)

% Seed Fourier coefficients: mean at fixed point, first harmonic from eigenvector
C0 = zeros(3, 2*K+1);  % columns: k=0,1c,1s,2c,2s,...,Kc,Ks (per coordinate)
C0(:,1) = Xstar;       % mean term a0
% place first harmonic: a1 = Re(v)*amp0, b1 = -Im(v)*amp0
C0(:,2) = real(v) * amp0;       % cos term k=1
C0(:,3) = -imag(v) * amp0;      % sin term k=1
rho = rho0;

% Pack unknown vector U = [coeffs(:); rho; gamma]
U0 = [C0(:); rho; gamma0];

%% 1) Solve collocation (invariance + phase) at gamma0 (no arclength yet)
[U0, ok] = torus_newton(U0, K, M, par, dt, newton, fd_eps);
assert(ok, 'Initial collocation solve failed; tweak amp0/K/M/gamma0.');

% Take a small natural-parameter step to form secant (change gamma a bit)
U1 = U0; U1(end) = U0(end) + 1e-3;
[U1, ok] = torus_newton(U1, K, M, par, dt, newton, fd_eps);
assert(ok, 'Second collocation solve failed; reduce step.');

% Initial secant tangent for pseudo-arclength
tsec = (U1 - U0); tsec = tsec / norm(tsec);

% Storage
branch = struct();
branch.U   = nan(numel(U0), Nsteps+2);
branch.normR  = nan(1, Nsteps+2);
branch.gamma  = nan(1, Nsteps+2);
branch.rho    = nan(1, Nsteps+2);

branch.U(:,1) = U0; branch.gamma(1)=U0(end); branch.rho(1)=U0(end-1);
branch.U(:,2) = U1; branch.gamma(2)=U1(end); branch.rho(2)=U1(end-1);

branch.normR(1) = torus_residual_norm(U0, K, M, par, dt);
branch.normR(2) = torus_residual_norm(U1, K, M, par, dt);

U_prevprev = U0; U_prev = U1;

fprintf('Starting Fourier-collocation torus continuation: K=%d, M=%d, ds=%.3g\n', K, M, ds);

%% 2) Pseudo-arclength continuation
for k = 3:Nsteps+2
    Upred = U_prev + ds * tsec;
    [Ucorr, ok] = torus_newton(Upred, K, M, par, dt, newton, fd_eps, tsec, Upred);
    if ~ok
        % try a half-step
        Upred = U_prev + 0.5 * ds * tsec;
        [Ucorr, ok] = torus_newton(Upred, K, M, par, dt, newton, fd_eps, tsec, Upred);
        if ~ok
            fprintf(' Corrector failed at step %d; stopping.\n', k);
            branch.U      = branch.U(:,1:k-1);
            branch.normR  = branch.normR(1:k-1);
            branch.gamma  = branch.gamma(1:k-1);
            branch.rho    = branch.rho(1:k-1);
            break
        end
    end

    % Store
    branch.U(:,k) = Ucorr;
    branch.gamma(k) = Ucorr(end);
    branch.rho(k)   = Ucorr(end-1);
    branch.normR(k) = torus_residual_norm(Ucorr, K, M, par, dt);

    % Update tangent
    U_prevprev = U_prev; U_prev = Ucorr;
    tsec = (U_prev - U_prevprev); tsec = tsec / norm(tsec);

    fprintf(' step %3d | gamma=%.6g | rho=%.6g | ||R||=%.2e\n', ...
        k-2, branch.gamma(k), branch.rho(k), branch.normR(k));
end

%% 3) Quick plots: amplitude vs gamma, rotation number vs gamma
gam = branch.gamma; rho = branch.rho;

% Extract first-harmonic amplitude of x-component
[a0,ac,as] = coeffs_unpack(branch.U(:,end), K); %#ok<ASGLU>
amp_x1 = zeros(1, size(branch.U,2));
for j=1:size(branch.U,2)
    [~,acj,asj] = coeffs_unpack(branch.U(:,j), K);
    amp_x1(j) = norm([acj(1,1); asj(1,1)],2);
end

figure; subplot(1,2,1);
plot(gam, amp_x1, 'k.-','LineWidth',1.25); grid on
xlabel('\gamma'); ylabel('||first harmonic (x)||');
title('Invariant circle amplitude');

subplot(1,2,2);
plot(gam, rho, 'b.-','LineWidth',1.25); grid on
xlabel('\gamma'); ylabel('\rho');
title('Rotation number along branch');

% Optional: visualize the circle embedding at the last step
[theta_grid, Xgrid] = torus_sample(branch.U(:,end), K, 400);
figure; plot3(Xgrid(1,:), Xgrid(2,:), Xgrid(3,:), 'k-'); grid on
xlabel('x'); ylabel('y'); zlabel('z'); title('Invariant circle (last solution)');

end
%% ===================== Core collocation solvers =====================

function [U, ok] = torus_newton(Uinit, K, M, par, dt, newton, fd_eps, tsec, Upred)
% Solve H(U)=0 with Newton:
% H = [ R_collocation(U);  phase(U) ; (tsec^T)(U-Upred) ]  (last term optional)
U = Uinit(:);
useArc = (nargin >= 9);

ok = false;
for it = 1:newton.maxit
    [H, ~] = torus_residual(U, K, M, par, dt);
    if useArc
        Harc = tsec.' * (U - Upred);
        H = [H; Harc];
    end
    nrm = norm(H,2);
    if nrm < newton.tol, ok = true; return; end

    J = torus_jacobian_fd(U, K, M, par, dt, fd_eps, useArc, tsec); % finite-diff Jacobian
    dU = - J \ H;
    U  = U + dU;
end
% final check
[H, ~] = torus_residual(U, K, M, par, dt);
if useArc, H = [H; tsec.'*(U-Upred)]; end
ok = (norm(H) < 10*newton.tol);
end

function [H, parts] = torus_residual(U, K, M, par, dt)
% Build residual H = [R_collocation; phase_condition]
% Unknown vector U = [C(:); rho; gamma], C is 3 x (2K+1)
[C, rho, gamma] = coeffs_unpack(U, K);

% Collocation nodes
theta = 2*pi*(0:M-1)/M;
omega = 2*pi*rho;

% Evaluate embedding X(theta) and shifted X(theta+omega)
Xth   = eval_trig(C, theta);
Xsh   = eval_trig(C, wrap_angles(theta + omega));
% Map at X(theta)
PhiX  = Phi_batch(Xth, gamma, par, dt);

R = (Xsh - PhiX);      % 3-by-M residual at nodes
Rflat = R(:);

% Phase fix: set b1 of x-component to zero (b1x = 0)
b1x = C(1,3);         % column 3 is sin(1*theta) for our packing
H = [Rflat; b1x];

if nargout > 1
    parts.R = R; parts.Xth = Xth; parts.Xsh = Xsh; parts.PhiX = PhiX;
end
end

function J = torus_jacobian_fd(U, K, M, par, dt, eps, useArc, tsec)
% Finite-difference Jacobian of H(U).
% Size: H has 3M + 1 (+1 if arc) rows; U has nC + 2 cols where nC=3*(2K+1)
[C, rho, gamma] = coeffs_unpack(U, K); %#ok<ASGLU>
nC = numel(C);
mH = 3*M + 1 + (useArc~=0);
J  = zeros(mH, nC + 2);

H0 = torus_residual(U, K, M, par, dt);
H0 = H0(:);

for j = 1:(nC + 2)
    dU = zeros(size(U));
    dU(j) = eps;
    H1 = torus_residual(U + dU, K, M, par, dt);
    H1 = H1(:);
    J(:,j) = (H1 - H0)/eps;
end

if useArc
    % append arc row (tsec^T)
    J(end, :) = tsec.';
end
end

function nrm = torus_residual_norm(U, K, M, par, dt)
H = torus_residual(U, K, M, par, dt); H = H(:);
nrm = norm(H(1:end-1));  % ignore phase row
end

%% ===================== Fourier helpers =====================

function X = eval_trig(C, theta)
% Evaluate X(theta) from coefficients C (3 x (2K+1)) at vector theta (1 x M)
% Packing: [a0, a1c, a1s, a2c, a2s, ..., aKc, aKs]
[~, L] = size(C); K = (L-1)/2;
M = numel(theta);
X = zeros(3, M);
% a0 term
X = X + C(:,1) * ones(1,M);
% harmonics
for k=1:K
    ck = C(:, 2*k    ); % cos coeffs
    sk = C(:, 2*k + 1); % sin coeffs
    X = X + ck * cos(k*theta) + sk * sin(k*theta);
end
end

function [theta] = wrap_angles(theta)
% Wrap angles into [0, 2π) (not strictly needed for trig, but keeps numbers tame)
theta = mod(theta, 2*pi);
end

function [C, rho, gamma] = coeffs_unpack(U, K)
% U = [C(:); rho; gamma], C is 3 x (2K+1)
nC = 3*(2*K+1);
C = reshape(U(1:nC), 3, 2*K+1);
rho   = U(nC + 1);
gamma = U(nC + 2);
end

function [C, rho, gamma] = coeffs_unpack_only(U, K) %#ok<DEFNU>
[C, rho, gamma] = coeffs_unpack(U, K);
end

%% ===================== Model & vectorized map =====================

function Y = Phi_batch(X, gamma, par, dt)
% X is 3xM, returns Phi(X(:,j)) for each column
M = size(X,2);
Y = X + dt * f_rhs_batch(X, gamma, par, M);
end

function FX = f_rhs_batch(X, gamma, par, M)
x = X(1,:); y = X(2,:); z = X(3,:); z = max(z, 1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
f1 = x.*(1 - x./z) - x.*y;
f2 = a*y.*(1 - b*y./z) + x.*y;
f3 = z.*(gamma - d*x - e*y);
FX = [f1; f2; f3];
end

function Y = Phi(X, gamma, par, dt)
[f1,f2,f3] = f_rhs(X, gamma, par);
Y = X + dt*[f1; f2; f3];
end

function [f1, f2, f3] = f_rhs(X, gamma, par)
x=X(1); y=X(2); z=X(3); z = max(z,1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
f1 = x*(1 - x/z) - x*y;
f2 = a*y*(1 - b*y/z) + x*y;
f3 = z*(gamma - d*x - e*y);
end

function A = DPhi(X, gamma, par, dt)
A = eye(3) + dt*J_f(X, gamma, par);
end

function J = J_f(X, gamma, par)
x=X(1); y=X(2); z=X(3); z=max(z,1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
J = [ (1-2*x/z)-y , -x , (x^2)/(z^2) ;
    y  , a*(1-2*b*y/z)+x , a*b*(y^2)/(z^2) ;
    -d*z   , -e*z  , gamma-d*x-e*y   ];
end

%% ===================== Fixed point Newton (for seeding) =====================

function [X, ok] = newton_fp(X0, gamma, par, dt, newton)
X = X0; ok = false;
for it=1:newton.maxit
    [G, JG] = G_and_JG(X, gamma, par, dt);
    if norm(G,2) < newton.tol, ok=true; return; end
    X = X - JG\G;
end
ok = (norm(G_and_JG(X,gamma,par,dt)) < 10*newton.tol);
end

function [G, JG] = G_and_JG(X, gamma, par, dt)
[f1,f2,f3] = f_rhs(X, gamma, par);
G  = dt*[f1; f2; f3];
JG = dt*J_f(X, gamma, par);
end