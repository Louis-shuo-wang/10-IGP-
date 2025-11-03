function cont_IGP_discrete()
% CONT_IGP_DISCRETE
%   Pseudo-arclength continuation of fixed points for the discrete
%   (forward-Euler) Leslie–Gower intraguild predation map.
%   Detects SN (mu~+1), Flip (mu~-1), and NS (|mu|~1 with complex pair).
%
% Model (nondimensional continuous-time):
%   f1 = x*(1 - x/z) - x*y
%   f2 = alpha*y*(1 - beta*y/z) + x*y
%   f3 = z*(gamma - delta*x - epsilon*y)
% Discrete map (Euler): Phi(X; p) = X + dt * f(X; p)
%
% Author: (your name)
% -------------------------------------------------------------------------

%% Parameters (you can tune these)
par.alpha  = 1.0;
par.beta = 2.0;
par.delta  = 0.5;
par.epsilon = 0.2;

dt = 0.1;        % time step of the map  (smaller = closer to ODE)
gamma0 = 0.24;       % starting parameter (enrichment)
ds = 2.5e-3;     % arclength stepsize for continuation
Nsteps= 600;        % number of continuation steps

% Newton settings
newton.maxit   = 20;
newton.tol     = 1e-10;
newton.verbose = false;

% Bifurcation detection tolerances
tol_mu1  = 2e-3;  % distance to +1 or -1
tol_ns   = 2e-3;  % distance of |mu|-1
tol_imag = 1e-6;  % to consider a pair "non-real"

%% Get first two solutions: natural-parameter Newton solves for G(X,gamma)=0
% We solve the fixed point condition G(X,gamma) = Phi(X,gamma) - X = dt*f(X,gamma) = 0
fprintf('Solving for initial fixed points at gamma0 and gamma1...\n');

X0_guess = [0.35; 0.35; 0.5]; % reasonable positive guess (tune as needed)

[X0, ok0] = newton_fp(X0_guess, gamma0, par, dt, newton);
assert(ok0, 'Initial Newton solve failed at gamma0');

gamma1 = gamma0 + 1.0e-3;     % small step in parameter to seed secant tangent
[X1, ok1] = newton_fp(X0, gamma1, par, dt, newton);
if ~ok1
    % try a gentle perturbation if needed
    [X1, ok1] = newton_fp(X0.*1.01, gamma1, par, dt, newton);
end
assert(ok1, 'Second Newton solve failed at gamma1');

U0 = [X0; gamma0];
U1 = [X1; gamma1];

% Storage for results
storeU   = nan(4, Nsteps+2);
storeEig = cell(1, Nsteps+2);
labels   = strings(1, Nsteps+2);

storeU(:,1) = U0;
storeU(:,2) = U1;

labels(1) = classify_stability(X0, gamma0, par, dt);
labels(2) = classify_stability(X1, gamma1, par, dt);
storeEig{1} = eig_DPhi(X0, gamma0, par, dt);
storeEig{2} = eig_DPhi(X1, gamma1, par, dt);

% Initial secant tangent, normalized:
tsec = (U1 - U0);
tsec = tsec / norm(tsec);

fprintf('Starting pseudo-arclength continuation with ds = %.3g, Nsteps = %d\n', ds, Nsteps);

%% Continuation loop
U_prevprev = U0;
U_prev     = U1;
for k = 3:Nsteps+2
    % Predictor
    Upred = U_prev + ds * tsec;

    % Corrector (augmented Newton on H(U) = [G(X,g)=0;  tsec^T (U - Upred) = 0])
    [Ucorr, ok] = newton_corrector(Upred, tsec, par, dt, newton);
    if ~ok
        warning('Corrector failed at step %d; trying half step...', k);
        % try a shorter step
        Upred2 = U_prev + 0.5*ds * tsec;
        [Ucorr, ok] = newton_corrector(Upred2, tsec, par, dt, newton);
        if ~ok
            warning('Corrector failed again; stopping.');
            storeU = storeU(:,1:k-1);
            labels = labels(1:k-1);
            storeEig = storeEig(1:k-1);
            break
        end
    end

    storeU(:,k) = Ucorr;
    Xk   = Ucorr(1:3);
    gammk = Ucorr(4);

    % Multipliers and classification
    lam  = eig_DPhi(Xk, gammk, par, dt);
    storeEig{k} = lam;
    labels(k) = classify_stability(Xk, gammk, par, dt);

    % Report bifurcation hits (approximate)
    detect_and_report_bif(k, lam, tol_mu1, tol_ns, tol_imag, gammk);

    % Update tangent (secant) for the next step:
    U_prevprev = U_prev;
    U_prev = Ucorr;
    tsec = (U_prev - U_prevprev);
    tsec = tsec / norm(tsec);
end

%% Simple plots
gammas = storeU(4,:);
xs     = storeU(1,:);
specR  = cellfun(@(v) max(abs(v)), storeEig);

figure; subplot(1,2,1);
plot(gammas, xs, 'k.-','LineWidth',1.25); grid on
xlabel('\gamma'); ylabel('x^*');
title('Fixed-point branch (x^* vs \gamma)');

subplot(1,2,2);
plot(gammas, specR, 'b.-','LineWidth',1.25); hold on; yline(1,'r--');
grid on; xlabel('\gamma'); ylabel('max |multiplier|');
title('Stability along branch (unit circle in red)');

fprintf('\n=== Done. ===\n');
end

% ===================== Helpers below =====================

function [X, ok] = newton_fp(X0, gamma, par, dt, newton)
% Solve G(X,gamma) = Phi(X,gamma) - X = dt * f(X,gamma) = 0 (3x3) by Newton
X = X0;
ok = false;
for it=1:newton.maxit
    [G, JG] = G_and_JG(X, gamma, par, dt); % G = dt f, JG = dt * J_f
    nrm = norm(G,2);
    if nrm < newton.tol, ok = true; return; end
    DX = -JG \ G;
    X  = X + DX;
    if newton.verbose
        fprintf('  Newton it=%d, ||G||=%.3e\n', it, nrm);
    end
end
% final check
ok = (norm(G_and_JG(X,gamma,par,dt)) < 10*newton.tol);
end

function [U, ok] = newton_corrector(Upred, tsec, par, dt, newton)
% Corrector for pseudo-arclength: solve H(U) = [G(X,g)=0 ; tsec^T (U-Upred)=0]
U = Upred;
ok = false;
for it=1:newton.maxit
    X = U(1:3); gamma = U(4);

    % Residuals
    [G, JG] = G_and_JG(X, gamma, par, dt);  % (3x1), (3x3)
    % G_gamma = dt * df/dgamma = dt * [0; 0; z]
    Gg = dt * [0; 0; X(3)];   % (3x1)
    % Arc-length equation
    H4 = tsec.'*(U - Upred);   % scalar

    % Assemble augmented residual & Jacobian
    H = [G; H4];
    JH = [JG, Gg; tsec.'];

    nrmH = norm(H,2);
    if nrmH < newton.tol, ok = true; return; end

    DU = - JH \ H;
    U  = U + DU;
end
ok = (norm(H) < 10*newton.tol);
end

function [G, JG] = G_and_JG(X, gamma, par, dt)
% G(X,g) = Phi(X,g) - X = dt f(X,g)
% JG     = dG/dX     = dt J_f(X,g)
[f1, f2, f3] = f_rhs(X, gamma, par);
G  = dt * [f1; f2; f3];
Jf = J_f(X, gamma, par);
JG = dt * Jf;
end

function lam = eig_DPhi(X, gamma, par, dt)
% Multipliers (eigenvalues of D Phi) at the fixed point (X,gamma)
Jf   = J_f(X, gamma, par);
DPhi = eye(3) + dt*Jf;
lam  = eig(DPhi);
end

function label = classify_stability(X, gamma, par, dt)
lam = eig_DPhi(X, gamma, par, dt);
rho = max(abs(lam));
if rho < 1
    label = "stable FP";
else
    label = "unstable FP";
end
end

function detect_and_report_bif(k, lam, tol_mu1, tol_ns, tol_imag, gamma)
% Rough/instantaneous detectors (use bracket + refinement for publication-quality)
d_plus = min(abs(lam - 1));
d_minus= min(abs(lam + 1));
mods   = abs(lam);

% NS: must have a complex pair close to unit circle (exclude +1/-1)
isComplex = any(abs(imag(lam)) > tol_imag);
d_unit    = min(abs(mods - 1));

hitSN = d_plus  < tol_mu1;
hitPD = d_minus < tol_mu1;
hitNS = isComplex && (d_unit < tol_ns);

if hitSN
    fprintf('  [Step %4d] Possible SN (mu ~ +1) at gamma ~ %.6g\n', k, gamma);
end
if hitPD
    fprintf('  [Step %4d] Possible Flip (mu ~ -1) at gamma ~ %.6g\n', k, gamma);
end
if hitNS
    fprintf('  [Step %4d] Possible NS (|mu| ~ 1, complex pair) at gamma ~ %.6g\n', k, gamma);
end
end

% ---------------- Model-specific functions ----------------

function [f1, f2, f3] = f_rhs(X, gamma, par)
% Right-hand side of the nondimensional ODE (used inside the map)
x = X(1); y = X(2); z = X(3);
alpha = par.alpha; beta = par.beta; delta = par.delta; epsilon = par.epsilon;

% Guard (z>0 is required by the model)
if z <= 0
    % small regularization to avoid numerical issues (you may handle differently)
    z = max(z, 1e-12);
end

f1 = x*(1 - x/z) - x*y;
f2 = alpha*y*(1 - beta*y/z) + x*y;
f3 = z*(gamma - delta*x - epsilon*y);
end

function J = J_f(X, gamma, par)
% Analytic Jacobian of f(x,y,z)
x = X(1); y = X(2); z = X(3);
alpha = par.alpha; beta = par.beta; delta = par.delta; epsilon = par.epsilon;

% Guard
if z <= 0
    z = max(z, 1e-12);
end

% df1/d*
df1dx = (1 - 2*x/z) - y;
df1dy = -x;
df1dz = (x^2)/(z^2);

% df2/d*
df2dx = y;
df2dy = alpha*(1 - 2*beta*y/z) + x;
df2dz = alpha*beta*(y^2)/(z^2);

% df3/d*
df3dx = -delta*z;
df3dy = -epsilon*z;
df3dz = gamma - delta*x - epsilon*y;

J = [ df1dx, df1dy, df1dz;
    df2dx, df2dy, df2dz;
    df3dx, df3dy, df3dz ];
end