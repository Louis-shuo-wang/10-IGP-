% plot_NS_by_gamma.m
% Demonstrate Neimark–Sacker (discrete Hopf) bifurcation for the forward-Euler map.
% Uses your NS parameter set:
%   alpha=0.2646, beta=1.453, gamma_crit≈1.151, delta=0.239, epsilon=1.939, dt=0.6346

clear; close all; clc;

% --- Parameters (NS case) ---
alpha=1.011;
beta=1.635;
gamma0=3.174;
delta=3.673;
epsilon=3.708;
dt_NS=0.4017;

% alpha   = 0.2646;
% beta    = 1.453;
% gamma0  = 1.155;    % reported critical gamma (NS)
% delta   = 0.239;
% epsilon = 1.939;
% dt_NS   = 0.6346;   % Δτ that produced the NS in your search

%% --- Choose gammas below and above the critical value ---
dg = 0.03;                       % small offset around gamma0; tweak if needed
gamma_below = max(1e-6, gamma0 - dg);
gamma_above = gamma0 + dg;

fprintf('Using gamma_below = %.6g, gamma_crit≈%.6g, gamma_above = %.6g\n', gamma_below, gamma0, gamma_above);

%% --- helper: equilibrium and Jacobian (same formulas as in your script) ---
function [Rstar, Nstar_val, Pstar_val, J] = computeEquilJ(alpha,beta,gamma,delta,epsilon)
    % coefficients used in your provided snippet
    c1 = gamma + alpha*delta - epsilon;
    c2 = -alpha*(beta*delta + epsilon);
    c3 = alpha*beta*gamma;
    rts = roots([c1, c2, c3]);
    rts = rts(abs(imag(rts))<1e-10);   % real roots only
    rts = real(rts);
    rts = rts(rts>0 & rts < beta);     % interior positive root(s) < beta
    if isempty(rts)
        Rstar = NaN; Nstar_val = NaN; Pstar_val = NaN; J = NaN(3);
        return;
    end
    % pick the largest positive root (same convention used earlier)
    Rstar = max(rts);
    Nstar_val = alpha*Rstar*(beta - Rstar)./(Rstar.^2 + alpha*beta);
    Pstar_val = Rstar.*(Rstar + alpha)./(Rstar.^2 + alpha*beta);
    % Jacobian entries (same as your script)
    J11 = 1 - 2*Nstar_val/Rstar - Pstar_val;
    J12 = -Nstar_val;
    J13 = (Nstar_val^2)/(Rstar^2);
    J21 = Pstar_val;
    J22 = alpha*(1 - 2*beta*Pstar_val/Rstar) + Nstar_val;
    J23 = alpha*beta*Pstar_val^2/(Rstar^2);
    J31 = -delta*Rstar;
    J32 = -epsilon*Rstar;
    J33 = gamma - delta*Nstar_val - epsilon*Pstar_val;
    J = [J11 J12 J13; J21 J22 J23; J31 J32 J33];
end

%% --- simulate forward-Euler map (iterate) ---
function traj = iterate_euler(dt, steps, X0, alpha, beta, delta, epsilon, gamma)
    if nargin < 3 || isempty(X0)
        % fallback X0 (will be set by caller)
        X0 = [0.5; 0.5; 0.5];
    end
    traj = zeros(steps,3);
    X = X0;
    for n = 1:steps
        X = max(X, 1e-12); % protect divisions
        % ODE right-hand side (vectorized)
        N = X(1); P = X(2); R = X(3);
        dN = N.*(1 - N./R) - N.*P;
        dP = alpha*P.*(1 - beta*P./R) + N.*P;
        dR = R.*(gamma - delta*N - epsilon*P);
        X = X + dt*[dN; dP; dR];
        traj(n,:) = X';
    end
end

%% --- compute equilibrium & J for below and above ---
[R_b, N_b, P_b, J_b] = computeEquilJ(alpha,beta,gamma_below,delta,epsilon);
[R_a, N_a, P_a, J_a] = computeEquilJ(alpha,beta,gamma_above,delta,epsilon);

if any(isnan([R_b R_a]))
    warning('Could not find a valid interior equilibrium for one of the gamma values. Consider widening search or using different offset dg.');
end

fprintf('\nEquilibrium (below): R=%.6g, N=%.6g, P=%.6g\n', R_b, N_b, P_b);
fprintf('Equilibrium (above): R=%.6g, N=%.6g, P=%.6g\n\n', R_a, N_a, P_a);

%% --- initial conditions: small perturbations around equilibrium ---
X0_b = [N_b*0.98; P_b*1.02; R_b*1.01];
X0_a = [N_a*0.98; P_a*1.02; R_a*1.01];

steps = 1500;    % total iterations (transient + sample)
transient = 1000; % drop first 'transient' iterates when plotting

traj_below = iterate_euler(dt_NS, steps, X0_b, alpha, beta, delta, epsilon, gamma_below);
traj_above = iterate_euler(dt_NS, steps, X0_a, alpha, beta, delta, epsilon, gamma_above);

%% --- compute multipliers along a small gamma-sweep to visualize crossing ---
ng = 301;
gammas = linspace(gamma0 - 5*dg, gamma0 + 5*dg, ng);
minDistToMinus1 = nan(size(gammas));
specRad = nan(size(gammas));
closestMu = complex(nan(size(gammas)));
for k = 1:ng
    gk = gammas(k);
    [Rk, Nk, Pk, Jk] = computeEquilJ(alpha,beta,gk,delta,epsilon);
    if any(isnan([Rk Nk Pk]))
        minDistToMinus1(k) = NaN;
        specRad(k) = NaN;
        continue;
    end
    eigJ = eig(Jk);
    mu = 1 + dt_NS * eigJ;
    % distance of the multiplier closest to -1:
    dists = abs(mu + 1);
    [minDistToMinus1(k), idxmin] = min(dists);
    closestMu(k) = mu(idxmin);
    specRad(k) = max(abs(mu));
end

%% --- PLOT: time series (below/above) and multiplier crossing ---
defaultsetting;
figure('Position',[100,100,900,1200]);
tiledlayout(3,2,"TileSpacing","tight","Padding","tight");

t = (1:steps);

% (a) P_n below
nexttile;
plot(t(transient+1:end), traj_below(transient+1:end,2));
xlabel('n'); ylabel('$P_n$'); 
title(sprintf('(a) $\\gamma = %.5g$', gamma_below));
grid on;

% (b) P_n above
nexttile;
plot(t(transient+1:end), traj_above(transient+1:end,2));
xlabel('n'); ylabel('$P_n$'); 
title(sprintf('(b) $\\gamma = %.5g$', gamma_above));
grid on;

% (c) multiplier distance to -1 (shows crossing near zero)
nexttile(3,[2,2]);
plot(gammas, specRad, ':','DisplayName','$\rho$');
yline(1, 'k--', 'DisplayName','ref.');
xline(gamma0, 'b--', 'DisplayName','$\gamma_{\mathrm{crit}}$');
xlabel('$\gamma$'); 
ylabel('spectral radius $\rho(D\Phi)$');
legend('Location','best');
title('(c)');

% save figure if desired
% exportgraphics(gcf,'flip_by_gamma.svg','Resolution',600);

%% --- quick diagnostics printed to command window ---
mu_b = eig(eye(3) + dt_NS * J_b);
mu_a = eig(eye(3) + dt_NS * J_a);
fprintf('Multipliers (below gamma):\n'); disp(mu_b.');
fprintf('Multipliers (above gamma):\n'); disp(mu_a.');

% also print min |mu+1| at gamma_below/gamma_above
fprintf('min|mu+1| at gamma_below = %.6g\n', min(abs(mu_b + 1)));
fprintf('min|mu+1| at gamma_above = %.6g\n', min(abs(mu_a + 1)));
drawnow;
exportgraphics(gcf,"Fig7.svg",'Resolution',600);
