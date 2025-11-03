%% Leslie–Gower IGP model: Simulation 1 (gamma = 0.05 < epsilon)
% Model (dimensionless):
%   dx/dt = x(1 - x/z) - x y
%   dy/dt = alpha*y(1 - beta*y/z) + x y
%   dz/dt = z*(gamma - delta*x - epsilon*y)
%
% Parameters for Simulation 1:
%   alpha = 1, beta = 1, delta = 2, epsilon = 0.1, gamma = 0.05
% Behavior: unique positive equilibrium; trajectories converge to coexistence.

clear; clc; close all;

%% Parameters
params1 = struct('alpha',1,'beta',2,'delta',0.5,'epsilon',1,'gamma',0.05);
params2 = struct('alpha',1,'beta',2,'delta',0.5,'epsilon',0.2,'gamma',0.2);
params3 = struct('alpha',1,'beta',2,'delta',0.5,'epsilon',0.2,'gamma',0.24);
params4 = struct('alpha',1,'beta',2,'delta',0.5,'epsilon',0.2,'gamma',0.3);
params5 = struct('alpha',1,'beta',2,'delta',0.5,'epsilon',1,'gamma',2);
params6 = struct('alpha',1,'beta',2,'delta',0.5,'epsilon',0.2,'gamma',0.5);
Params = {params1, params2, params3, params4, params5, params6};

%% Compute interior equilibrium analytically via quadratic in z:
% c1 z^2 + c2 z + c3 = 0
labels = {'(a)', '(b)', '(c)', '(d)', '(e)', '(f)'};

defaultsetting;
figure('Color','w', 'Position',[100,100,1200,800]);
tiledlayout(3,2,"TileSpacing","tight","Padding","tight");
for i = 1:6
    params = Params{i};
    [x_star, y_star, z_star] = interior_equilibrium(params);
    alpha = params.alpha;
    beta = params.beta;
    delta = params.delta;
    epsilon = params.epsilon;
    gamma = params.gamma;

    gamma_plus = (alpha/2) * (epsilon/alpha - delta + sqrt((epsilon/alpha - delta)^2 + (beta*delta + epsilon)^2 / (alpha * beta)));
    fprintf('the %d-th simulation starts', i);
    fprintf("the critical gamma1 is %.4f\n", gamma_plus);
    fprintf('N* = %.6g,  P* = %.6g,  R* = %.6g\n', x_star, y_star, z_star);

    %% Local stability: Jacobian and eigenvalues at the equilibrium
    J = jacobian_at_equilibrium(x_star, y_star, z_star, params);
    eigJ = eig(J);
    fprintf('Jacobian eigenvalues at (N*,P*,R*):\n');
    disp(eigJ);

    %% Time integration (ODE45)
    tspan = [0 500];                 % long enough to see convergence
    x0 = [0.5; 0.5; 0.5];         % positive initial condition
    opts = odeset('RelTol',1e-9,'AbsTol',1e-12);
    rhs  = @(t,x) igp_rhs(t,x,params);

    [t, X] = ode45(rhs, tspan, x0, opts);

    x = X(:,1); y = X(:,2); z = X(:,3);

    %% Plots
    nexttile;
    plot(t, x); hold on;
    plot(t, y);
    plot(t, z);
    yline(x_star,'--','Color',[0 0 0 0.25]);
    yline(y_star,'--','Color',[0 0 0 0.25]);
    yline(z_star,'--','Color',[0 0 0 0.25]);
    legend({'N(t)','P(t)','R(t)'}, 'Location','northeast');
    xlabel('t'); ylabel('states');
    title(labels{i});
    grid on; box on;

    %% (Optional) print gamma_SN for context (saddle-node threshold)
    gamma_SN = 0.5*alpha*( (epsilon/alpha) - delta + sqrt( (epsilon/alpha - delta)^2 + ((beta*delta+epsilon)^2)/(alpha*beta) ) );
    fprintf('Saddle-node threshold gamma_SN (for these alpha,beta,delta,epsilon): %.6g\n', gamma_SN);
    fprintf('Here gamma = %.3f < epsilon = %.3f < gamma_SN = %.3f (unique interior equilibrium expected).\n', gamma, epsilon, gamma_SN);
end

drawnow;
exportgraphics(gcf, 'Fig1a.svg', 'Resolution',600);  

%%%%
figure('Color','w','Position',[100,100,1200,800]);
tiledlayout(3,2,"TileSpacing","tight","Padding","tight");
for i = 1:6
    params = Params{i};
    [x_star, y_star, z_star] = interior_equilibrium(params);
    alpha = params.alpha;
    beta = params.beta;
    delta = params.delta;
    epsilon = params.epsilon;
    gamma = params.gamma;

    gamma_plus = (alpha/2) * (epsilon/alpha - delta + sqrt((epsilon/alpha - delta)^2 + (beta*delta + epsilon)^2 / (alpha * beta)));
    fprintf('the %d-th simulation starts', i);
    fprintf("the critical gamma1 is %.4f\n", gamma_plus);
    fprintf('N* = %.6g,  P* = %.6g,  R* = %.6g\n', x_star, y_star, z_star);

    %% Local stability: Jacobian and eigenvalues at the equilibrium
    J = jacobian_at_equilibrium(x_star, y_star, z_star, params);
    eigJ = eig(J);
    fprintf('Jacobian eigenvalues at (N*,P*,R*):\n');
    disp(eigJ);

    %% Time integration (ODE45)
    tspan = [0 500];                 % long enough to see convergence
    x0 = [0.5; 0.5; 0.5];         % positive initial condition
    opts = odeset('RelTol',1e-9,'AbsTol',1e-12);
    rhs  = @(t,x) igp_rhs(t,x,params);

    [t, X] = ode45(rhs, tspan, x0, opts);

    x = X(:,1); y = X(:,2); z = X(:,3);

    nexttile;
    plot3(x, y, z); hold on;
    plot3(x(1), y(1), z(1), 'o', 'MarkerSize',6, 'MarkerFaceColor',[0.2 0.2 0.2], 'MarkerEdgeColor','k');
    if i <= 4
        plot3(x_star, y_star, z_star, 'p', 'MarkerSize',10, 'MarkerFaceColor',[0.85 0.2 0.2], 'MarkerEdgeColor','k');
    end
    grid on; box on;
    xlabel('N (IG-prey)'); ylabel('P (IG-predator)'); zlabel('R (resource)');
    title(labels{i});
    view(45,25);
end

drawnow;
exportgraphics(gcf, 'Fig1b.svg', 'Resolution',600);

%% ---------- Local functions ----------

function dx = igp_rhs(~, x, p)
% Right-hand side of the IGP ODE system.
% x = [x; y; z]
X = max(x(1), 0);   % guard against tiny negatives from numerics
Y = max(x(2), 0);
Z = max(x(3), 0);
% Avoid division by zero in the rare event Z ~ 0 numerically:
if Z < 1e-14, Z = 1e-14; end

dX = X*(1 - X/Z) - X*Y;
dY = p.alpha*Y*(1 - p.beta*Y/Z) + X*Y;
dZ = Z*(p.gamma - p.delta*X - p.epsilon*Y);

dx = [dX; dY; dZ];
end

function [x_star, y_star, z_star] = interior_equilibrium(p)
% Solve c1 z^2 + c2 z + c3 = 0 for z>0, then back out x*, y*:
% x* = (alpha*z*(beta - z))/(z^2 + alpha*beta)
% y* = (z*(alpha + z))      /(z^2 + alpha*beta)
c1 = (p.gamma - p.epsilon)/p.alpha + p.delta;
c2 = -(p.beta*p.delta + p.epsilon);
c3 = p.beta*p.gamma;

r = roots([c1 c2 c3]);
z_pos = r(real(r) > 0 & abs(imag(r)) < 1e-12);
if isempty(z_pos)
    disp('No positive interior equilibrium for these parameters.');
    x_star = 0; y_star = 0; z_star = 0;
    return;
end
% If two positive roots exist, pick the one consistent with positivity of x*, y*.
% Here (gamma < epsilon) there should be exactly one positive root.
z_star = min(z_pos);  % safe choice

denom = (z_star^2 + p.alpha*p.beta);
x_star = (p.alpha*z_star*(p.beta - z_star)) / denom;
y_star = (z_star*(p.alpha + z_star)) / denom;

if ~(x_star > 0 && y_star > 0)
    disp("the equilibrium with min z is not positive, continue with bigger z")
    z_star = max(z_pos);
    denom = (z_star^2 + p.alpha*p.beta);
    x_star = (p.alpha*z_star*(p.beta - z_star)) / denom;
    y_star = (z_star*(p.alpha + z_star)) / denom;
    if ~(x_star > 0 && y_star > 0)
        disp("the equilibrium with bigger z is not positive, pay attention")
    end
end
end

function J = jacobian_at_equilibrium(x, y, z, p)
% Jacobian matrix of the system at (x,y,z):
% See derivation in the manuscript.
a11 = 1 - 2*x/z - y;
a12 = -x;
a13 = (x^2)/(z^2);

a21 = y;
a22 = p.alpha - 2*p.alpha*p.beta*y/z + x;
a23 = (p.alpha*p.beta*y^2)/(z^2);

a31 = -p.delta*z;
a32 = -p.epsilon*z;
a33 = p.gamma - p.delta*x - p.epsilon*y;

J = [a11 a12 a13; a21 a22 a23; a31 a32 a33];
end