%% Transcritical bifurcation demo for the Leslie–Gower IGP model
% du/dt = u(1 - u/w) - u v
% dv/dt = alpha*v(1 - beta*v/w) + u v
% dw/dt = w*(gamma - delta*u - epsilon*v)
%
% We pick parameters with beta*delta <= epsilon so that:
%  - For gamma < epsilon: unique interior (coexistence) equilibrium (typically stable).
%  - For gamma > epsilon: NO interior equilibrium; boundary P2 is stable => prey (u) extinct.
% With beta=1, the interior branch hits the boundary at z = beta = 1 when gamma = epsilon,
% so it crosses P2 at exactly (u,v,w) = (0,1,1): a clean transcritical picture.

clear; clc; close all;

%% Parameters (canonical transcritical case)
alpha   = 1.0;
beta    = 1.0;      % keeps the crossing point exactly at (0,1,1) when gamma=epsilon
delta   = 0.05;
epsilon = 0.10;     % transcritical threshold at gamma = epsilon

fprintf('Transcritical at gamma = epsilon = %.3f\n', epsilon);
fprintf('Here beta*delta = %.3f <= epsilon, so interior disappears for gamma > epsilon.\n', beta*delta);

%% 1) One-parameter sweep in gamma to plot branches and stability
gmin = 0.01; gmax = 0.25; Ng = 600;
G = linspace(gmin, gmax, Ng);
tolStable = 1e-7;

% Boundary P2 branch
xP2 = zeros(size(G));
yP2 = G./epsilon;
zP2 = yP2;
P2_stable = G > epsilon;   % analytic stability of P2

% Interior (coexistence) branch (0, 1, or 2 roots but here at most 1 for gamma <= epsilon)
U = nan(1,Ng); V = nan(1,Ng); W = nan(1,Ng); ST = false(1,Ng);
for k = 1:Ng
    gamma = G(k);
    p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gamma);
    EQ = interior_equilibria(p);   % (may be empty for gamma>epsilon)
    if ~isempty(EQ)
        % In this parameter regime (beta*delta <= epsilon) we expect at most 1 interior
        U(k) = EQ(1).x; V(k) = EQ(1).y; W(k) = EQ(1).z;
        ST(k) = is_stable(EQ(1), p, tolStable);
    end
end

% Exact transcritical point (gamma = epsilon): crossing at (u,v,w) = (0,1,1) for beta=1
u_TC = 0; v_TC = 1; w_TC = 1; g_TC = epsilon;

%% Plot: (u*, v*, w*) vs gamma with stability and branch crossing
f = figure('Color','w'); t = tiledlayout(3,1,'TileSpacing','compact','Padding','compact');
labels = {'u^* (IG-prey)','v^* (IG-predator)','w^* (resource)'};
datInt = {U, V, W};
datP2  = {xP2, yP2, zP2};

for r = 1:3
    ax = nexttile; hold(ax,'on'); grid(ax,'on'); box(ax,'on');

    % interior branch: solid if stable, dashed if unstable
    plot_branch(ax, G, datInt{r}, ST, [0.10 0.45 0.85], 2.0);

    % boundary P2: solid if stable, dashed if unstable
    stab = P2_stable;
    plot_branch(ax, G, datP2{r}, stab, [0.15 0.15 0.15], 1.8);

    % transcritical marker
    tcVal = [u_TC, v_TC, w_TC]; tcVal = tcVal(r);
    plot(ax, g_TC, tcVal, 'kp', 'MarkerSize',9, 'MarkerFaceColor',[0.1 0.1 0.1], ...
        'DisplayName','transcritical point');

    xline(ax, epsilon, '--k', 'LineWidth',1.2, 'Label','\gamma=\epsilon', ...
        'LabelOrientation','horizontal','LabelVerticalAlignment','bottom');

    ylabel(ax, labels{r});
    if r==1
        title(ax, sprintf('Transcritical bifurcation: branch crossing at \\gamma=\\epsilon (\\alpha=%.2g, \\beta=%.2g, \\delta=%.2g, \\epsilon=%.2g)', ...
            alpha,beta,delta,epsilon));
    end
    if r==3, xlabel(ax, '\gamma'); end
end
legend(nexttile(1), 'Location','best');

%% 2) Trajectories just below and above the bifurcation
% Below (gamma < epsilon): coexistence attractor
% Above (gamma > epsilon): predator-only boundary P2 (u -> 0)
IC = [0.2; 0.15; 0.3];    % common initial condition (positive)

Tmax = 300; opts = odeset('RelTol',1e-9,'AbsTol',1e-12);
gam_low  = epsilon - 0.02;
gam_high = epsilon + 0.02;

pL = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gam_low);
pH = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gam_high);

rhsL = @(t,x) igp_rhs(t,x,pL);
rhsH = @(t,x) igp_rhs(t,x,pH);

[tL, XL] = ode45(rhsL, [0 Tmax], IC, opts);
[tH, XH] = ode45(rhsH, [0 Tmax], IC, opts);

% Plot time series (left: below -> coexistence; right: above -> boundary extinction of u)
figure('Color','w');
subplot(1,2,1); hold on; grid on; box on;
plot(tL, XL(:,1), 'LineWidth',1.6); plot(tL, XL(:,2), 'LineWidth',1.6); plot(tL, XL(:,3), 'LineWidth',1.6);
title(sprintf('Below: \\gamma=%.3f < \\epsilon (coexistence)', gam_low));
xlabel('t'); ylabel('states'); legend({'u(t)','v(t)','w(t)'},'Location','best');

subplot(1,2,2); hold on; grid on; box on;
plot(tH, XH(:,1), 'LineWidth',1.6); plot(tH, XH(:,2), 'LineWidth',1.6); plot(tH, XH(:,3), 'LineWidth',1.6);
title(sprintf('Above: \\gamma=%.3f > \\epsilon (u \\rightarrow 0, P_2)', gam_high));
xlabel('t'); ylabel('states'); legend({'u(t)','v(t)','w(t)'},'Location','best');

%% ==================== helpers ====================

function EQ = interior_equilibria(p)
% Solve c1 z^2 + c2 z + c3 = 0; keep positive z; back out x,y>0
    c1 = (p.gamma - p.epsilon)/p.alpha + p.delta;
    c2 = -(p.beta*p.delta + p.epsilon);
    c3 =  p.beta*p.gamma;
    r = roots([c1 c2 c3]);
    zc = r(abs(imag(r))<1e-12);
    zc = real(zc); zc = zc(zc > 0);  % positive z only

    EQ = struct('x',{},'y',{},'z',{});
    for i = 1:numel(zc)
        z = zc(i);
        den = z^2 + p.alpha*p.beta;
        x = (p.alpha*z*(p.beta - z))/den;
        y = (z*(p.alpha + z))/den;
        if x > 0 && y > 0
            EQ(end+1) = struct('x',x,'y',y,'z',z); 
        end
    end
end

function J = jacobian_at(u, v, w, p)
% Jacobian of the vector field at (u,v,w)
    a11 = 1 - 2*u/w - v;
    a12 = -u;
    a13 = (u^2)/(w^2);

    a21 = v;
    a22 = p.alpha - 2*p.alpha*p.beta*v/w + u;
    a23 = (p.alpha*p.beta*v^2)/(w^2);

    a31 = -p.delta*w;
    a32 = -p.epsilon*w;
    a33 = p.gamma - p.delta*u - p.epsilon*v;

    J = [a11 a12 a13; a21 a22 a23; a31 a32 a33];
end

function tf = is_stable(E, p, tol)
% Stable iff max real part of eigenvalues < -tol
    ev = eig(jacobian_at(E.x, E.y, E.z, p));
    tf = max(real(ev)) < -tol;
end

function dx = igp_rhs(~, X, p)
% RHS of the ODE with guards for tiny negatives
    u = max(X(1),0); v = max(X(2),0); w = max(X(3),0);
    if w < 1e-14, w = 1e-14; end
    dx = zeros(3,1);
    dx(1) = u*(1 - u/w) - u*v;
    dx(2) = p.alpha*v*(1 - p.beta*v/w) + u*v;
    dx(3) = w*(p.gamma - p.delta*u - p.epsilon*v);
end

function plot_branch(ax, G, val, stab, col, lw)
% Plot solid (stable) and dashed (unstable), skipping NaNs
    isn = isnan(val);
    ed = find(diff([true, ~isn, true]) ~= 0);
    starts = ed(1:2:end-1); stops  = ed(2:2:end)-1;
    for k=1:numel(starts)
        idx = starts(k):stops(k);
        Gi = G(idx); Vi = val(idx); Si = stab(idx);
        [s1,e1] = runs_of_true(Si);
        for r=1:numel(s1)
            ii = s1(r):e1(r);
            plot(ax, Gi(ii), Vi(ii), '-', 'Color', col, 'LineWidth', lw, 'DisplayName','stable');
        end
        [s0,e0] = runs_of_true(~Si);
        for r=1:numel(s0)
            ii = s0(r):e0(r);
            plot(ax, Gi(ii), Vi(ii), '--', 'Color', col, 'LineWidth', lw, 'DisplayName','unstable');
        end
    end
end

function [s,e] = runs_of_true(mask)
    if isempty(mask), s = []; e = []; return; end
    d = diff([false, mask(:).', false]);
    s = find(d==1); e = find(d==-1)-1;
end