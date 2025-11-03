%% Hopf bifurcation demo (Leslie–Gower IGP)
% du/dt = u(1 - u/w) - u v
% dv/dt = alpha*v(1 - beta*v/w) + u v
% dw/dt = w*(gamma - delta*u - epsilon*v)
%
% What this script does:
% 1) Find an interior equilibrium branch and locate gamma_H where max Re(eig) crosses 0.
% 2) Build an amplitude-vs-gamma diagram by integrating near gamma_H:
%       - supercritical: small stable cycles appear on the side where eq. becomes unstable
%       - subcritical/hysteresis: amplitudes jump; different ICs track different attractors
% 3) Plot time series just below/above gamma_H showing the birth of oscillations.

clear; clc; close all;

%% ---------------- Choose a preset ----------------
% 'super' tends to show a supercritical Hopf; 'sub' tends to show hysteresis.
CASE = 'super';  % <- set to 'sub' to explore a subcritical-like regime

switch lower(CASE)
    case 'super'
        alpha=1.0; beta=1.0; delta=2.0; epsilon=0.10;
        g_search = [0.08 2];   % gamma scan range
    case 'sub'
        % A regime that often shows hysteresis (bistability with cycles further out).
        % You can tweak these to deepen the subcritical effect.
        alpha=1.0; beta=1.2; delta=1.6; epsilon=0.18;
        g_search = [0.10 2];
    otherwise, error('CASE must be ''super'' or ''sub''.');
end

fprintf('Params: alpha=%.3g, beta=%.3g, delta=%.3g, epsilon=%.3g\n', alpha,beta,delta,epsilon);

%% ---------------- 1) Find Hopf candidate by scanning gamma ----------------
Ng   = 800;
G    = linspace(g_search(1), g_search(2), Ng);
tolStable = 1e-6;

% storage (interior branch 1 only; if two exist we pick the one with smaller z)
U = nan(1,Ng); V = nan(1,Ng); W = nan(1,Ng);
sig = nan(1,Ng);   % sigma(gamma) = max real part of eigenvalues at interior eq.

for k=1:Ng
    p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',G(k));
    EQ = interior_equilibria(p);
    if isempty(EQ), continue; end
    % Use the branch with smaller z (typically the one that undergoes Hopf first)
    [~,ix] = min([EQ.z]); E = EQ(ix);
    U(k)=E.x; V(k)=E.y; W(k)=E.z;
    ev = eig(jacobian_at(E.x,E.y,E.z,p));
    sig(k) = max(real(ev));
end

% locate sign change in sigma (Hopf candidate)
gH = crossing_points(G, sig);
if isempty(gH)
    error('No Hopf candidate detected in the scanned range. Expand g_search or adjust parameters.');
end
gamma_H = gH(1);   % take the first crossing
fprintf('Detected Hopf candidate near gamma_H = %.6f\n', gamma_H);

%% ---------------- 2) Amplitude-vs-gamma diagram near gamma_H ----------------
% We sample on both sides of gamma_H; we integrate from two different ICs
% (near the equilibrium and a "far" IC) to reveal hysteresis if present.

% sample grid around Hopf
dL = 0.12*abs(g_search(2)-g_search(1));   % window size (relative to search interval)
g_left  = linspace(max(g_search(1), gamma_H-0.6*dL), gamma_H, 16);
g_right = linspace(gamma_H, min(g_search(2), gamma_H+0.6*dL), 16);
Gscan   = unique([g_left, g_right]);  Gscan = Gscan(:)';

% integration setup
Tmax   = 350;       % horizon
trans  = 0.5;       % discard first 50% as transient
ampThr = 1e-3;      % amplitude threshold
IC_near_scale = 1e-3;      % tiny perturbation about equilibrium
IC_far  = [0.2; 0.15; 0.25];  % "far" initial condition to probe other basins

Amp_near = nan(size(Gscan));  % amplitude using near-IC
Amp_far  = nan(size(Gscan));  % amplitude using far-IC

for i=1:numel(Gscan)
    gi = Gscan(i);
    p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gi);

    % equilibrium at this gamma (take interior with smaller z)
    EQ = interior_equilibria(p);
    if isempty(EQ), continue; end
    [~,ix] = min([EQ.z]); E = EQ(ix);

    % near-equilibrium initial condition
    IC_near = [E.x;E.y;E.z] .* (1+IC_near_scale*[1;-1;1]);
    Amp_near(i) = measure_amp(p, IC_near, Tmax, trans);

    % far initial condition
    Amp_far(i) = measure_amp(p, IC_far, Tmax, trans);
end

% classify criticality (simple empirical rule)
% If Amp_near just above Hopf is small & positive, it's supercritical.
% If Amp_near ~ 0 above Hopf but Amp_far jumps big (and depends on direction), hysteresis/subcritical.
[~,idxH] = min(abs(Gscan - gamma_H));
sideIdx = min(numel(Gscan), max(1, [idxH-2, idxH+2]));

amp_left  = nanmean(Amp_near(1:idxH-1));
amp_right = nanmean(Amp_near(idxH+1:end));
is_super  = (amp_right > 5*ampThr) && (amp_left <= 5*ampThr);

if is_super
    crit_str = 'SUPERCRITICAL Hopf (small stable cycles for \gamma > \gamma_H)';
else
    crit_str = 'SUBCRITICAL/HYSTERESIS (no small cycle from near-IC; jump in far-IC)';
end
fprintf('Classification: %s\n', crit_str);

% Plot amplitude diagram
figure('Color','w'); hold on; grid on; box on;
plot(Gscan, Amp_near, 'o-', 'LineWidth',1.6, 'MarkerSize',5, 'DisplayName','near-IC amplitude');
plot(Gscan, Amp_far,  's-', 'LineWidth',1.6, 'MarkerSize',5, 'DisplayName','far-IC amplitude');
xline(gamma_H,'--k','LineWidth',1.2,'Label','\gamma_H','LabelVerticalAlignment','bottom');
xlabel('\gamma'); ylabel('Cycle amplitude (max peak-to-peak across u,v,w)');
title(sprintf('Amplitude vs \\gamma near Hopf (\\alpha=%.2g, \\beta=%.2g, \\delta=%.2g, \\epsilon=%.2g)\n%s', ...
    alpha,beta,delta,epsilon, crit_str));
legend('Location','best');

%% ---------------- 3) Sample trajectories below / above Hopf ----------------
% Choose gammas flanking gamma_H. If supercritical, pick very close on both sides.
% If subcritical/hysteresis, pick on the side showing larger amplitude.

g_below = max(g_search(1), gamma_H - 0.03*abs(diff(g_search)));
g_above = min(g_search(2), gamma_H + 0.03*abs(diff(g_search)));

IC = [0.12; 0.10; 0.18];  % generic positive IC

% below
pL = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',g_below);
[tL, XL] = ode45(@(t,x) igp_rhs(t,x,pL), [0 350], IC, odeset('RelTol',1e-9,'AbsTol',1e-12));
% above
pH = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',g_above);
[tH, XH] = ode45(@(t,x) igp_rhs(t,x,pH), [0 350], IC, odeset('RelTol',1e-9,'AbsTol',1e-12));

figure('Color','w','Name','Time series across Hopf');
subplot(1,2,1); hold on; grid on; box on;
plot(tL, XL(:,1),'LineWidth',1.6); plot(tL, XL(:,2),'LineWidth',1.6); plot(tL, XL(:,3),'LineWidth',1.6);
xlabel('t'); ylabel('states'); title(sprintf('Below Hopf: \\gamma=%.4f (eq. stable)', g_below));
legend({'u(t)','v(t)','w(t)'},'Location','best');

subplot(1,2,2); hold on; grid on; box on;
plot(tH, XH(:,1),'LineWidth',1.6); plot(tH, XH(:,2),'LineWidth',1.6); plot(tH, XH(:,3),'LineWidth',1.6);
xlabel('t'); ylabel('states'); title(sprintf('Above Hopf: \\gamma=%.4f (oscillatory)', g_above));
legend({'u(t)','v(t)','w(t)'},'Location','best');

%% ==================== helpers ====================

function Amp = measure_amp(p, x0, T, fracTransient)
    % Integrate and return peak-to-peak amplitude (max over u,v,w) after transient
    rhs  = @(t,x) igp_rhs(t,x,p);
    opts = odeset('RelTol',1e-8,'AbsTol',1e-11,'MaxStep',0.5);
    x0   = max(x0, 1e-12);
    try
        [t, X] = ode45(rhs, [0 T], x0, opts);
    catch
        opts = odeset(opts,'MaxStep',0.1);
        [t, X] = ode45(rhs, [0 T], x0, opts);
    end
    tcut = t(1) + fracTransient*(t(end)-t(1));
    idx  = find(t >= tcut);
    if numel(idx) < 12, Amp = 0; return; end
    xs = X(idx,1); ys = X(idx,2); zs = X(idx,3);
    Amp = max([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]);
end

function EQ = interior_equilibria(p)
    % Solve c1 z^2 + c2 z + c3 = 0; keep positive z, then x,y>0
    c1 = (p.gamma - p.epsilon)/p.alpha + p.delta;
    c2 = -(p.beta*p.delta + p.epsilon);
    c3 =  p.beta*p.gamma;
    r = roots([c1 c2 c3]);
    zc = r(abs(imag(r))<1e-12); zc = real(zc); zc = zc(zc>0);
    EQ = struct('x',{},'y',{},'z',{});
    for i=1:numel(zc)
        z = zc(i);
        den = z^2 + p.alpha*p.beta;
        x = (p.alpha*z*(p.beta - z))/den;
        y = (z*(p.alpha + z))/den;
        if x>0 && y>0
            EQ(end+1) = struct('x',x,'y',y,'z',z);  
        end
    end
end

function J = jacobian_at(u, v, w, p)
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

function gxs = crossing_points(G, val)
    % find parameter values where val crosses 0 (linear interpolation)
    gxs = [];
    for k=1:numel(G)-1
        a = val(k); b = val(k+1);
        if any(isnan([a b])), continue; end
        if a==0, gxs(end+1) = G(k); 
        elseif b==0, gxs(end+1) = G(k+1); 
        elseif a*b < 0
            t = abs(a)/(abs(a)+abs(b));
            gxs(end+1) = (1-t)*G(k) + t*G(k+1); 
        end
    end
end

function dx = igp_rhs(~, X, p)
    u = max(X(1),0); v = max(X(2),0); w = max(X(3),0);
    if w < 1e-14, w = 1e-14; end
    dx = zeros(3,1);
    dx(1) = u*(1 - u/w) - u*v;
    dx(2) = p.alpha*v*(1 - p.beta*v/w) + u*v;
    dx(3) = w*(p.gamma - p.delta*u - p.epsilon*v);
end