%% Bistability demo (equilibrium vs. limit cycle) for Leslie–Gower IGP
% du/dt = u(1 - u/w) - u v
% dv/dt = alpha*v(1 - beta*v/w) + u v
% dw/dt = w*(gamma - delta*u - epsilon*v)

clear; clc; close all;

%% ---------- Parameters (choose a regime prone to hysteresis/subcritical Hopf) ----------
alpha   = 0.0052;
beta    = 0.5;
delta   = 2.0;
epsilon = 2.5;
% gamma scan window to detect Hopf and then search for bistability
g_search = [0.10, 1.00];

fprintf('Params: alpha=%.3g, beta=%.3g, delta=%.3g, epsilon=%.3g\n',alpha,beta,delta,epsilon);

%% ---------- 1) Locate a Hopf candidate by scanning gamma ----------
Ng = 900;
G  = linspace(g_search(1), g_search(2), Ng);
sigma = nan(1,Ng);      % max Re eigenvalue at interior equilibrium (smaller-z branch)

for k=1:Ng
    p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',G(k));
    EQ = interior_equilibria(p);
    if isempty(EQ), continue; end
    [~,ix] = min([EQ.z]); E = EQ(ix);
    ev = eig(jacobian_at(E.x,E.y,E.z,p));
    sigma(k) = max(real(ev));
end

gH = crossing_points(G, sigma);
assert(~isempty(gH), 'No Hopf crossing detected. Adjust g_search or parameters.');
gamma_H = gH(1);
fprintf('Detected Hopf candidate near gamma_H = %.6f\n', gamma_H);

%% ---------- 2) Find a gamma with bistability (eq stable for near-IC; cycle for far-IC) ----------
% We probe a small window around gamma_H on BOTH sides.
win   = 0.12 * (g_search(2)-g_search(1));
Gsamp = unique([linspace(max(g_search(1),gamma_H-0.8*win),gamma_H,12), ...
            linspace(gamma_H,min(g_search(2),gamma_H+0.8*win),12)]);
Tmax   = 360;      % integration horizon
trans  = 0.5;      % discard first half as transient
thrSmall = 5e-4;   % "near 0 amplitude" threshold (equilibrium)
thrBig   = 5e-3;   % "definitely oscillatory" threshold (cycle)
IC_far   = [0.2; 0.15; 0.25];  % far initial condition

found = false; gamma_bi = NaN; Ebi = [];
for gi = Gsamp
    p  = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gi);
    EQ = interior_equilibria(p);
    if isempty(EQ), continue; end
    [~,ix] = min([EQ.z]); E = EQ(ix);           % use smaller-z interior equilibrium
    IC_near = [E.x;E.y;E.z] .* (1 + 1e-3*[1;-1;1]);

    A_near = measure_amp(p, IC_near, Tmax, trans);
    A_far  = measure_amp(p, IC_far,  Tmax, trans);

    % Bistability signature: near-IC -> small amplitude (eq), far-IC -> large amplitude (cycle)
    if (A_near <= thrSmall) && (A_far >= thrBig)
        gamma_bi = gi; Ebi = E; found = true; break;
    end
end
assert(found, 'Could not find a gamma showing eq–cycle bistability in the test window. Tweak parameters or thresholds.');

fprintf('Bistability found at gamma = %.6f (near-IC -> eq; far-IC -> cycle)\n', gamma_bi);

%% ---------- 3) Simulate the two outcomes at gamma_bi ----------
pbi = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gamma_bi);
opts = odeset('RelTol',1e-9,'AbsTol',1e-12,'MaxStep',0.5);

IC_eq    = [Ebi.x;Ebi.y;Ebi.z] .* (1 + 1e-3*[1;-1;1]);  % near equilibrium
IC_cycle = IC_far;                                      % far initial condition (goes to cycle)

[t1, X1] = ode45(@(t,x) igp_rhs(t,x,pbi), [0 Tmax], max(IC_eq,1e-12), opts);
[t2, X2] = ode45(@(t,x) igp_rhs(t,x,pbi), [0 Tmax], max(IC_cycle,1e-12), opts);

% post-transient windows
idx1 = t1 >= t1(1) + trans*(t1(end)-t1(1));
idx2 = t2 >= t2(1) + trans*(t2(end)-t2(1));

% measure amplitudes & period on the cycle trajectory
[Tper, Au, Av, Aw] = measure_period_amp(t2(idx2), X2(idx2,1), X2(idx2,2), X2(idx2,3));
Amax = max([Au,Av,Aw]);
fprintf('Cycle at gamma=%.6f: period T=%.6f, amplitudes [Au,Av,Aw]=[%.3g, %.3g, %.3g]\n', ...
        gamma_bi, Tper, Au, Av, Aw);

%% ---------- 4) Plots: time series (eq vs cycle) ----------
figure('Color','w','Name','Bistability: time series');
subplot(1,2,1); hold on; grid on; box on;
plot(t1, X1(:,1),'LineWidth',1.6); plot(t1, X1(:,2),'LineWidth',1.6); plot(t1, X1(:,3),'LineWidth',1.6);
yline(Ebi.x,'--','Color',[0 0 0 0.25]); yline(Ebi.y,'--','Color',[0 0 0 0.25]); yline(Ebi.z,'--','Color',[0 0 0 0.25]);
title(sprintf('Converges to equilibrium  (\\gamma=%.4f)', gamma_bi));
xlabel('t'); ylabel('states'); legend({'u','v','w'},'Location','best');

subplot(1,2,2); hold on; grid on; box on;
plot(t2, X2(:,1),'LineWidth',1.6); plot(t2, X2(:,2),'LineWidth',1.6); plot(t2, X2(:,3),'LineWidth',1.6);
title(sprintf('Converges to limit cycle  (\\gamma=%.4f,  T=%.3g)', gamma_bi, Tper));
xlabel('t'); ylabel('states'); legend({'u','v','w'},'Location','best');

%% ---------- 5) Plot: 3D phase portrait with BOTH attractors ----------
figure('Color','w','Name','Bistability: phase portrait');
hold on; grid on; box on; view(45,25);
% equilibrium trajectory (post-transient tail)
plot3(X1(idx1,1), X1(idx1,2), X1(idx1,3), '-', 'LineWidth',1.5, 'Color',[0.1 0.45 0.85]);
% cycle trajectory (post-transient)
plot3(X2(idx2,1), X2(idx2,2), X2(idx2,3), '-', 'LineWidth',2.0, 'Color',[0.85 0.25 0.15]);
% markers
plot3(X1(1,1), X1(1,2), X1(1,3), 'o', 'MarkerSize',6, 'MarkerFaceColor',[0.1 0.45 0.85], 'MarkerEdgeColor','k');
plot3(X2(1,1), X2(1,2), X2(1,3), '^', 'MarkerSize',7, 'MarkerFaceColor',[0.85 0.25 0.15], 'MarkerEdgeColor','k');
plot3(Ebi.x, Ebi.y, Ebi.z, 'p', 'MarkerSize',10, 'MarkerFaceColor',[0.15 0.15 0.15], 'MarkerEdgeColor','k');
xlabel('u'); ylabel('v'); zlabel('w');
legend({'toward equilibrium','limit cycle','IC (eq)','IC (cycle)','equilibrium'},'Location','best');
title(sprintf('Bistability at \\gamma=%.4f  (eq \\& cycle attractors)', gamma_bi));

%% ---------- 6) (Optional) 1-D basin slice along a line of initial conditions ----------
do_slice = true;
if do_slice
    dirv = [0.5; -0.4; 0.2]; dirv = dirv / norm(dirv);
    sVals = linspace(-0.4, 1.6, 30);   % offsets along the line
    cls   = nan(size(sVals));
    for i=1:numel(sVals)
        x0 = [Ebi.x;Ebi.y;Ebi.z] + sVals(i)*dirv .* max([Ebi.x,Ebi.y,Ebi.z],[],2)';
        x0 = max(x0, 1e-12);
        Amp = measure_amp(pbi, x0, Tmax, trans);
        cls(i) = (Amp >= thrBig); % 0 -> eq, 1 -> cycle
    end
    figure('Color','w','Name','Basin slice (eq vs cycle)');
    stem(sVals, cls, 'filled', 'LineWidth',1.2); ylim([-0.2 1.2]); grid on; box on;
    yticks([0 1]); yticklabels({'equilibrium','cycle'});
    xlabel('offset s along line through equilibrium'); ylabel('attractor class');
    title('1-D basin slice: initial-condition dependence');
end

%% ==================== helper functions ====================

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
    idx  = find(t >= tcut);
    if numel(idx) < 12, Amp = 0; return; end
    xs = X(idx,1); ys = X(idx,2); zs = X(idx,3);
    Amp = max([max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs)]);
end

function [Tper, Au, Av, Aw] = measure_period_amp(t, u, v, w)
    Au = max(u)-min(u); Av = max(v)-min(v); Aw = max(w)-min(w);
    [~,which] = max([Au,Av,Aw]); sigs = {u,v,w}; s = sigs{which};
    Tper = estimate_period(t, s);
end

function Tper = estimate_period(t, s)
    s = s - mean(s);
    n = numel(s); if n < 6, Tper = NaN; return; end
    % peaks method
    idx = 2:n-1;
    pk  = idx( s(2:end-1) > s(1:end-2) & s(2:end-1) >= s(3:end) );
    if ~isempty(pk)
        promin = 0.15*max(abs(s));
        pk = pk(s(pk) >= promin);
    end
    if numel(pk) >= 3
        k = min(6, numel(pk)-1);
        Tper = median(diff(t(pk(end-k+1:end)))); if Tper > 0, return; end
    end
    % autocorr fallback
    dt = mean(diff(t));
    ac = xcorr(s, 'coeff'); mid = (numel(ac)+1)/2; acp = ac(mid+1:end);
    L = numel(acp);
    if L >= 5
        jj = 2:L-1; locMax = jj( acp(jj) > acp(jj-1) & acp(jj) >= acp(jj+1) & acp(jj) > 0 );
        if ~isempty(locMax), Tper = locMax(1)*dt; if Tper > 0, return; end, end
    end
    % FFT fallback
    S = abs(fft(s)); S(1)=0; f = (0:n-1)/(n*dt);
    [~,imx] = max(S(1:floor(n/2))); fmax = f(imx);
    Tper = (fmax>0) * (1/fmax) + (fmax==0) * NaN;
end

function EQ = interior_equilibria(p)
    % Solve c1 z^2 + c2 z + c3 = 0; keep positive z; back out x,y>0
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