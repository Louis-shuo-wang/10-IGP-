%% Limit-cycle demo for Leslie–Gower IGP model
% du/dt = u(1 - u/w) - u v
% dv/dt = alpha*v(1 - beta*v/w) + u v
% dw/dt = w*(gamma - delta*u - epsilon*v)

clear; clc; close all;

%% ---- Parameters (choose a set that shows Hopf & stable cycles) ----
alpha   = 0.001;
beta    = 0.5;
delta   = 2.0;
epsilon = 2.5;

% scan range for gamma to detect Hopf
g_search = [0.05, 0.8];

%% ---- 1) Locate Hopf: scan gamma and detect max Re(lambda)=0 crossing ----
Ng   = 900;
G    = linspace(g_search(1), g_search(2), Ng);
sigma = nan(1,Ng);         % max Re eigenvalue at interior equilibrium
omegaH_at = nan(1,Ng);     % |Im| of the pair closest to imaginary axis

for k = 1:Ng
    p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',G(k));
    EQ = interior_equilibria(p);
    if isempty(EQ), continue; end
    % pick the interior eq with smaller z (typically the first to Hopf)
    [~,ix] = min([EQ.z]); E = EQ(ix);
    J = jacobian_at(E.x,E.y,E.z,p);
    ev = eig(J);
    sigma(k) = max(real(ev));
    % take the complex pair with smallest |Re| for omega estimate
    [~,jmin] = min(abs(real(ev)));
    omegaH_at(k) = abs(imag(ev(jmin)));
end

% linear locate where sigma crosses zero
gamma_H = crossing_points(G, sigma);
assert(~isempty(gamma_H), 'No Hopf crossing detected. Adjust g_search/parameters.');
gamma_H = gamma_H(1);
% estimate omega_H at the index nearest gamma_H
[~,iH]  = min(abs(G - gamma_H));
omega_H = omegaH_at(iH);
T_pred  = 2*pi/omega_H;  % small-amplitude analytic period
fprintf('Detected Hopf near gamma_H=%.6f with omega_H=%.6f -> T_pred=%.6f\n', gamma_H, omega_H, T_pred);

%% ---- 2) Simulate a persistent limit cycle just beyond Hopf ----
% pick a gamma slightly above Hopf (supercritical side)
dG      = 0.02*(g_search(2)-g_search(1));
gamma_c = min(g_search(2), gamma_H + 0.4*dG);

% equilibrium at gamma_c (for near-IC)
pc  = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gamma_c);
EQc = interior_equilibria(pc);  [~,ix] = min([EQc.z]); Ec = EQc(ix);

% integrate from a small perturbation of equilibrium
ICnear = [Ec.x; Ec.y; Ec.z] .* (1 + 1e-3*[1;-1;1]);
Tmax   = 400;
opts   = odeset('RelTol',1e-9,'AbsTol',1e-12,'MaxStep',0.5);
[t, X] = ode45(@(t,x) igp_rhs(t,x,pc), [0 Tmax], max(ICnear,1e-12), opts);

% discard transient and measure period & amplitudes
fracTransient = 0.5;
idx = t >= t(1)+fracTransient*(t(end)-t(1));
[u,v,w] = deal(X(:,1),X(:,2),X(:,3));
[Per_meas, Amp_u, Amp_v, Amp_w] = measure_period_amp(t(idx), u(idx), v(idx), w(idx));
Amp_max = max([Amp_u, Amp_v, Amp_w]);
fprintf('Measured at gamma=%.6f: T_meas=%.6f (rel. err vs T_pred = %.2g), amplitudes [Au,Av,Aw]=[%.3g,%.3g,%.3g]\n', ...
        gamma_c, Per_meas, abs(Per_meas-T_pred)/T_pred, Amp_u, Amp_v, Amp_w);

%% ---- Plots: time series & 3D phase portrait of the cycle ----
figure('Color','w','Name','Limit cycle time series');
plot(t, u, 'LineWidth',1.6); hold on;
plot(t, v, 'LineWidth',1.6);
plot(t, w, 'LineWidth',1.6);
xlabel('t'); ylabel('states'); grid on; box on;
title(sprintf('Persistent cycle at \\gamma=%.4f  (T_{pred}=%.3g, T_{meas}=%.3g)', gamma_c, T_pred, Per_meas));
legend({'u(t)','v(t)','w(t)'},'Location','best');

figure('Color','w','Name','Limit cycle phase portrait');
plot3(u(idx), v(idx), w(idx), 'LineWidth',1.8); grid on; box on; hold on;
plot3(u(find(idx,1,'first')), v(find(idx,1,'first')), w(find(idx,1,'first')), 'o', ...
        'MarkerFaceColor',[.2 .2 .2],'MarkerEdgeColor','k');
xlabel('u'); ylabel('v'); zlabel('w');
title('Phase portrait (post-transient)');

%% ---- 3) Verify supercritical scaling: A ~ K * sqrt(gamma - gamma_H) ----
% sample several gammas just above Hopf, integrate & measure amplitude A
mu_list = linspace(0.02, 0.12, 7) * (g_search(2)-g_search(1));  % small offsets
G_test  = gamma_H + mu_list;
A_meas  = nan(size(G_test));

for i = 1:numel(G_test)
    gi = G_test(i); if gi >= g_search(2), continue; end
    p  = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gi);
    EQ = interior_equilibria(p); if isempty(EQ), continue; end
    [~,ix] = min([EQ.z]); E = EQ(ix);
    ICn = [E.x;E.y;E.z].*(1+1e-3*[1;-1;1]);
    [~, Xg] = ode45(@(t,x) igp_rhs(t,x,p), [0 Tmax], max(ICn,1e-12), opts);
    idxg = round(numel(Xg)*fracTransient):numel(Xg);
    Ag_u = max(Xg(idxg,1))-min(Xg(idxg,1));
    Ag_v = max(Xg(idxg,2))-min(Xg(idxg,2));
    Ag_w = max(Xg(idxg,3))-min(Xg(idxg,3));
    A_meas(i) = max([Ag_u,Ag_v,Ag_w]);
end

% Fit A ~ K * sqrt(mu) (ignore zeros/NaNs)
mu = G_test - gamma_H;
mask = isfinite(A_meas) & (A_meas>0) & (mu>0);
K = NaN; R2 = NaN;
if any(mask)
    Xfit = sqrt(mu(mask));
    Xfit = Xfit(:);
    yfit = A_meas(mask);
    yfit = yfit(:);
    P = polyfit(Xfit, yfit, 1);
    yhat = polyval(P, Xfit);
    SSres = sum((yfit - yhat).^2); SStot = sum((yfit - mean(yfit)).^2);
    R2 = 1 - SSres/SStot;  K = P(1);
end

figure('Color','w','Name','Amplitude scaling near Hopf');
subplot(1,2,1); hold on; grid on; box on;
plot(G_test, A_meas, 'o-', 'LineWidth',1.5);
xline(gamma_H,'--k','LineWidth',1.2,'Label','\gamma_H');
xlabel('\gamma'); ylabel('Amplitude A (max peak-to-peak)');
title('Amplitude vs \gamma (near Hopf)');

subplot(1,2,2); hold on; grid on; box on;
if any(mask)
    plot(sqrt(mu(mask)), A_meas(mask), 'o', 'LineWidth',1.5, 'DisplayName','data');
    xq = linspace(min(sqrt(mu(mask))), max(sqrt(mu(mask))), 100);
    plot(xq, polyval([K 0], xq), '-', 'LineWidth',1.5, 'DisplayName',sprintf('fit: A = %.3g \\sqrt{\\mu}, R^2=%.3f',K,R2));
else
    plot(0,0,'.'); % placeholder
end
xlabel('\sqrt{\mu} = \sqrt{\gamma-\gamma_H}'); ylabel('Amplitude A');
title('Supercritical scaling check'); legend('Location','best');

%% ==================== helpers ====================

function EQ = interior_equilibria(p)
    % Solve c1 z^2 + c2 z + c3 = 0; keep positive z; back out x,y>0
    c1 = (p.gamma - p.epsilon)/p.alpha + p.delta;
    c2 = -(p.beta*p.delta + p.epsilon);
    c3 =  p.beta*p.gamma;
    r = roots([c1 c2 c3]);
    zc = r(abs(imag(r))<1e-12);
    zc = real(zc); zc = zc(zc > 0);
    EQ = struct('x',{},'y',{},'z',{});
    for i = 1:numel(zc)
        z = zc(i);
        den = z^2 + p.alpha*p.beta;
        x = (p.alpha*z*(p.beta - z))/den;
        y = (z*(p.alpha + z))/den;
        if x>0 && y>0
            EQ(end+1) = struct('x',x,'y',y,'z',z); %#ok<AGROW>
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

function [Tper, Au, Av, Aw] = measure_period_amp(t, u, v, w)
    % pick the signal with largest amplitude to estimate period
    Au = max(u)-min(u); Av = max(v)-min(v); Aw = max(w)-min(w);
    [~,which] = max([Au,Av,Aw]); sigs = {u,v,w}; s = sigs{which};
    Tper = estimate_period(t, s);
end

function Tper = estimate_period(t, s)
    s = s - mean(s);
    n = numel(s);
    if n < 6, Tper = NaN; return; end
    % peaks method
    idx = 2:n-1;
    pk  = idx( s(2:end-1) > s(1:end-2) & s(2:end-1) >= s(3:end) );
    if ~isempty(pk)
        promin = 0.15*max(abs(s));
        pk = pk(s(pk) >= promin);
    end
    if numel(pk) >= 3
        k = min(6, numel(pk)-1);
        Tper = median(diff(t(pk(end-k+1:end))));
        if Tper > 0, return; end
    end
    % autocorrelation fallback
    dt = mean(diff(t));
    ac = xcorr(s, 'coeff');
    mid = (numel(ac)+1)/2;
    acp = ac(mid+1:end);
    L = numel(acp);
    if L >= 5
        jj = 2:L-1;
        locMax = jj( acp(jj) > acp(jj-1) & acp(jj) >= acp(jj+1) & acp(jj) > 0 );
        if ~isempty(locMax)
            lag = locMax(1); Tper = lag*dt; if Tper > 0, return; end
        end
    end
    % FFT fallback
    S = abs(fft(s)); S(1)=0;
    f = (0:n-1)/(n*dt);
    [~,imx] = max(S(1:floor(n/2)));
    fmax = f(imx);
    if fmax > 0, Tper = 1/fmax; else, Tper = NaN; end
end

function gxs = crossing_points(G, val)
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