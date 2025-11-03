%% Leslie–Gower IGP: 1D gamma sweep + 2D scans with oscillation confirmation + vector export
%   dx/dt = x(1 - x/z) - x y
%   dy/dt = alpha*y(1 - beta*y/z) + x y
%   dz/dt = z*(gamma - delta*x - epsilon*y)
%
% Classes in maps:
% 0: predator-only (P2) stable (no stable interior)
% 1: unique stable coexistence
% 2: two interior equilibria (at least one stable) -- SN region
% 3: interior exists but unstable (Hopf candidate)
% 4: bistability (stable coexistence & P2 both stable)
% 5: CONFIRMED limit cycle by integration

clear; clc;

%% ---------------- Baseline parameters ----------------
alpha   = 1.0;
beta    = 1.0;
delta   = 2.0;
epsilon = 0.10;

%% ---------------- 1) One-parameter sweep in gamma ----------------
gmin = 1e-3; gmax = 1.5; Ng = 600;
G = linspace(gmin, gmax, Ng);
tolStable = 1e-6;

x1 = nan(1,Ng); y1 = nan(1,Ng); z1 = nan(1,Ng); stab1 = false(1,Ng);
x2 = nan(1,Ng); y2 = nan(1,Ng); z2 = nan(1,Ng); stab2 = false(1,Ng);

% boundary P2 (predator-only): (0, gamma/epsilon, gamma/epsilon)
yP2 = G./epsilon; zP2 = yP2; xP2 = zeros(size(G));
stabP2 = G > epsilon; % analytic

for k = 1:Ng
    gamma = G(k);
    params = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gamma);
    EQ = interior_equilibria(params);
    if numel(EQ) >= 1
        x1(k)=EQ(1).x; y1(k)=EQ(1).y; z1(k)=EQ(1).z;
        stab1(k) = is_stable(EQ(1), params, tolStable);
    end
    if numel(EQ) >= 2
        x2(k)=EQ(2).x; y2(k)=EQ(2).y; z2(k)=EQ(2).z;
        stab2(k) = is_stable(EQ(2), params, tolStable);
    end
end

% analytic saddle-node threshold for reference
gamma_SN = 0.5*alpha*( (epsilon/alpha) - delta + sqrt( (epsilon/alpha - delta)^2 + ((beta*delta+epsilon)^2)/(alpha*beta) ) );

% ---- Plot x* vs gamma with stability
f1 = figure('Name','1D sweep: x* vs gamma','Color','w'); hold on; grid on; box on;
plot_branch(G, x1, stab1, [0 0.45 0.75], 1.8);
plot_branch(G, x2, stab2, [0.85 0.33 0.10], 1.8);
plot(G, zeros(size(G)), ':', 'Color', [0 0 0 0.25], 'LineWidth', 1.0); % x=0 baseline
xline(gamma_SN, '--k', 'LineWidth',1.2, 'Label','\gamma_{SN}', 'LabelVerticalAlignment','middle');
xline(epsilon, ':k', 'LineWidth',1.0, 'Label','\epsilon', 'LabelVerticalAlignment','middle');
xlabel('\gamma'); ylabel('x^*  (IG-prey at equilibrium)');
legend({'branch 1 (stable)','branch 1 (unstable)','branch 2 (stable)','branch 2 (unstable)','x=0','\gamma_{SN}','\epsilon'}, ...
        'Location','bestoutside');
title(sprintf('1D sweep in \\gamma (\\alpha=%.2g, \\beta=%.2g, \\delta=%.2g, \\epsilon=%.2g)',alpha,beta,delta,epsilon));

% rough Hopf markers (largest real part ~ 0)
[gh1, gh2] = hopf_markers(G, x1, y1, z1, alpha, beta, delta, epsilon);
if ~isempty(gh1), xline(gh1, '-', 'Color',[0.2 0.6 0.2], 'LineWidth',1.4, 'Label','Hopf (br.1)'); end
if ~isempty(gh2), xline(gh2, '-', 'Color',[0.55 0.75 0.2], 'LineWidth',1.4, 'Label','Hopf (br.2)'); end

%% ---------------- 2) Two-parameter map (gamma, epsilon) ----------------
Ng2 = 80; Ne2 = 80;               % keep moderate so confirmation runs comfortably
G2 = linspace(0.01, 1.2, Ng2);
E2 = linspace(0.02, 0.8, Ne2);

confirmOsc = true;                % << turn ON integration-based confirmation
mapTmax    = 220;                 % integration horizon for confirmation
mapTrans   = 0.5;                 % fraction to discard as transient
ampThresh  = 1e-3;                % amplitude threshold for cycle confirmation
IC         = [0.15; 0.12; 0.18];  % default initial condition for confirmation runs

classGE = zeros(Ne2, Ng2);
for ie = 1:Ne2
    eps_i = E2(ie);
    for ig = 1:Ng2
        gam_i = G2(ig);
        classGE(ie, ig) = classify_point(alpha, beta, delta, eps_i, gam_i, tolStable);
        if confirmOsc && classGE(ie, ig) == 3
            p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',eps_i,'gamma',gam_i);
            isLC = confirm_limit_cycle(p, IC, mapTmax, mapTrans, ampThresh);
            if isLC
                classGE(ie, ig) = 5; % upgrade to confirmed limit cycle
            end
        end
    end
end

f2 = figure('Name','2D map: (gamma, epsilon)','Color','w');
imagesc(G2, E2, classGE);
set(gca,'YDir','normal'); axis tight; grid on; box on;
xlabel('\gamma'); ylabel('\epsilon');
title('( \gamma , \epsilon ) regime map   (with limit-cycle confirmation)');
cb2 = colorbar; cb2.Label.String = 'class';
colormap(class_cmap());
set(gca,'CLim',[-0.5 5.5]);
set_colorbar_labels(cb2);

%% ---------------- 3) Two-parameter map (gamma, beta) ----------------
Ng3 = 80; Nb3 = 80;
G3 = linspace(0.01, 1.2, Ng3);
B3 = linspace(0.2,  3.0, Nb3);

classGB = zeros(Nb3, Ng3);
for ib = 1:Nb3
    beta_i = B3(ib);
    for ig = 1:Ng3
        gam_i = G3(ig);
        classGB(ib, ig) = classify_point(alpha, beta_i, delta, epsilon, gam_i, tolStable);
        if confirmOsc && classGB(ib, ig) == 3
            p = struct('alpha',alpha,'beta',beta_i,'delta',delta,'epsilon',epsilon,'gamma',gam_i);
            isLC = confirm_limit_cycle(p, IC, mapTmax, mapTrans, ampThresh);
            if isLC
                classGB(ib, ig) = 5;
            end
        end
    end
end

f3 = figure('Name','2D map: (gamma, beta)','Color','w');
imagesc(G3, B3, classGB);
set(gca,'YDir','normal'); axis tight; grid on; box on;
xlabel('\gamma'); ylabel('\beta');
title('( \gamma , \beta ) regime map   (with limit-cycle confirmation)');
cb3 = colorbar; cb3.Label.String = 'class';
colormap(class_cmap());
set(gca,'CLim',[-0.5 5.5]);
set_colorbar_labels(cb3);

%% ---------------- 4) Vector export ----------------
outdir = 'figs'; if ~exist(outdir,'dir'), mkdir(outdir); end
save_vec(f1, fullfile(outdir,'bifurcation_x_vs_gamma.pdf'));
save_vec(f2, fullfile(outdir,'map_gamma_epsilon.pdf'));
save_vec(f3, fullfile(outdir,'map_gamma_beta.pdf'));
fprintf('Saved vector PDFs to folder: %s\n', outdir);

%% ==================== helper functions ====================

function save_vec(figH, filename)
    % Save figure as vector PDF (fallback to print if exportgraphics missing)
    try
        exportgraphics(figH, filename, 'ContentType','vector');
    catch
        set(figH,'PaperPositionMode','auto');
        print(figH, filename, '-dpdf', '-painters');
    end
end

function plot_branch(G, val, stab, col, lw)
    % Stable: solid; Unstable: dashed; handles NaNs
    isn = isnan(val);
    ed = find(diff([true, ~isn, true]) ~= 0);
    starts = ed(1:2:end-1); stops  = ed(2:2:end)-1;
    for k=1:numel(starts)
        idx = starts(k):stops(k);
        Gi = G(idx); Vi = val(idx); Si = stab(idx);
        segSt  = find_runs(Si);
        for r=1:size(segSt,1)
            ii = segSt(r,1):segSt(r,2);
            plot(Gi(ii), Vi(ii), '-', 'Color', col, 'LineWidth', lw);
        end
        segUn = find_runs(~Si);
        for r=1:size(segUn,1)
            ii = segUn(r,1):segUn(r,2);
            plot(Gi(ii), Vi(ii), '--', 'Color', col, 'LineWidth', lw);
        end
    end
end

function rr = find_runs(mask)
    if isempty(mask), rr = zeros(0,2); return; end
    d = diff([false, mask(:).', false]);
    s = find(d==1); e = find(d==-1)-1;
    rr = [s(:), e(:)];
end

function cmap = class_cmap()
    % 0 gray, 1 blue, 2 orange, 3 magenta, 4 green, 5 red
    cmap = [0.65 0.65 0.65;   % 0 predator-only
            0.15 0.45 0.95;   % 1 stable coexistence
            0.90 0.45 0.10;   % 2 two interior (SN side)
            0.75 0.15 0.75;   % 3 oscillation candidate
            0.25 0.70 0.30;   % 4 bistability
            0.90 0.15 0.10];  % 5 confirmed limit cycle
end

function set_colorbar_labels(cb)
    cb.Ticks = 0:5;
    cb.TickLabels = { ...
        '0: predator-only', ...
        '1: stable coexistence', ...
        '2: two interior (SN)', ...
        '3: oscillation candidate', ...
        '4: bistability', ...
        '5: confirmed limit cycle'};
end

function EQ = interior_equilibria(p)
    % Solve c1 z^2 + c2 z + c3 = 0, keep positive z, then x,y>0
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
    if ~isempty(EQ)
        [~,ix] = sort([EQ.z]); EQ = EQ(ix);
    end
end

function J = jacobian_at(x, y, z, p)
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

function tf = is_stable(E, p, tol)
    ev = eig(jacobian_at(E.x, E.y, E.z, p));
    tf = max(real(ev)) < -tol;
end

function [gh1, gh2] = hopf_markers(G, x1, y1, z1, alpha, beta, delta, epsilon)
    % crude detection on branch 1 (extend similarly for branch 2 if needed)
    gh1 = []; gh2 = [];
    if all(isnan(x1)), return; end
    sig = nan(size(G));
    for k=1:numel(G)
        if ~isnan(x1(k))
            p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',G(k));
            ev = eig(jacobian_at(x1(k), y1(k), z1(k), p));
            sig(k) = max(real(ev));
        end
    end
    gh1 = crossing_points(G, sig);
end

function gxs = crossing_points(G, val)
    gxs = [];
    for k=1:numel(G)-1
        a = val(k); b = val(k+1);
        if any(isnan([a b])), continue; end
        if a==0, gxs(end+1) = G(k); 
        elseif b==0, gxs(end+1) = G(k+1); 
        elseif a*b < 0
            t = abs(a) / (abs(a)+abs(b));
            gxs(end+1) = (1-t)*G(k) + t*G(k+1); 
        end
    end
end

function cls = classify_point(alpha, beta, delta, epsilon, gamma, tolStable)
    % 0 P2 stable; 1 unique stable coexistence; 2 two interior (some stable)
    % 3 interior unstable (Hopf candidate); 4 bistability; 5 is assigned only after confirmation
    p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gamma);
    P2_stable = gamma > epsilon;
    EQ = interior_equilibria(p);
    nInt = numel(EQ);
    if nInt == 0
        cls = P2_stable * 0 + (~P2_stable) * 0;
        return;
    end
    isSt = false(1,nInt);
    isUn = false(1,nInt);
    for i=1:nInt
        ev = eig(jacobian_at(EQ(i).x, EQ(i).y, EQ(i).z, p));
        mx = max(real(ev));
        isSt(i) = (mx < -tolStable);
        isUn(i) = (mx >  tolStable);
    end
    hasStable = any(isSt);
    hasUnstable = any(isUn);

    if hasStable && ~P2_stable
        cls = (nInt==1)*1 + (nInt>=2)*2;
        return;
    end
    if hasStable && P2_stable
        cls = 4; % coexistence & P2 both stable
        return;
    end
    if ~hasStable && hasUnstable
        cls = 3; % oscillation candidate
        return;
    end
    cls = P2_stable * 0 + (~P2_stable) * 1;
end

function isLC = confirm_limit_cycle(p, x0, T, fracTransient, ampThresh)
    % Integrate and test for sustained oscillations (simple amplitude test after transient)
    rhs  = @(t,X) igp_rhs(t,X,p);
    opts = odeset('RelTol',1e-8,'AbsTol',1e-11,'MaxStep',0.5);
    try
        [t, X] = ode45(rhs, [0 T], enforce_pos(x0), opts);
    catch
        % fallback small steps if needed
        opts = odeset(opts,'MaxStep',0.1);
        [t, X] = ode45(rhs, [0 T], enforce_pos(x0), opts);
    end
    % discard transient
    t0 = t(1); t1 = t(end);
    tcut = t0 + fracTransient*(t1 - t0);
    idx  = find(t >= tcut);
    if numel(idx) < 10
        isLC = false; return;
    end
    Xs = X(idx,1); Ys = X(idx,2); Zs = X(idx,3);
    ampX = max(Xs) - min(Xs);
    ampY = max(Ys) - min(Ys);
    ampZ = max(Zs) - min(Zs);
    amp  = max([ampX ampY ampZ]);
    isLC = amp > ampThresh;
end

function Xp = enforce_pos(x)
    Xp = max(x, 1e-12);
end

function dx = igp_rhs(~, X, p)
    x = max(X(1),0); y = max(X(2),0); z = max(X(3),0);
    if z < 1e-14, z = 1e-14; end
    dx = zeros(3,1);
    dx(1) = x*(1 - x/z) - x*y;
    dx(2) = p.alpha*y*(1 - p.beta*y/z) + x*y;
    dx(3) = z*(p.gamma - p.delta*x - p.epsilon*y);
end