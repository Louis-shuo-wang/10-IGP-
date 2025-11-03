%% Leslie–Gower IGP: simplified gamma sweep + 2D scans (only classes 0/1/2)
%% dx/dt = x(1 - x/z) - x y
%% dy/dt = alpha*y(1 - beta*y/z) + x y
%% dz/dt = z*(gamma - delta*x - epsilon*y)
%
% Classes in maps:
% 0: predator-only
% 1: one positive equilibrium
% 2: two positive equilibria

clear; clc; close all;

%% ---------------- Baseline parameters ----------------
alpha   = 1.0;
beta    = 2.0;
delta   = 0.5;
epsilon = 0.2;

%% ---------------- 1) Two-parameter map (gamma, epsilon) ----------------
Ng2 = 100; Ne2 = 100;               % tune for accuracy vs runtime
G2 = linspace(0.01, 1.2, Ng2);
E2 = linspace(0.02, 0.8,  Ne2);

tolStable = 1e-6;

classGE  = zeros(Ne2, Ng2);

parfor ie = 1:Ne2
    eps_i = E2(ie);
    for ig = 1:Ng2
        gam_i = G2(ig);
        classGE(ie, ig) = classify_point_simple(alpha, beta, delta, eps_i, gam_i);
    end
end

f2 = figure('Name','2D map: $(\gamma, \epsilon)$','Color','w','Position',[100,100,900,450]);
imagesc(G2, E2, classGE);
set(gca,'YDir','normal'); axis tight; grid on; box on;
xlabel('$\gamma$'); ylabel('$\epsilon$');
title('$(\gamma , \epsilon)$ regime map (0: predator-only, 1: one positive eq., 2: two positive eq.)');
cb2 = colorbar; 
cb2.Orientation = 'vertical';
colormap(simple_cmap());
set(gca,'CLim',[-0.5 2.5]);
set_colorbar_labels_simple(cb2);
drawnow;
exportgraphics(gcf,'Fig2a.svg','Resolution',600);

%% ---------------- 2) Two-parameter map (gamma, beta) ----------------
Ng3 = 100; Nb3 = 100;
G3 = linspace(0.01, 1.2, Ng3);
B3 = linspace(0.2,  3.0, Nb3);

classGB  = zeros(Nb3, Ng3);

parfor ib = 1:Nb3
    beta_i = B3(ib);
    for ig = 1:Ng3
        gam_i = G3(ig);
        classGB(ib, ig) = classify_point_simple(alpha, beta_i, delta, epsilon, gam_i);
    end
end

f3 = figure('Name','2D map: $(\gamma, \beta)$','Color','w', 'Position',[100,100,900,450]);
imagesc(G3, B3, classGB);
set(gca,'YDir','normal'); axis tight; grid on; box on;
xlabel('$\gamma$'); ylabel('$\beta$');
title('$(\gamma , \beta)$ regime map (0: predator-only, 1: one positive eq., 2: two positive eq.)');
cb3 = colorbar; 
cb3.Orientation = 'vertical';
colormap(simple_cmap());
set(gca,'CLim',[-0.5 2.5]);
set_colorbar_labels_simple(cb3);
drawnow;
exportgraphics(gcf,'Fig2b.svg','Resolution',600);

%% ==================== helper functions ====================

function cmap = simple_cmap()
    % 0 gray, 1 blue, 2 orange
    cmap = [0.65 0.65 0.65;   % 0 predator-only
            0.15 0.45 0.95;   % 1 one positive equilibrium
            0.90 0.45 0.10];  % 2 two positive equilibria
end

function set_colorbar_labels_simple(cb)
    cb.Ticks = 0:2;
    cb.TickLabels = { ...
        '0: pre.-only', ...
        '1: one eq. >0', ...
        '2: two eq. >0'};
end

function cls = classify_point_simple(alpha, beta, delta, epsilon, gamma)
    % Simplified classifier:
    % 0: no interior equilibrium (predator-only)
    % 1: exactly one positive interior equilibrium
    % 2: two or more positive interior equilibria
    p = struct('alpha',alpha,'beta',beta,'delta',delta,'epsilon',epsilon,'gamma',gamma);
    EQ = interior_equilibria(p);
    nInt = numel(EQ);
    if nInt == 0
        cls = 0;
    elseif nInt == 1
        cls = 1;
    else
        cls = 2;
    end
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
