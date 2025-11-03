% find_discrete_bifurcations_fixed.m
clear; close all; 
rng(1);

%% User-configurable parts
useGrid = false;        % true -> do full grid (may be enormous). false -> LHS sampling
gridN = 21;
Nsamples = 200000;
paramRange = [1e-3, 4];
dt_list = linspace(1e-4,2,250);

tol_mu = 1e-4;
tol_other = 1e-4;
tol_lambda = 1e-4;
tol_imag = 1e-4;
gamma_cont_radius = 0.3;
gamma_cont_points = 401;

%% Generate parameter samples
if useGrid
    vec = linspace(paramRange(1), paramRange(2), gridN);
    [A,B,G,D,E] = ndgrid(vec,vec,vec,vec,vec);
    combos = [A(:), B(:), G(:), D(:), E(:)];
else
    lhs = lhsdesign(Nsamples,5,'criterion','maximin','iterations',100);
    combos = paramRange(1) + (paramRange(2)-paramRange(1)) * lhs;
end

nCombos = size(combos,1);
fprintf('Testing %d parameter combos (useGrid=%d)\n', nCombos, useGrid);

%% Containers
found = struct('type',{},'alpha',{},'beta',{},'gamma',{},'delta',{},'epsilon',{},'dt',{},'eq',{},'mu',{});
foundTypes = struct('fold',false,'flip',false,'NS',false);

%% Main loop (compatibility fix: no struct2array)
for idx = 1:nCombos
    alpha = combos(idx,1);
    beta  = combos(idx,2);
    gamma = combos(idx,3);
    delta = combos(idx,4);
    epsilon= combos(idx,5);

    eqs = computeEquilibriaAndJac(alpha,beta,gamma,delta,epsilon);
    if isempty(eqs)
        continue;
    end
    for k = 1:length(eqs)
        J = eqs(k).J;
        lambda = eig(J);
        for dt = dt_list
            mu = 1 + dt * lambda;
            lambda_vals = lambda;
            % Check conditions
            cond_fold = any(abs(real(lambda_vals)) < 0.1 * tol_lambda & abs(imag(lambda_vals)) < 0.1 * tol_imag) ...
             && all(abs(mu(abs(real(lambda_vals)) >= tol_lambda)) < 1 - 1e-1 * tol_other) && beta * delta > epsilon;
            cond_flip = any(abs(mu + 1) < tol_mu) && all(abs(mu(abs(mu+1)>=tol_mu)) < 1 - tol_other);
            cond_NS = false;
            for i=1:2
                for j=i+1:3
                    mu1 = mu(i); mu2 = mu(j);
                    if abs(abs(mu1)-1) < tol_mu && abs(abs(mu2)-1) < tol_mu
                        if abs(mu1 - conj(mu2)) < 1e-6 || (abs(imag(mu1))>1e-6 && abs(mu1 - conj(mu2))<1e-2)
                            ang = angle(mu1);
                            if abs(mod(ang,pi)) > 1e-2
                                other_idx = setdiff([1,2,3],[i,j]);
                                if abs(mu(other_idx)) < 1 - tol_other
                                    cond_NS = true;
                                end
                            end
                        end
                    end
                end
            end

            if cond_fold && ~foundTypes.fold
                foundTypes.fold = true;
                f = struct('type','fold','alpha',alpha,'beta',beta,'gamma',gamma,'delta',delta,'epsilon',epsilon,'dt',dt,'eq',eqs(k),'mu',mu);
                found(end+1) = f; %#ok<AGROW>
                fprintf('Found fold at idx=%d dt=%.4g alpha=%.4g beta=%.4g gamma=%.4g delta=%.4g eps=%.4g\n', idx, dt, alpha,beta,gamma,delta,epsilon);
            end
            if cond_flip && ~foundTypes.flip
                foundTypes.flip = true;
                f = struct('type','flip','alpha',alpha,'beta',beta,'gamma',gamma,'delta',delta,'epsilon',epsilon,'dt',dt,'eq',eqs(k),'mu',mu);
                found(end+1) = f; %#ok<AGROW>
                fprintf('Found flip at idx=%d dt=%.4g alpha=%.4g beta=%.4g gamma=%.4g delta=%.4g eps=%.4g\n', idx, dt, alpha,beta,gamma,delta,epsilon);
            end
            if cond_NS && ~foundTypes.NS
                foundTypes.NS = true;
                f = struct('type','NS','alpha',alpha,'beta',beta,'gamma',gamma,'delta',delta,'epsilon',epsilon,'dt',dt,'eq',eqs(k),'mu',mu);
                found(end+1) = f; %#ok<AGROW>
                fprintf('Found NS at idx=%d dt=%.4g alpha=%.4g beta=%.4g gamma=%.4g delta=%.4g eps=%.4g\n', idx, dt, alpha,beta,gamma,delta,epsilon);
            end

            % COMPATIBILITY FIX: replace struct2array(...) with explicit check
            if foundTypes.fold && foundTypes.flip && foundTypes.NS
                break;
            end
        end
        if foundTypes.fold && foundTypes.flip && foundTypes.NS
            break;
        end
    end
    if foundTypes.fold && foundTypes.flip && foundTypes.NS
        break;
    end
end

if isempty(found)
    warning('No parameter setting found with the current sampling/grid and tolerances.');
else
    fprintf('Summary of found bifurcations:\n');
    for i=1:length(found)
        fprintf('  %s: alpha=%.4g beta=%.4g gamma=%.4g delta=%.4g eps=%.4g dt=%.4g\n', ...
            found(i).type, found(i).alpha, found(i).beta, found(i).gamma, found(i).delta, found(i).epsilon, found(i).dt);
    end
end

%% Local continuation and plotting (unchanged)
defaultsetting;
figure('Position',[100,100,900,1200]);
tiledlayout(3,2,'TileSpacing','tight','Padding','tight')
for i=1:length(found)
    casei = found(i);
    alpha = casei.alpha; beta = casei.beta; delta = casei.delta; epsilon = casei.epsilon;
    gamma0 = casei.gamma;
    gammas = linspace(max(paramRange(1), gamma0 - gamma_cont_radius), gamma0 + gamma_cont_radius, gamma_cont_points);
    Rvals = nan(size(gammas)); Nvals = nan(size(gammas)); Pvals = nan(size(gammas));
    maxMu = nan(size(gammas)); secondMu = nan(size(gammas)); thirdMu = nan(size(gammas));
    prevR = casei.eq.R;
    for k = 1:length(gammas)
        gamma_k = gammas(k);
        eqs_k = computeEquilibriaAndJac(alpha,beta,gamma_k,delta,epsilon);
        if isempty(eqs_k)
            continue;
        end
        Rs = [eqs_k.R];
        [~, idxmin] = min(abs(Rs - prevR));
        chosen = eqs_k(idxmin);
        Rvals(k) = chosen.R;
        Nvals(k) = chosen.N;
        Pvals(k) = chosen.P;
        lam = eig(chosen.J);
        mu = 1 + casei.dt * lam;
        muabs = abs(mu);
        muabs_sorted = sort(muabs,'descend');
        maxMu(k) = muabs_sorted(1);
        secondMu(k) = muabs_sorted(2);
        thirdMu(k) = muabs_sorted(3);
        prevR = chosen.R;
    end

    if i == 1
        nexttile(5);
        plot(gammas, Nvals, '--b'); hold on;
        plot(gammas, Pvals, ':r');
        plot(gammas, Rvals, '-k');
        xlabel('$\gamma$'); 
        ylabel('Equilibrium (N, P, R)');
        legend('$R^*$','$N^*$','$P^*$','Location','best');
        title(sprintf('Local continuation around gamma=%.4g (%s)', gamma0, casei.type));
        grid on;
        nexttile(6);
        plot(gammas, maxMu,'-k'); hold on;
        plot(gammas, secondMu,'--b');
        plot(gammas, thirdMu,':r');
        yline(1,'r--','$|\mu|=1$');
        xlabel('$\gamma$'); ylabel('Multiplier magnitudes $|\mu|$');
        legend('$\max|\mu|$','2nd','3rd','Location','best');
        grid on;
    elseif i == 2
        nexttile(3);
        plot(gammas, Nvals, '--b'); hold on;
        plot(gammas, Pvals, ':r');
        plot(gammas, Rvals, '-k');
        xlabel('$\gamma$'); ylabel('Equilibrium (N, P, R)');
        legend('$R^*$','$N^*$','$P^*$','Location','best');
        title(sprintf('Local continuation around gamma=%.4g (%s)', gamma0, casei.type));
        grid on;
        nexttile(4);
        plot(gammas, maxMu,'-k'); hold on;
        plot(gammas, secondMu,'--b');
        plot(gammas, thirdMu,':r');
        yline(1,'r--','$|\mu|=1$');
        xlabel('$\gamma$'); ylabel('Multiplier magnitudes $|\mu|$');
        legend('$\max|\mu|$','2nd','3rd','Location','best');
        grid on;
    else
        nexttile(1);
        plot(gammas, Nvals, '--b'); hold on;
        plot(gammas, Pvals, ':r');
        plot(gammas, Rvals, '-k');
        xlabel('$\gamma$'); ylabel('Equilibrium (N, P, R)');
        legend('$R^*$','$N^*$','$P^*$','Location','best');
        title(sprintf('Local continuation around gamma=%.4g (%s)', gamma0, casei.type));
        grid on;
        nexttile(2);
        plot(gammas, maxMu,'-k'); hold on;
        plot(gammas, secondMu,'--b');
        plot(gammas, thirdMu,':r');
        yline(1,'r--','$|\mu|=1$');
        xlabel('$\gamma$'); ylabel('Multiplier magnitudes $|\mu|$');
        legend('$\max|\mu|$','2nd','3rd','Location','best');
        grid on;
    end
end
fprintf('Script complete. If nothing was found, try increasing Nsamples or expanding dt_list / tolerances.\n');
drawnow;
exportgraphics(gcf,'Fig4.svg', 'Resolution',600);

%% helper function
function eqs = computeEquilibriaAndJac(alpha,beta,gamma,delta,epsilon)
c1 = (gamma - epsilon)/alpha + delta;
c2 = - (beta*delta + epsilon);
c3 = beta*gamma;
eqs = struct('R',{},'N',{},'P',{},'J',{});
if abs(c1) < 1e-12
    if abs(c2) > 0
        rootsR = -c3/c2;
    else
        rootsR = [];
    end
else
    disc = c2^2 - 4*c1*c3;
    if disc < 0
        rootsR = [];
    else
        r1 = (-c2 + sqrt(disc))/(2*c1);
        r2 = (-c2 - sqrt(disc))/(2*c1);
        rootsR = [r1,r2];
    end
end
for R = rootsR
    if isreal(R) && R > 0
        Nstar = (alpha * R * (beta - R)) / (R^2 + alpha*beta);
        Pstar = (R * (alpha + R)) / (R^2 + alpha*beta);
        if Nstar > 0 && Pstar > 0
            J = zeros(3,3);
            J(1,1) = 1 - 2*Nstar/R - Pstar;
            J(1,2) = -Nstar;
            J(1,3) = (Nstar^2)/(R^2);
            J(2,1) = Pstar;
            J(2,2) = alpha - 2*alpha*beta*Pstar/R + Nstar;
            J(2,3) = (alpha*beta*Pstar^2)/(R^2);
            J(3,1) = -delta*R;
            J(3,2) = -epsilon*R;
            J(3,3) = gamma - delta*Nstar - epsilon*Pstar;
            eqs(end+1).R = R; %#ok<AGROW>
            eqs(end).N = Nstar;
            eqs(end).P = Pstar;
            eqs(end).J = J;
        end
    end
end
end
