function continue_periodic_orbits()
% CONTINUE_PERIODIC_ORBITS
% Continuation for (A) 2-cycles (post-flip) and (B) general n-cycles (locked orbits).
%
% Run: continue_periodic_orbits

% ----- parameters -----
par.alpha   = 0.003;
par.beta    = 2.0;
par.delta   = 0.5;
par.epsilon = 0.2;
dt = 0.1;

% ----- NEWTON/continuation options -----
newton.maxit   = 25;
newton.tol     = 1e-12;
newton.verbose = false;

ds     = 2.5e-3;  % arclength step
Nsteps = 200;

% ====== A) PERIOD-2 continuation ======
fprintf('\n== Period-2 continuation (post-flip) ==\n');
% initial gamma and seed 2-cycle (two points close but not equal)
gamma0 = 0.295;
X1_0   = [0.31; 0.18; 0.43];
X2_0   = [0.30; 0.19; 0.44];

[branch2, eig2] = continue_period2(X1_0, X2_0, gamma0, par, dt, ds, Nsteps, newton);

% simple plot: x-component of first point vs gamma
figure;
plot(branch2.gamma, squeeze(branch2.X(1,1,:)), 'k.-', ...
     branch2.gamma, squeeze(branch2.X(1,2,:)), 'r.-');
grid on
xlabel('\gamma'); ylabel('x_1');
title('Period-2 branch');
legend('point 1','point 2');
drawnow

% ====== B) GENERAL n-cycle continuation (e.g., n=5) ======
fprintf('\n== n-cycle continuation (here n=5) ==\n');
n = 5;
gamma0 = 0.315;                 % choose inside a resonance tongue / near NS
Xseed  = repmat([0.30;0.19;0.44], 1, n);  % crude initial polygon
% small perturbation around a circle:
for k=1:n
    Xseed(:,k) = Xseed(:,k) + 1e-3*[cos(2*pi*k/n); sin(2*pi*k/n); 0.0];
end

[branchn, eign] = continue_cycle_n(Xseed, gamma0, par, dt, ds, Nsteps, newton);

figure; 
plot(branchn.gamma, squeeze(branchn.X(1,1,:)), 'b.-'); grid on
xlabel('\gamma'); ylabel('x_1^{(1)}');
title(sprintf('Period-%d branch (first point)',n));

end

% ==================== PERIOD-2 ====================

function [branch, eigs_store] = continue_period2(X1_0, X2_0, gamma0, par, dt, ds, N, newton)
U0 = [X1_0; X2_0; gamma0];
% correct initial to satisfy equations
[U0, ok] = newton_corr_P2(U0, par, dt, newton);
assert(ok, 'Initial P2 corrector failed.');

% take a small natural-parameter step to form a secant
gamma1 = gamma0 + 1e-3;
U1 = U0; U1(end) = gamma1;
[U1, ok] = newton_corr_P2(U1, par, dt, newton);
assert(ok,'Second P2 corrector failed.');

tsec = (U1-U0)/norm(U1-U0);
U_prevprev = U0; U_prev = U1;

% storage
branch.X = nan(3,2,N+2);
branch.gamma = nan(1,N+2);
eigs_store = cell(1,N+2);

branch.X(:,:,1) = [U0(1:3), U0(4:6)];
branch.gamma(1) = U0(7);
eigs_store{1}   = p2_eigs(U0(1:3),U0(4:6),U0(7),par,dt);

branch.X(:,:,2) = [U1(1:3), U1(4:6)];
branch.gamma(2) = U1(7);
eigs_store{2}   = p2_eigs(U1(1:3),U1(4:6),U1(7),par,dt);

for k=3:N+2
    Upred = U_prev + ds*tsec;
    [Ucorr, ok] = newton_corr_P2(Upred, par, dt, newton, tsec, Upred);
    if ~ok
        % try half-step
        Upred = U_prev + 0.5*ds*tsec;
        [Ucorr, ok] = newton_corr_P2(Upred, par, dt, newton, tsec, Upred);
        if ~ok, break; end
    end
    U_prevprev = U_prev; U_prev = Ucorr;
    tsec = (U_prev-U_prevprev)/norm(U_prev-U_prevprev);

    X1=Ucorr(1:3); X2=Ucorr(4:6); g=Ucorr(7);
    branch.X(:,:,k) = [X1,X2]; branch.gamma(k) = g;
    eigs_store{k}   = p2_eigs(X1,X2,g,par,dt);
end
end

function lam = p2_eigs(X1,X2,gamma,par,dt)
% Floquet multipliers of 2-cycle: eig(DPhi(X2)*DPhi(X1))
A1 = DPhi(X1, gamma, par, dt);
A2 = DPhi(X2, gamma, par, dt);
lam = eig(A2*A1);
end

function [U, ok] = newton_corr_P2(Uinit, par, dt, newton, tsec, Upred)
% Solve H(U) = [ Phi(X1)-X2 ; Phi(X2)-X1 ; (tsec^T)(U-Upred)=0 ] if secant given,
% otherwise just the 6 equations (natural parameter correction).
X1=Uinit(1:3); X2=Uinit(4:6); g=Uinit(7); U = Uinit;
if nargin<4, error('newton struct required'); end
useArc = (nargin>=6);

for it=1:newton.maxit
    P1 = Phi(X1,g,par,dt) - X2;
    P2 = Phi(X2,g,par,dt) - X1;
    if useArc, H = [P1; P2; tsec.'*(U-Upred)];
    else,      H = [P1; P2];
    end
    if norm(H,2) < newton.tol, ok=true; return; end

    % Jacobian of [P1;P2] wrt [X1;X2;g]
    A1=DPhi(X1,g,par,dt); A2=DPhi(X2,g,par,dt);
    dP1dX1 = A1; dP1dX2 = -eye(3); dP1dg = dPhidg(X1,g,par,dt);
    dP2dX1 = -eye(3);   dP2dX2 = A2; dP2dg = dPhidg(X2,g,par,dt);
    J = [dP1dX1, dP1dX2, dP1dg; dP2dX1, dP2dX2, dP2dg];
    if useArc
        J = [J; tsec.']; %#ok<AGROW>
    end

    DU = -J\H; U = U + DU;
    X1=U(1:3); X2=U(4:6); g=U(7);
end
ok=false;
end

% ==================== GENERAL n-CYCLE ====================

function [branch, eigs_store] = continue_cycle_n(Xs0, gamma0, par, dt, ds, N, newton)
% Xs0: 3-by-n matrix (columns are points X_1..X_n)
n = size(Xs0,2);
U0 = [Xs0(:); gamma0];
[U0, ok] = newton_corr_n(U0, n, par, dt, newton);
assert(ok,'Initial n-cycle corrector failed.');

gamma1 = gamma0 + 1e-3;
U1 = U0; U1(end)=gamma1;
[U1, ok] = newton_corr_n(U1, n, par, dt, newton);
assert(ok,'Second n-cycle corrector failed.');

tsec = (U1-U0)/norm(U1-U0);
U_prevprev=U0; U_prev=U1;

branch.X     = nan(3,n,N+2);
branch.gamma = nan(1,N+2);
eigs_store   = cell(1,N+2);

[Xs, g] = unpack(Unorm(U0), n);
branch.X(:,:,1)=Xs; branch.gamma(1)=g; eigs_store{1}=ncycle_eigs(Xs,g,par,dt);
[Xs, g] = unpack(Unorm(U1), n);
branch.X(:,:,2)=Xs; branch.gamma(2)=g; eigs_store{2}=ncycle_eigs(Xs,g,par,dt);

for k=3:N+2
    Upred = U_prev + ds*tsec;
    [Ucorr, ok] = newton_corr_n(Upred, n, par, dt, newton, tsec, Upred);
    if ~ok
        Upred = U_prev + 0.5*ds*tsec;
        [Ucorr, ok] = newton_corr_n(Upred, n, par, dt, newton, tsec, Upred);
        if ~ok, break; end
    end
    U_prevprev=U_prev; U_prev=Ucorr; tsec=(U_prev-U_prevprev)/norm(U_prev-U_prevprev);
    [Xs,g]=unpack(Ucorr,n);
    branch.X(:,:,k)=Xs; branch.gamma(k)=g; eigs_store{k}=ncycle_eigs(Xs,g,par,dt);
end
end

function lam = ncycle_eigs(Xs, gamma, par, dt)
% Monodromy multipliers: eigenvalues of A_n ... A_1, where A_k = D Phi(X_k)
n=size(Xs,2); M=eye(3);
for k=1:n, M = DPhi(Xs(:,k), gamma, par, dt) * M; end
lam = eig(M);
end

function [U, ok] = newton_corr_n(Uinit, n, par, dt, newton, tsec, Upred)
% Solve R(U) = [Phi(X1)-X2; ...; Phi(Xn)-X1; (tsec^T)(U-Upred)=0]
U = Uinit; useArc=(nargin>=6);
for it=1:newton.maxit
    [R, JR] = R_and_J(U, n, par, dt);
    if useArc, H=[R; tsec.'*(U-Upred)]; J=[JR; tsec.'];
    else,      H=R; J=JR;
    end
    if norm(H,2) < newton.tol, ok=true; return; end
    U = U - J\H;
end
ok=false;
end

function [R, JR] = R_and_J(U, n, par, dt)
[Xs, g]=unpack(U,n); R=zeros(3*n,1); JR=zeros(3*n,3*n+1);
for k=1:n
    kp = mod(k,n)+1;  % next index (wrap)
    Xk = Xs(:,k); Xkp = Xs(:,kp);
    Pk = Phi(Xk,g,par,dt) - Xkp;  R(3*(k-1)+(1:3)) = Pk;

    A  = DPhi(Xk,g,par,dt); dPg = dPhidg(Xk,g,par,dt);
    row = 3*(k-1)+(1:3);
    colk= 3*(k-1)+(1:3);
    colp= 3*(kp-1)+(1:3);
    JR(row,colk) = A;           % dPk/dXk
    JR(row,colp) = -eye(3);     % dPk/dX_{k+1}
    JR(row,3*n+1)= dPg;         % dPk/dg
end
end

function [Xs,g] = unpack(U,n)
Xs = reshape(U(1:3*n),3,n); g = U(3*n+1);
end

function U = Unorm(U), U = U(:); end

% ==================== map & derivatives ====================

function Y = Phi(X,gamma,par,dt)
[f1,f2,f3] = f_rhs(X,gamma,par);
Y = X + dt*[f1;f2;f3];
end

function A=DPhi(X,gamma,par,dt)
A = eye(3)+dt*J_f(X,gamma,par);
end

function Pg = dPhidg(X,gamma,par,dt)
% d/dgamma of Phi = dt * d/dgamma f; only z-equation depends on gamma: z*(gamma-...)
Pg = dt * [0;0; X(3)];
end

function [f1,f2,f3]=f_rhs(X,gamma,par)
x=X(1); y=X(2); z=X(3); z=max(z,1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
f1 = x*(1 - x/z) - x*y;
f2 = a*y*(1 - b*y/z) + x*y;
f3 = z*(gamma - d*x - e*y);
end

function J=J_f(X,gamma,par)
x=X(1); y=X(2); z=X(3); z=max(z,1e-12);
a=par.alpha; b=par.beta; d=par.delta; e=par.epsilon;
J = [ (1-2*x/z)-y , -x , (x^2)/(z^2) ;
      y           , a*(1-2*b*y/z)+x , a*b*(y^2)/(z^2) ;
     -d*z         , -e*z            , gamma-d*x-e*y   ];
end
