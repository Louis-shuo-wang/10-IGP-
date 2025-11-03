% Create a plot of Δ(γ) = -4 α β γ^2 + 4 α β (ε - α δ) γ + α^2 (β δ + ε)^2
% with example parameters satisfying βδ > ε, and mark ε (gamma2) and γ1+.

clear; clc; close all;

% Example parameters (βδ > ε)
alpha = 0.8;
beta = 2.0;
delta = 1.0;
epsilon = 0.6; % gamma2

% Discriminant as a function of gamma
Delta = @(gamma, alpha, beta, delta, epsilon) ...
    -4*alpha*beta.*gamma.^2 + 4*alpha*beta*(epsilon - alpha*delta).*gamma + ...
    alpha^2*(beta*delta + epsilon)^2;

% Roots γ1± from the closed form
disc_inside = (epsilon - alpha*delta)^2 + (alpha/beta)*(beta*delta + epsilon)^2;
gamma1_minus = ((epsilon - alpha*delta) - sqrt(disc_inside)) / 2;
gamma1_plus  = ((epsilon - alpha*delta) + sqrt(disc_inside)) / 2;

% Gamma range for plotting
g_min = 0.0;
g_max = max(1.2*gamma1_plus, epsilon*1.8);
g = linspace(g_min, g_max, 400);
Dg = Delta(g, alpha, beta, delta, epsilon);

gammas = [0.6,gamma1_plus,0.8];
y = zeros(1001,3);
for i =1:numel(gammas)
    gamma = gammas(i);
    c1 = (gamma-epsilon)/alpha + delta;
    c2 = -(beta*delta+epsilon);
    c3 = beta*gamma;
    r = linspace(0,2.5,1001);
    f_qua = @(x) c1*x^2 + c2*x + c3;
    y(:,i) = arrayfun(@(x) f_qua(x), r);
end


% Plotting
defaultsetting;
figure('Position', [100, 100, 900, 450]);
colors = lines(3);
tiledlayout(1,2,"TileSpacing",'tight','Padding','tight');

nexttile;
plot(g, Dg, 'DisplayName', '$\Delta(\gamma)$');
hold on;
yline(0, 'k-', 'Displayname', '$\Delta=0$');
xline(epsilon, '--', 'DisplayName', '$\gamma = \epsilon$');
xline(gamma1_plus, ':', 'DisplayName', '$\gamma_1^+$');
hold off;
xlabel('$\gamma$');
ylabel('$\Delta(\gamma)$');
title('(a)');
legend('Location', 'northeast', 'Box', 'off');
grid on;

nexttile;
plot(r,y(:,1),'LineStyle','-','Color',colors(:,1),'DisplayName','$\gamma=0.6$'); hold on;
plot(r,y(:,2),'LineStyle','--','Color',colors(:,2),'DisplayName','$\gamma=\gamma_1^+$');
plot(r,y(:,3),'LineStyle','-','Color',colors(:,3),'DisplayName','$\gamma=0.8$');
yline(0,'r', 'DisplayName','y=0');
xlabel('$R$');
ylabel('$c_1R^2+c_2R+c_3$');
title('(b)');
legend('Location','best');


drawnow;
exportgraphics(gcf,'Fig3.svg', 'Resolution',600);


% Display results
fprintf('\\alpha=%.2f, \\beta=%.2f, \\delta=%.2f, \\epsilon=%.2f', alpha, beta, delta, epsilon);
fprintf('gamma1_minus = %.4f\n', gamma1_minus);
fprintf('gamma1_plus  = %.4f\n', gamma1_plus);
fprintf('epsilon (gamma2) = %.4f\n', epsilon);
