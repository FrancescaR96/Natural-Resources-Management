load -ascii Dati.txt
% NOTE: Precipitation in [mm/d], Streamflow in [m^3/s], Temperature in [°C]

Td = 365;          % of days in a period
N = length(Dati); % "length" of data set
Periods = N/Td; %numero anni (27)
time = datetime(1990,1,1) + caldays(0:N-1);

% Datasets loading
p = Dati (:,4); % precipitation vector
q = Dati (:,5); % streamflow vector
t = Dati (:,6); % temperature vector

% Plot inflow along the time series
figure('Name','Inflow dataset')
plot(time, q, 'black')
xlabel('[day]')
ylabel('[m^3/s]')
title('Inflow data')

% Plot precipitation along the time series
figure('Name','Precipiation dataset')
plot(time, p, 'black')
xlabel('[day]')
ylabel('[mm/d]')
title('Precipitation data')

% Plot temperature along the time series
figure('Name','Temperature dataset')
plot(time, t, 'black')
xlabel('[day]')
ylabel('[°C]')
title('Temperature data')

%--------------------------------------------------------------------------
% Basic statistics
q_min   = min(q);
q_max   = max(q);
q_range = q_max - q_min;
q_mean  = mean(q);
q_var   = var(q);

p_min   = min(p);
p_max   = max(p);
p_range = p_max - p_min;
p_mean  = mean(p);
p_var   = var(p);

temp_min   = min(t);
temp_max   = max(t);
temp_range = temp_max - temp_min;
temp_mean  = mean(t);
temp_var   = var(t);

% Periodicity of dataset
tt = repmat((1:365)', N/Td, 1);

figure('Name','Periodicity of Inflow data')
plot(tt, q, 'black.')
xlabel('[day of the year]')
ylabel('[m^3/s]')
title('Periodicity of Inflow data')
xlim([1 365])

figure('Name','Periodicity of Precipitation data')
plot(tt, p, 'black.')
xlabel('[day of the year]')
ylabel('[mm/d]')
title('Periodicity of Precipitation data')
xlim([1 365])

figure('Name','Periodicity of Temperature data')
plot(tt, t, 'black.')
xlabel('[day of the year]')
ylabel('[°C]')
title('Periodicity of Temperature data')
xlim([1 365])

% reshape the vector n containing the inflow data
Q = reshape(q, Td, 27);
% cyclo-stationary mean
Cm = mean(Q, 2); % Cm = mean(Q')';
% cyclo-stationary variance
Cv = var(Q, 0, 2); % Cv = var(Q')';

figure
plot(tt, q, 'black.')
hold on
plot(Cm, 'r', 'LineWidth', 2)
legend('Inflow data', 'Cyclost. mean')
xlabel('[day of the year]')
ylabel('[m^3/s]')
title('Periodicity of inflow with Cyclostationary mean')
xlim([1 365])

%--------------------------------------------------------------------------
% moving averages and moving variances
[mi_q, m_q] = moving_average(q, Td, 5);
[~, s2_q]   = moving_average( (q-m_q).^2, Td, 5);
s_q = sqrt(s2_q);

[~, m_p]  = moving_average(p, Td, 5);
[~, s2_p] = moving_average( (p-m_p).^2, Td, 5);
s_p = sqrt(s2_p);

[~, m_t]  = moving_average(t, Td, 5);
[~, s2_t] = moving_average( (t-m_t).^2, Td, 5);
s_t = sqrt(s2_t);

% Deseasonalization
x   = (q-m_q) ./ s_q;
u_p = (p-m_p) ./ s_p;
u_t = (t-m_t) ./ s_t;

% Series of plots
figure
plot(tt, q, 'black.')
hold on
plot(1:Td, mi_q, 'r', 'LineWidth', 2)
xlabel('[day of the year]')
ylabel('[m^3/s]')
legend('Inflow data', 'Moving average mean')
xlim([1 365])

figure
plot(time,x,'black.')
xlabel('[day]')
ylabel('[m^3/s]')
title('Deseasonalized inflow')
% COMMENT : periodicity has been (partially) removed

% Autocorrelation
figure('Name','Autocorrelation of Inflow data')
correlogram(x, x, 20);
xlabel('k')
ylabel('r_k')
title('Autocorrelation of Inflow data')

%%  AR(1)
% With cross validation study, through 'splitting.m' function
k = 3; % number of splits

[xc_split, xv_split, pc_split, pv_split, tc_split, tv_split, s_q_c, s_q_v, m_q_c, m_q_v] = ...
    splitting(x, u_p, u_t, k, s_q, m_q);

qc_split = xc_split .* s_q_c + m_q_c;
qv_split = xv_split .* s_q_v + m_q_v;

% NOTE: splitted exogenous arrays are for later use 

%--------------------------------------------------------------------------
% initializations
qc_hat_split = zeros(length(xc_split),k);
qv_hat_split = zeros(length(xv_split),k);

MSE_split = zeros(k,2);
R2_split  = zeros(k,2);
% first column calibration, second column validation

for i = 1:k
    % calibration of i-th split
    yc_i = xc_split(2:end,i);
    Mc_i = xc_split(1:end-1,i);

    theta = Mc_i\yc_i;

    % prediction + seasonality
    xc_hat_i = [ xc_split(1,i) ; Mc_i*theta ];
    qc_hat_split(:,i) = xc_hat_i .* s_q_c(:,i) + m_q_c(:,i);

    MSE_split(i,1) = mean((qc_split(2:end,i)-qc_hat_split(2:end,i)).^2);

    R2_split(i,1) = 1 - sum((qc_split(2:end,i)-qc_hat_split(2:end,i)).^2) / ...
        sum((qc_split(2:end,i)-m_q_c(2:end,i)).^2);  

    % validation of i-th split
    Mv_i = xv_split(1:end-1,i);

    xv_hat_i = [ xv_split(1,i) ; Mv_i*theta ];
    qv_hat_split(:,i) = xv_hat_i .* s_q_v(:,i) + m_q_v(:,i);

    MSE_split(i,2) = mean((qv_split(2:end,i)-qv_hat_split(2:end,i)).^2);

    R2_split(i,2) = 1 - sum((qv_split(2:end,i)-qv_hat_split(2:end,i)).^2) / ...
        sum((qv_split(2:end,i)-m_q_v(2:end,i)).^2);  
end

%------------------------------------------------------------------------
% plot section
h = 0.5; % plotting start height for R2

% plot indicators
X = categorical({'Split #1','Split #2','Split #3'});
X = reordercats(X,{'Split #1','Split #2','Split #3'});
figure('Name', 'Indicators for AR(1) depending on split')
subplot(2,1,1)
Y = MSE_split;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
box off
title('MSE along split')
subplot(2,1,2)
Y = R2_split;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
box off
ylim([h 1.2])
hold on
yline(1,'black--','perfect pred')
title('R2 along split') %compare performance over splitting

kbest=3; 
% best split chosen arbitrarly because performances are very similar 
figure('Name', 'AR(1) wrt best split')
subplot(2,1,1)
plot([qc_split(:,kbest) qc_hat_split(:,kbest)])
xlim([1 length(qc_split)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Calibration dataset')
subplot(2,1,2)
plot([qv_split(:,kbest) qv_hat_split(:,kbest)])
xlim([1 length(qv_split)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Validation dataset') 



%% AR(i) cycle
% Calibration and validation of AR(i) models on the previosly defined
% best dataset from the 'cross validation'('best splitting') done on AR(1), procedure done
% until hit the maximum defined model order i_max
 
% Done with the help of two functions, for conciseness.
% 'calibrate.m' to do an automatic model calibration for the AR(i)
% 'validate.m' same but for validation

i_maxAR    = 10; % maximum model order to investigate
k_choice   = 3;  % which splitting wanted as operative dataset
 
%--------------------------------------------------------------------------
% selection of operative datasets and respective mean/variance arrays
xc = xc_split(:,k_choice);
xv = xv_split(:,k_choice);

qc = qc_split(:,k_choice);
qv = qv_split(:,k_choice);

sc = s_q_c(:,k_choice);
sv = s_q_v(:,k_choice);

mc = m_q_c(:,k_choice);
mv = m_q_v(:,k_choice);

%--------------------------------------------------------------------------
% initializations for the procedure
qc_hat_ARi = zeros(length(qc),i_maxAR);
qv_hat_ARi = zeros(length(qv),i_maxAR);

MSE_ARi = zeros(i_maxAR,2);
R2_ARi  = zeros(i_maxAR,2);
% left calibration right for validation

%--------------------------------------------------------------------------
% main procedure

 for i = 1:i_maxAR
     [theta_i, xc_hat_i] = calibrate(xc, i, 0, 0, 0);
     qc_hat_ARi(:,i) = xc_hat_i .* sc + mc;

     MSE_ARi(i,1) = mean((qc(1+i:end) - qc_hat_ARi(1+i:end,i)).^2);

     R2_ARi(i,1) = 1 - sum((qc(i+1:end) - qc_hat_ARi(i+1:end,i)).^2)/...
         sum((qc(i+1:end) - mc(i+1:end)).^2);

     xv_hat_i = validate(theta_i, xv, i, 0, 0, 0);
     qv_hat_ARi(:,i) = xv_hat_i .* sv + mv;

     MSE_ARi(i,2) = mean((qv(1+i:end) - qv_hat_ARi(i+1:end,i)).^2);

     R2_ARi(i,2)= 1 - sum((qv(i+1:end) - qv_hat_ARi(i+1:end,i)).^2)/...
         sum((qv(i+1:end) - mv(i+1:end)).^2);
 end
 
%--------------------------------------------------------------------------
% plotting sequence
i_plot = 4; % order to be plotted
h = 0.8; % height of indicators plot, better vision

% indicators
X = categorical({'AR(1)','AR(2)','AR(3)','AR(4)','AR(5)','AR(6)','AR(7)','AR(8)','AR(9)','AR(10)'});
X = reordercats(X,{'AR(1)','AR(2)','AR(3)','AR(4)','AR(5)','AR(6)','AR(7)','AR(8)','AR(9)','AR(10)'});
figure('Name', 'Indicators for AR(i)')
ylim([h 1.2])
subplot(2,1,1)
Y = MSE_ARi;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
title('MSE scores')
subplot(2,1,2)
Y = R2_ARi;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
ylim([h 1.2])
hold on
yline(1,'black--','perfect pred')
title('R2 scores')

% datasets
figure('Name', 'AR(i) Datasets ')
subplot(2,1,1)
plot([qc qc_hat_ARi(:,i_plot)])
xlim([1 length(qc)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Calibration dataset')
subplot(2,1,2)
plot([qv qv_hat_ARi(:,i_plot)])
xlim([1 length(qv)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Validation dataset') 

%% ARX(i,2)
% Same procedure as before, but investigated for ARX(i,2) models
% still using the best datasets from the AR(1) cc case and
% still based on the two functions defined before, which now require as an
% input the informations about the EX part as well, both precipitation and 
% temperature in this case

i_maxARX = 10; % maximum model order to reach
pro = 0;       % properness, 0 if proper 1 if improper

%--------------------------------------------------------------------------
% Exogenous datasets of the case, always related to kbest chosen before
tc = tc_split(:,k_choice);
tv = tv_split(:,k_choice);

pc = pc_split(:,k_choice);
pv = pv_split(:,k_choice);

% NOTE: For the AR datasets, used the same as for the AR
% Also the corresponding mean and variances arrays chosen before

%--------------------------------------------------------------------------
% initializations
qc_hat_ARXi2 = zeros(length(qc),i_maxARX);
qv_hat_ARXi2 = zeros(length(qv),i_maxARX);

MSE_ARXi2 = zeros(i_maxAR, 2);
R2_ARXi2  = zeros(i_maxAR, 2);
% left calibration right validation

%--------------------------------------------------------------------------
% main procedure

 for i = 1:i_maxARX
     [theta_i, xc_hat_i] = calibrate(xc, i, [pc tc], 2, pro);
     qc_hat_ARXi2(:,i) = xc_hat_i .* sc + mc;

     MSE_ARXi2(i,1) = mean((qc(1+i:end) - qc_hat_ARXi2(1+i:end,i)).^2);

     R2_ARXi2(i,1) = 1 - sum((qc(i+1:end) - qc_hat_ARXi2(i+1:end,i)).^2)/...
         sum((qc(i+1:end) - mc(i+1:end)).^2);

     xv_hat_i = validate(theta_i, xv, i, [pv tv], 2, pro);
     qv_hat_ARXi2(:,i) = xv_hat_i .* sv + mv; 

     MSE_ARXi2(i,2) = mean((qv(1+i:end) - qv_hat_ARXi2(1+i:end,i)).^2);

     R2_ARXi2(i,2)= 1 - sum((qv(i+1:end) - qv_hat_ARXi2(i+1:end,i)).^2)/...
         sum((qv(i+1:end) - mv(i+1:end)).^2);
 end
 
%--------------------------------------------------------------------------
% plotting sequence
i_plot = 2; % order to be plotted
h = 0.8; % height of indicators plot, better vision

% indicators
X = categorical({'ARX(1,2)','ARX(2,2)','ARX(3,2)','ARX(4,2)','ARX(5,2)','ARX(6,2)','ARX(7,2)','ARX(8,2)','ARX(9,2)','ARX(10,2)'});
X = reordercats(X,{'ARX(1,2)','ARX(2,2)','ARX(3,2)','ARX(4,2)','ARX(5,2)','ARX(6,2)','ARX(7,2)','ARX(8,2)','ARX(9,2)','ARX(10,2)'});
figure('Name', 'Indicators for ARX(i,2)')
ylim([h 1.2])
subplot(2,1,1)
Y = MSE_ARXi2;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
title('MSE scores')
subplot(2,1,2)
Y = R2_ARXi2;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
ylim([h 1.2])
hold on
yline(1,'black--','perfect pred')
title('R2 scores (imp)')

% datasets
figure('Name', 'ARX(i,2) Datasets ')
subplot(2,1,1)
plot([qc qc_hat_ARXi2(:,i_plot)])
xlim([1 length(qc)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Calibration dataset')
subplot(2,1,2)
plot([qv qv_hat_ARXi2(:,i_plot)])
xlim([1 length(qv)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Validation dataset') 

%rifare procedura anche con pro=1 per avere modelli impropri

%%  ARX(1,1) PROPER/IMPROPER (precipitation only)
% Realization of an ARX of order 1 with only precipitation, either proper
% or improper according to parameter pro

pro = 0; % 0 if proper, 1 if improper

% NOTE: Uses same datasets as before!

MSE_ARXp = [0, 0];
R2_ARXp  = [0, 0];

% calibration
y = xc(2:end);
Mc = [ xc(1:end-1) pc(1+pro:end-1+pro) ];  

theta = Mc \ y;

xc_hat_ARXp = [ xc(1) ; Mc*theta ];
qc_hat_ARXp = xc_hat_ARXp .* sc + mc;

MSE_ARXp(1,1) = mean((qc(2:end) - qc_hat_ARXp(2:end)).^2);
R2_ARXp(1,1) = 1 - sum((qc(2:end) - qc_hat_ARXp(2:end)).^2)/...
         sum((qc(2:end) - mc(2:end)).^2); 

% validation
Mv = [ xv(1:end-1) pv(1+pro:end-1+pro) ];
xv_hat_i = [ xv(1) ; Mv*theta ];
qv_hat_ARXp = xv_hat_i .* sv + mv;

MSE_ARXp(1,2) = mean((qv(2:end) - qv_hat_ARXp(2:end)).^2);
R2_ARXp(1,2) = 1 - sum((qv(2:end) - qv_hat_ARXp(2:end)).^2)/...
         sum((qv(2:end) - mv(2:end)).^2);

%--------------------------------------------------------------------------
% plot sequence
h = 0.8; % height of indicators plot, better vision

% indicators
X = categorical({'ARX(1,1)'});
figure('Name', 'Indicators for ARX(1,1) with precipitation')
ylim([h 1.2])
subplot(2,1,1)
Y = MSE_ARXp;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
title('MSE scores')
subplot(2,1,2)
Y = R2_ARXp;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
ylim([h 1.2])
hold on
yline(1,'black--','perfect pred')
title('R2 scores')

% datasets
figure('Name', 'ARX(1,1) Datasets with precipitation ')
subplot(2,1,1)
plot([qc qc_hat_ARXp])
xlim([1 length(qc)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Calibration dataset')
subplot(2,1,2)
plot([qv qv_hat_ARXp])
xlim([1 length(qv)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Validation dataset')


%%  ARX(1,1) PROPER/IMPROPER (temperature only)
% Same procedure as before but throught the use of temperature

pro = 0; % 0 if proper, 1 if improper

% NOTE: Uses same datasets as before!

MSE_ARXt = [0, 0];
R2_ARXt  = [0, 0];

% calibration
y = xc(2:end);
Mc = [ xc(1:end-1) tc(1+pro:end-1+pro) ];  

theta = Mc \ y;

xc_hat_ARXt = [ xc(1) ; Mc*theta ];
qc_hat_ARXt = xc_hat_ARXt .* sc + mc;

MSE_ARXt(1,1) = mean((qc(2:end) - qc_hat_ARXt(2:end)).^2);
R2_ARXt(1,1) = 1 - sum((qc(2:end) - qc_hat_ARXt(2:end)).^2)/...
         sum((qc(2:end) - mc(2:end)).^2); 

% validation
Mv = [ xv(1:end-1) tv(1+pro:end-1+pro) ];
xv_hat_i = [ xv(1) ; Mv*theta ];
qv_hat_ARXt = xv_hat_i .* sv + mv;

MSE_ARXt(1,2) = mean((qv(2:end) - qv_hat_ARXt(2:end)).^2);
R2_ARXt(1,2) = 1 - sum((qv(2:end) - qv_hat_ARXt(2:end)).^2)/...
         sum((qv(2:end) - mv(2:end)).^2); 

%--------------------------------------------------------------------------
% plot sequence
h = 0.8; % height of indicators plot, better vision

% indicators
X = categorical({'ARX(1,1)'});
figure('Name', 'Indicators for ARX(1,1) with temperature')
subplot(2,1,1)
Y = MSE_ARXt;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
title('MSE scores')
subplot(2,1,2)
ylim([h 1.2])
Y = R2_ARXt;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
hold on
yline(1,'black--','perfect pred')
ylim([h 1.2])
title('R2 scores')

% datasets
figure('Name', 'ARX(1,1) Datasets with temperature ')
subplot(2,1,1)
plot([qc qc_hat_ARXt])
xlim([1 length(qc)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Calibration dataset')
subplot(2,1,2)
plot([qv qv_hat_ARXt])
xlim([1 length(qv)])
ylim([1 10000])
ylabel('[m^3/s]')
xlabel('time step')
legend('Obs', 'Pred')
title('Validation dataset')


%%  ANN Shallow/Deep Proper/Improper
% ANN training procedure done N times to hopefully counteract possible ill
% initializations, that could lead to bad performances
% (ANN training is a Non Linear optimization process)

% done through a function 'ann.m' for different set of neurons
% either proper or improper with variable pro
%--------------------------------------------------------------------------
% preliminary values
N_runs  = 5;   % training procedure done multiple times
pro = 0;        % properness of the network
R2t = 0.8;      % R2 score target

n_deep = [10 10];   
n_shallow = 20;

% NOTE: for a single ANN realization put N_runs = 1 and ignore R2t

%--------------------------------------------------------------------------
% Operative datasets used
uc = [pc tc];
uv = [pv tv];

% tried with both of the exogenous informations

% NOTE: X, Y for the ANN actual training are defined inside function 'ann.m'

%--------------------------------------------------------------------------
% All types of ANN that were tried

[qc_ANNdeep, qv_ANNdeep, R2_ANNdeep, NR2t_deep, ibest_deep] = ...
    ann(xc, xv, uc, uv, mc, mv, sc, sv, pro, N_runs, R2t, n_deep); 

[qc_ANNshal, qv_ANNshal, R2_ANNshal, NR2t_shal, ibest_shal] = ...
    ann(xc, xv, uc, uv, mc, mv, sc, sv, pro, N_runs, R2t, n_shallow); 

% NOTE: Computed only R2 score

%--------------------------------------------------------------------------
% plotting sequence
% plotted only best ANN 
[~, i_bestDeep] = max(R2_ANNdeep(:,2));
[~, i_bestShal] = max(R2_ANNshal(:,2));
% best one is the one with higher R2 score in validation
figure('Name', 'ANN prediction, in validation ')
subplot(2,1,1)
plot([qv qv_ANNdeep(i_bestDeep,:)'])
legend('Val data','ANN pred')
ylabel('[m^3/s]')
xlabel('time step')
title('Deep ANN prediction')
subplot(2,1,2)
plot([qv qv_ANNshal(i_bestShal,:)'])
legend('Val data','ANN pred')
ylabel('[m^3/s]')
xlabel('time step')
title('Shallow ANN prediction')

h = 0.8; % usual
X = categorical({'#1','#2','#3','#4','#5'});
X = reordercats(X,{'#1','#2','#3','#4','#5'});
figure('Name', 'ANN R2 score wrt multiple initializations')
subplot(2,1,1)
Y = R2_ANNdeep;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
hold on
yline(1,'black--','perfect pred')
ylim([h 1.2])
xlabel('Initialization')
title('Deep')
subplot(2,1,2)
Y = R2_ANNshal;
B = bar(X,Y);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
hold on
yline(1,'black--','perfect pred')
ylim([h 1.2])
xlabel('Initialization')
title('Shallow')



% NOTE: If the performances of the ANN arent that much bigger than the ARXs ones
% it makes no sense to use it as it takes higher computational demands for
% not that much increase in performance, like in our case where results are
% comparable with those obtained beforen with the linear models 

%% CARTs  
% Carts models generally overfit, counteracted with manipolation of
% MaxNumSplits and MinSplitSize and MaxDepth

% operative datasets for ALL the CARTS
uc = [pc tc];
uv = [pv tv];

% for CART fitting and validation
Xc = [xc(1:end-1), uc(1+pro:end-1+pro,:)];
Yc = xc(2:end);

Xv = [xv(1:end-1), uv(1+pro:end-1+pro,:)];

%--------------------------------------------------------------------------
% indicators for all the CARTS
R2_cart_mls10  = [0, 0];
R2_cart_mls100 = [0, 0];
R2_cart_mls50 = [0, 0];

% indicators for all the RANDOM FORESTS
R2_RFf10 = [0, 0];
R2_RFf100 = [0, 0];
R2_RFf50 = [0, 0];

% Only R2 score


%% CART MinLeafSize 10

T10 = fitrtree(Xc, Yc, 'MinLeafSize', 10);
view(T10, 'mode', 'graph')

Yc_mls10 = [ x(1); predict(T10, Xc) ];
qc_cart_mls10 = Yc_mls10 .* sc + mc;

R2_cart_mls10(1,1) = 1 - sum((qc(2:end)-qc_cart_mls10(2:end)).^2) / ...
    sum((qc(2:end)-mc(2:end)).^2); 

% validation
Yv_mls10 = [ xv(1); predict(T10, Xv) ] ;
qv_cart_mls10 = Yv_mls10 .* sv + mv ;

R2_cart_mls10(1,2) = 1 - sum((qv(2:end)-qv_cart_mls10(2:end)).^2) / ...
    sum((qv(2:end)-mv(2:end)).^2); 

%% CART MinLeafSize 100

T100 = fitrtree(Xc, Yc, 'MinLeafSize', 100);
view(T100, 'mode', 'graph')

Y_mls100 = [ x(1); predict(T100, Xc) ];
qc_cart_mls100 = Y_mls100 .* sc + mc;

R2_cart_mls100(1,1) = 1 - sum((qc(2:end)-qc_cart_mls100(2:end)).^2) / ...
    sum((qc(2:end)-mc(2:end)).^2); 

% validation
Yv_mls100 = [ xv(1); predict(T100, Xv)] ;
qv_cart_mls100 = Yv_mls100 .* sv + mv ;

R2_cart_mls100(1,2) = 1 - sum((qv(2:end)-qv_cart_mls100(2:end)).^2) / ...
    sum((qv(2:end)-mv(2:end)).^2);

%% CART MinLeafSize 50

T50 = fitrtree(Xc, Yc, 'MinLeafSize', 50);
view(T50, 'mode', 'graph')

Yc_mls50 = [ x(1); predict(T50, Xc) ];
qc_cart_mls50 = Yc_mls50 .* sc + mc;

R2_cart_mls50(1,1) = 1 - sum((qc(2:end)-qc_cart_mls50(2:end)).^2) / ...
    sum((qc(2:end)-mc(2:end)).^2); 

% validation
Yv_pro50 = [ xv(1); predict(T50, Xv) ] ;
qv_cart_pro50 = Yv_pro50 .* sv + mv ;

R2_cart_mls50(1,2) = 1 - sum((qv(2:end)-qv_cart_pro50(2:end)).^2) / ...
    sum((qv(2:end)-mv(2:end)).^2); 

%--------------------------------------------------------------------------
% plot section
% indicators comparison of all CARTs
h = 0.8;
figure('Name','CARTs'' indicators')
B = bar([R2_cart_auto; R2_cart_mls10; R2_cart_mls100; R2_cart_mls50]);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
hold on
yline(1,'black--','perfect pred')
set(gca, 'XTickLabel',{ 'CART auto', 'CART mls 10', 'CART mls 100', 'CART mls 50'})
ylim([h 1.2])

%% RANDOM FOREST  MinLeafSize 10

% training
tf10 = templateTree('MinLeafSize', 10);
RF10 = fitrensemble(Xc, Yc, 'Method', 'Bag', ...
    'Learners', tf10, ...
    'NumLearningCycles', 20);

Yc_RFmls10 = [ x(1); predict(RF10, Xc) ];
qc_RFf10 = Yc_RFmls10 .* sc + mc ;

R2_RFf10(1,1) = 1 - sum((qc(2:end)-qc_RFf10(2:end)).^2) / ...
    sum((qc(2:end)-mc(2:end)).^2); 

% validation
Yv_RFf10 = [ xv(1); predict(RF10, Xv) ];
qv_RFf10 = Yv_RFf10 .* sv + mv;

R2_RFf10(1,2) = 1 - sum((qv(2:end)-qv_RFf10(2:end)).^2) / ...
    sum((qv(2:end)-mv(2:end)).^2); 


%% RANDOM FOREST MinLeafSize 100

% training
tf100 = templateTree('MinLeafSize', 100);
RF100 = fitrensemble(Xc, Yc, 'Method', 'Bag', ...
    'Learners', tf100, ...
    'NumLearningCycles', 20);

Yc_RFf100 = [ x(1); predict(RF100, Xc) ];
q_RFf100 = Yc_RFf100 .* sc + mc ;

R2_RFf100(1,1) = 1 - sum((qc(2:end)-q_RFf100(2:end)).^2) / ...
    sum((qc(2:end)-mc(2:end)).^2);

% validation
Yv_RFf100 = [ xv(1); predict(RF100, Xv) ];
qv_RFf100 = Yv_RFf100 .* sv + mv;

R2_RFf100(1,2) = 1 - sum((qv(2:end)-qv_RFf100(2:end)).^2) / ...
    sum((qv(2:end)-mv(2:end)).^2);

%% RANDOM FOREST MinLeafSize 50

% training
tf50 = templateTree('MinLeafSize', 50);
RF50 = fitrensemble(Xc, Yc, 'Method', 'Bag', ...
    'Learners', tf50, ...
    'NumLearningCycles', 20);

Yc_RFf50 = [ x(1); predict(RF50, Xc) ];
q_RFf50 = Yc_RFf50 .* sc + mc ;

R2_RFf50(1,1) = 1 - sum((qc(2:end)-q_RFf50(2:end)).^2) / ...
    sum((qc(2:end)-mc(2:end)).^2); 

% validation
Yv_RFf50 = [ xv(1); predict(RF50, Xv) ];
qv_RFf50 = Yv_RFf50 .* sv + mv;
R2_RFf50(1,2) = 1 - sum((qv(2:end)-qv_RFf50(2:end)).^2) / ...
    sum((qv(2:end)-mv(2:end)).^2); 

%--------------------------------------------------------------------------
% plot section
% indicators comparison of all RFs
h = 0.8;
figure('Name','Random Forests'' indicators')
B = bar([R2_RFf10; R2_RFf100; R2_RFf50;]);
text(B(1).XEndPoints,B(1).YEndPoints,num2str((B(1).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
text(B(2).XEndPoints,B(2).YEndPoints,num2str((B(2).YData)',2),...
    'HorizontalAlignment','center','VerticalAlignment','bottom'); 
hold on
yline(1,'black--','perfect pred')
set(gca, 'XTickLabel',{'RF mla10', 'RF mls100', 'RF mla50'})
ylim([h 1.2])

%% Part 2
% Investigation of a policy plus discussion of alternative 0 (no dam) and
% an hypothetical damming of the river, according to some indicators
%--------------------------------------------------------------------------
% data and parameters

% streanflow data is still 'q'

S = 5e7; % [m^2] dam surface, inferred from structures in the area


w_target = 1170; % target release = downstream water demand w  [m^3/s] 
%w=10th percentile of q( prctile(q,10)=1170)


%% Optimal storage computation
% With Sequent Peak Analysis
%   reservoir is simulated with a mass balance, to search for K max REAL
%   storage supporting the desired Rt+1
%   mass balance has an inverted sign, base idea is to have some water in
%   the reserve when n(t) < r(t)
%   its much more conservative as a method, assumes that we cannot store
%   water in surplus for future use
%   basically plotted kt sequence obtained with m. balance and then K is max
%   of that series
% 
%   ADVANTAGE: Exchange of water on >1 year is taken into account.
%   (can be done on multiple periods)

%--------------------------------------------------------------------------
% monthly mean flow
% dailyToMonthly converts a daily values into monthly average values
qMonth = dailyToMonthly(q, 27); % m3/s

deltaT = 60*60*24*[31 28 31 30 31 30 31 31 30 31 30 31]';
% detltaT contains the number of seconds in each month
Q = qMonth(:).*repmat(deltaT,27,1); % m3/month
W = w_target*ones(size(Q)).*repmat(deltaT,27,1); % m3/month
time_m = datetime(1990,1,1) + calmonths(0:27*12-1); % dates, for plotting
time_mk = datetime(1990,1,1) + calmonths(0:27*12);

figure('Name','Monthly request vs monthly flow')
plot(time_m,Q)
hold on
plot(time_m,W)
title('Monthly request vs monthly flow')
legend('Monthly flow','Target request')
xlabel('Month')
ylabel('[m^3]')

% Sequent Peak Analysis
K = zeros(size(Q));
K(1) = 0; 

for t = 1:length(Q)
    K(t+1) = K(t) + W(t) - Q(t);
    if K(t+1) < 0
        K(t+1) = 0;
    end
end

figure('Name','Sequent Peak Analysis')
plot(time_mk,K)
title('Optimal volume')
xlabel('Month')
ylabel('[m^3]')

Kopt = max(K); % optimal volume
hopt = Kopt/S; % optimal height


%% parameters of the natural release

hmax= hopt;
q_max= max(q);
param.nat.S = S;
param.nat.alpha = q_max/hmax;
param.nat.h0 = 0;

% regulated level-discharge relationship:
param.reg.w = w_target;
param.reg.h_min = 0;
param.reg.h_max = hmax;
param.reg.h1 = 0.5;
param.reg.h2 = 10;
param.reg.m1 = 300;
param.reg.m2 = 800;

%max natural release
h_nat= 0:1:30;
r_nat=(q_max/hmax)*h_nat;
figure('Name','Max natural release')
plot(h_nat, r_nat)
hold on
scatter(hmax, q_max, 'filled')
title('Max natural release')
xlabel('h [m]')
ylabel('r_{max} [m^3/s]')
legend('max release', 'C', 'Location','northwest')


%%  TEST REGULATED LEVEL-DISCHARGE RELATIONSHIP
h_test = -1.5:0.5:40;
r_test = regulated_release( param, h_test );
figure
plot(h_test, r_test)
title('Regulation policy')
xlabel('h [m]')
ylabel('release [m^3/s]')

%%  SIMULATION OF LAKE DYNAMICS
% -------------------------------
q_sim = [ nan; q ]; % for time convention
h_init = 10; % initial condition

 
[s_reg, h_reg, r_reg] = simulate_reg_lake( q_sim, h_init, param );

s_reg = s_reg(2:end);
h_reg = h_reg(2:end);
r_reg = r_reg(2:end);
 
%natural release coincides with q


%% indicators

w = param.reg.w;

% WATER SUPPLY (farmers)
% Vulnerability
def_nat = max( w-q, 0 ); % daily deficit
Iirr_nat = mean( def_nat.^2);

def_reg = max( w-r_reg, 0 ); % daily deficit
Iirr_reg = mean( def_reg.^2);

% FLOODING 
h_flo=24;
Iflo_nat=0;
%flooding indicator for natural case (i.e. the alternative 0) is considered
%zero because in our area there is no actual basin, furthermore, zero is
%the ideal value of the indicator

Ny_reg = length(h_reg)/365;
Iflo_reg = sum( h_reg>h_flo )/Ny_reg;

% ENVIRONMENT
q_sim=q_sim(2:end);
LP = prctile(q_sim, 25); 
IE1_reg = sum( r_reg < LP )/Ny_reg;
% natural conditions (using inflow)
Ny_nat=length(q)/365;
IE1_nat = sum( q_sim < LP )/Ny_nat;

IE= abs(IE1_reg - IE1_nat); %while the single indicators only focus on minimizing the days where
%the release is under a certain threshold(MEF), the ideal case would be the
%one where the days with the release under that threshold coincide between
%the regulated and the natural case


%%  NSGAII-EMODPS OPTIMIZATION 
global opt_inputs;
opt_inputs.q_sim = q_sim;
opt_inputs.h_init = h_init;
opt_inputs.param = param;
opt_inputs.h_flo = h_flo;

addpath('NSGA2')
pop = 40; %population size
gen = 20; %number of generation termination creterion
M = 3; %number of objectives
V = 4; %number of decision variables
min_range = [ 0 10 100 500 ]; %range of variation of decision variables
max_range = [ 10 24 900 1000 ];
[ chromosome_0, chromosome15 ] = nsga_2(pop,gen,M,V,min_range,max_range); %output values of objectives and decison variables

% objective space
figure('Name','Pareto frontier of EMODPS')
scatter3(chromosome_0(:,7), chromosome_0(:,6), chromosome_0(:,5), ...
    5, 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', [0 0 0])
%col 5 = irrigation, col 6 = flood, col 7=environment
hold on
scatter3(chromosome15(:,7), chromosome15(:,6), chromosome15(:,5), ...
    5, 'MarkerEdgeColor', [1 0 0], 'MarkerFaceColor', [1 0 0])
legend('Inital pop','Final pop')
title('Pareto frontier')
xlabel('irrigation'); ylabel('flood'); zlabel('environment');

% decision space
h_test = -1.5:0.5:25;
r_test = nan(pop, length(h_test));

for i = 1:pop
    xi = chromosome15(i,1:4);
    param.reg.h1 = xi(1);
    param.reg.h2 = xi(2);
    param.reg.m1 = xi(3);
    param.reg.m2 = xi(4);
    r_test(i,:) = regulated_release( param, h_test );
end

figure
plot( h_test, r_test )
xlabel('level')
ylabel('release')

% best irrigation alternative A1
[min_irr, index_irr] = min(chromosome15(:,5));
best_irr = chromosome15(index_irr, :);

% best flooding alternative A2
[min_flo, index_flo] = min(chromosome15(:,6));
best_flo = chromosome15(index_flo, :);

% best enviroment alternative A3
[min_env, index_env] = min(chromosome15(:,7));
best_env = chromosome15(index_env, :);

% compromise alternative A4
[compromise, index_compr]=min(chromosome15(:,5)/max(chromosome15(:,5)) + chromosome15(:,6)/max(chromosome15(:,6)) + chromosome15(:,7)/max(chromosome15(:,7)));
compr = chromosome15(index_compr, :);

% plot of indicators related to the identified alternatives
figure('Name','Indicators of alternatives')
scatter3( Iirr_nat, Iflo_nat, 0, 120, 'redp', 'filled')
hold on
scatter3(best_irr(1,5), best_irr(1,6), best_irr(1,7), 70, 'bo', 'filled')
hold on
scatter3(best_flo(1,5), best_flo(1,6), best_flo(1,7), 70, 'co', 'filled')
hold on
scatter3(best_env(1,5), best_env(1,6), best_env(1,7), 70, 'go', 'filled')
hold on
scatter3(compr(1,5), compr(1,6), compr(1,7), 70, 'ko', 'filled')
xlabel('irrigation'); ylabel('flood'); zlabel('environment');
legend('A0', 'A1','A2', 'A3', 'A4')
title('Indicators of alternatives')

