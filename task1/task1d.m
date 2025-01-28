%% Task 1(d)
clear; clc; close all;

rng(42,'twister');  % Fix random seed

% -------------------------------
% 1) Generate training data
% -------------------------------
x = -1:0.05:1;  % Inputs in [-1,1]
d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x + normrnd(0,0.02,size(x));  % Noisy cubic target

xtest = -0.97:0.1:0.93;  % Test inputs

% Containers for storing results
nets = {};          % store networks
netNames = {};      % store readable name strings
trainMSEs = [];     % store final MSE on training data
numEpochs = [];     % store actual epochs used by trainlm
Rvals = [];         % store correlation (R-value) on training data

%% -------------------------------
% Network A: 1-3-1 (tansig->purelin)
% -------------------------------
netA = feedforwardnet(3,'trainlm');
netA.layers{1}.transferFcn = 'tansig';
netA.layers{2}.transferFcn = 'purelin';
netA.trainParam.epochs = 30000;  % Max epochs

[netA, trA] = train(netA, x, d);

yA_train = netA(x);
mseA = mean((d - yA_train).^2);
epochsA = trA.epoch(end);
RmatA = corrcoef(d, yA_train);
RvalA = RmatA(1,2);

nets{end+1} = netA;
netNames{end+1} = 'A) 1-3-1 (tansig->purelin)';
trainMSEs(end+1) = mseA;
numEpochs(end+1) = epochsA;
Rvals(end+1) = RvalA;

%% -------------------------------
% Network B: 1-3-1 (logsig->purelin)
% -------------------------------
netB = feedforwardnet(3,'trainlm');
netB.layers{1}.transferFcn = 'logsig';
netB.layers{2}.transferFcn = 'purelin';
netB.trainParam.epochs = 30000;

[netB, trB] = train(netB, x, d);

yB_train = netB(x);
mseB = mean((d - yB_train).^2);
epochsB = trB.epoch(end);
RmatB = corrcoef(d, yB_train);
RvalB = RmatB(1,2);

nets{end+1} = netB;
netNames{end+1} = 'B) 1-3-1 (logsig->purelin)';
trainMSEs(end+1) = mseB;
numEpochs(end+1) = epochsB;
Rvals(end+1) = RvalB;

%% -------------------------------
% Network C: 1-5-1 (ReLU->purelin)
% -------------------------------
netC = feedforwardnet(5,'trainlm');
netC.layers{1}.transferFcn = 'poslin';  % ReLU
netC.layers{2}.transferFcn = 'purelin';
netC.trainParam.epochs = 30000;

[netC, trC] = train(netC, x, d);

yC_train = netC(x);
mseC = mean((d - yC_train).^2);
epochsC = trC.epoch(end);
RmatC = corrcoef(d, yC_train);
RvalC = RmatC(1,2);

nets{end+1} = netC;
netNames{end+1} = 'C) 1-5-1 (ReLU->purelin)';
trainMSEs(end+1) = mseC;
numEpochs(end+1) = epochsC;
Rvals(end+1) = RvalC;

%% -------------------------------
% Network D: 2-layer [8,8], ReLU->ReLU->purelin
% -------------------------------
netD = feedforwardnet([8,8],'trainlm');
netD.layers{1}.transferFcn = 'poslin';
netD.layers{2}.transferFcn = 'poslin';
netD.layers{3}.transferFcn = 'purelin';
netD.trainParam.epochs = 30000;

[netD, trD] = train(netD, x, d);

yD_train = netD(x);
mseD = mean((d - yD_train).^2);
epochsD = trD.epoch(end);
RmatD = corrcoef(d, yD_train);
RvalD = RmatD(1,2);

nets{end+1} = netD;
netNames{end+1} = 'D) 2-layer [8,8] (ReLU->ReLU->purelin)';
trainMSEs(end+1) = mseD;
numEpochs(end+1) = epochsD;
Rvals(end+1) = RvalD;

%% -------------------------------
% Network E: 1-5-1 (tansig->purelin)
% -------------------------------
netE = feedforwardnet(5,'trainlm');
netE.layers{1}.transferFcn = 'tansig';
netE.layers{2}.transferFcn = 'purelin';
netE.trainParam.epochs = 30000;

[netE, trE] = train(netE, x, d);

yE_train = netE(x);
mseE = mean((d - yE_train).^2);
epochsE = trE.epoch(end);
RmatE = corrcoef(d, yE_train);
RvalE = RmatE(1,2);

nets{end+1} = netE;
netNames{end+1} = 'E) 1-5-1 (tansig->purelin)';
trainMSEs(end+1) = mseE;
numEpochs(end+1) = epochsE;
Rvals(end+1) = RvalE;

%% -------------------------------
% Network F: 1-5-1 (logsig->purelin)
% -------------------------------
netF = feedforwardnet(5,'trainlm');
netF.layers{1}.transferFcn = 'logsig';
netF.layers{2}.transferFcn = 'purelin';
netF.trainParam.epochs = 30000;

[netF, trF] = train(netF, x, d);

yF_train = netF(x);
mseF = mean((d - yF_train).^2);
epochsF = trF.epoch(end);
RmatF = corrcoef(d, yF_train);
RvalF = RmatF(1,2);

nets{end+1} = netF;
netNames{end+1} = 'F) 1-5-1 (logsig->purelin)';
trainMSEs(end+1) = mseF;
numEpochs(end+1) = epochsF;
Rvals(end+1) = RvalF;

%% --------------------------------------------------------------------
% Print out summary for each network: MSE, actual epochs, R-value
% --------------------------------------------------------------------
fprintf('\n=== Training Summary for Different Networks ===\n');
for i = 1:numel(nets)
    fprintf('%s : MSE = %.6f, epochs = %d, R = %.4f\n', ...
        netNames{i}, trainMSEs(i), numEpochs(i), Rvals(i));
end

%% Plot test predictions
figure('Name','Test Predictions');
plot(x, d, 'ko', 'DisplayName','Training Data'); hold on; grid on;
plotStyles = {'r-','b--','g-.','m-','y--','k-.'}; 


% Evaluate each network on the test inputs and plot
for i = 1:numel(nets)
    ytest_i = nets{i}(xtest);
    plot(xtest, ytest_i, plotStyles{i}, 'LineWidth',1.5, ...
         'DisplayName', netNames{i});
end

xlabel('x'); ylabel('y');
title('Comparison of Networks on Test Data');
legend('Location','best');
hold off;

