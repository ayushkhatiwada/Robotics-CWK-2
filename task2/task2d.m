%% Configuration Section
% Dataset Parameters
dataFilename = 'IrisData.txt';
trainRatio = 0.7;
seed = 42; % Random seed for reproducibility

% General Training Parameters
maxEpochs = 500;
performanceGoal = 1e-6; % Mean squared error goal

%% Section 1: Data Loading and Preprocessing

[X, y] = loadIrisData(dataFilename);
X = (X - mean(X,1)) ./ std(X,0,1);
rng(seed);
n = size(X, 1);
randomIdx = randperm(n);
X = X(randomIdx, :);
y = y(randomIdx, :);
splitPoint = floor(trainRatio * n);
X_train = X(1:splitPoint, :);
y_train = y(1:splitPoint, :);
X_val = X(splitPoint + 1:end, :);
y_val = y(splitPoint + 1:end, :);
X_train = X_train';
y_train = y_train';
X_val = X_val';
y_val = y_val';

% --- Enhanced Plot Settings for Readability at Small Sizes ---
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontSize', 16);
set(0, 'DefaultLineLineWidth', 6);
% --------------------------------------------------------------

%% Section 2: Experiment Setup - Explicit Configuration

architectures = {
    % --- 1. Architecture Variation (Depth) ---
    % Structure: {hidden_layer_sizes, training_function, experiment_name, activation_functions}
    {[5], 'traingdm', 'Depth-1HL (4-5-3), tansig, traingdm', {'tansig', 'tansig'}},
    {[5 3], 'traingdm', 'Depth-2HL (4-5-3-3), tansig, traingdm', {'tansig', 'tansig', 'tansig'}},
    {[5 5 5], 'traingdm', 'Depth-3HL (4-5-5-5-3), tansig, traingdm', {'tansig', 'tansig', 'tansig', 'tansig'}},
    {[5 5 5 5], 'traingdm', 'Depth-4HL (4-5-5-5-5-3), tansig, traingdm', {'tansig', 'tansig', 'tansig', 'tansig', 'tansig'}},

    % --- 1.2 Varying Width (2 Hidden Layers) ---
    {[3 2], 'traingdm', 'Width-Narrower (4-3-2-3), tansig, traingdm', {'tansig', 'tansig', 'tansig'}},
    {[5 3], 'traingdm', 'Width-Original (4-5-3-3), tansig, traingdm', {'tansig', 'tansig', 'tansig'}},
    {[10 5], 'traingdm', 'Width-Wider (4-10-5-3), tansig, traingdm', {'tansig', 'tansig', 'tansig'}},
    {[20 3], 'traingdm', 'Width-WiderFirstLayer (4-20-3-3), tansig, traingdm', {'tansig', 'tansig', 'tansig'}},

    % --- 2. Activation Function Exploration ---
    % All use tansig output
    {[5 3], 'traingdm', 'Activation-tansig (4-5-3-3), traingdm', {'tansig', 'tansig', 'tansig'}},
    {[5 3], 'traingdm', 'Activation-ReLU (4-5-3-3), traingdm', {'poslin', 'poslin', 'tansig'}},
    {[5 3], 'traingdm', 'Activation-Sigmoid (4-5-3-3), traingdm', {'logsig', 'logsig', 'tansig'}},
    {[5 3], 'traingdm', 'Activation-Mixed_ReLU-tansig (4-5-3-3), traingdm', {'poslin', 'tansig', 'tansig'}},
    {[5 3], 'traingdm', 'Activation-Mixed_tansig-ReLU (4-5-3-3), traingdm', {'tansig', 'poslin', 'tansig'}},
};

numExperiments = length(architectures);

%% Section 3: Training and Evaluation

validationAccuracies = zeros(1, numExperiments);
experimentPerformance = cell(1, numExperiments);

for i = 1:numExperiments
    % Unpack architecture details
    hiddenLayers = architectures{i}{1};
    trainFcn = architectures{i}{2};
    experimentName = architectures{i}{3};
    activationFcns = architectures{i}{4};

    % Create a feedforward network with improved initialization
    net = feedforwardnet(hiddenLayers, trainFcn);
    
    % Set training parameters
    net.trainParam.goal = performanceGoal;
    net.trainParam.epochs = maxEpochs;
    net.trainParam.showWindow = false;
    net.trainParam.min_grad = 1e-7;
    net.trainParam.lr = 0.01;  % Set a smaller learning rate
    net.trainParam.mc = 0.9;   % Set momentum
    net.divideFcn = 'dividetrain';
    
    % Initialize weights with smaller values
    net.initFcn = 'initlay';
    for j = 1:length(net.layers)
        net.layers{j}.initFcn = 'initwb';
        net.layers{j}.transferFcn = activationFcns{min(j, length(activationFcns))};
    end
    
    % Configure and initialize the network
    net = configure(net, X_train, y_train);
    net = init(net);
    
    % Train the network
    [net, tr] = train(net, X_train, y_train);

    experimentPerformance{i} = tr;

    y_pred = net(X_val);
    validationAccuracy = mean(vec2ind(y_pred) == vec2ind(y_val));
    fprintf('%s - Validation Accuracy: %.2f%%\n', experimentName, validationAccuracy * 100);
    validationAccuracies(i) = validationAccuracy;
end

experimentNames = cell(1, numExperiments);
for i = 1:numExperiments
      experimentNames{i} = architectures{i}{3};
end


%% Section 4: Plotting

maxEpoch = 0;
for i = 1:length(experimentPerformance)
    if max(experimentPerformance{i}.epoch) > maxEpoch
        maxEpoch = max(experimentPerformance{i}.epoch);
    end
end

figure1 = figure('Position', [100, 100, 800, 600]);
hold on;
xlabel('Epoch');
ylabel('Performance (MSE)');
title('Architecture Comparison');
xlim([0, maxEpoch]);
grid on;

figure2 = figure('Position', [900, 100, 800, 600]);
hold on;
xlabel('Epoch');
ylabel('Performance (MSE)');
title('Activation Function Comparison in 4-5-3-3 Network');
xlim([0, maxEpoch]);
grid on;

for i = 1:length(experimentPerformance)
    if contains(experimentNames{i}, 'Depth') || contains(experimentNames{i}, 'Width')
        figure(figure1);
        plot(experimentPerformance{i}.epoch, experimentPerformance{i}.perf, 'DisplayName', experimentNames{i});
    elseif contains(experimentNames{i}, 'Activation')
        figure(figure2);
        plot(experimentPerformance{i}.epoch, experimentPerformance{i}.perf, 'DisplayName', experimentNames{i});
    end
end

figure(figure1);
legend('show', 'Location', 'northeast');
figure(figure2);
legend('show', 'Location', 'northeast');

allAxes = findall(0, 'Type', 'axes');
for i = 1:length(allAxes)
    allAxes(i).XLim = [0, maxEpoch];
    allAxes(i).XTick = 0:100:maxEpoch;
end

saveas(figure1, 'architecture_comparison.fig');
saveas(figure2, 'activation_comparison.fig');

accuracyTable = table(experimentNames', validationAccuracies', 'VariableNames', {'Architecture', 'Validation Accuracy'});
disp(accuracyTable);

%% Helper Function: loadIrisData
function [X, y] = loadIrisData(filename)
    fileID = fopen(filename, 'r');
    if fileID == -1
        error('Could not open file %s', filename);
    end
    X = [];
    y = [];
    while ~feof(fileID)
        currentLine = fgetl(fileID);
        if ischar(currentLine) && ~isempty(currentLine)
            lineParts = strsplit(currentLine, ',');
            irisFeatures = str2double(lineParts(1:4));
            X = [X; irisFeatures];
            if contains(lineParts{5}, 'setosa')
                oneHotLabel = [0.6; -0.6; -0.6];
            elseif contains(lineParts{5}, 'versicolor')
                oneHotLabel = [-0.6; 0.6; -0.6];
            else
                oneHotLabel = [-0.6; -0.6; 0.6];
            end
            y = [y; oneHotLabel];
        end
    end
    fclose(fileID);
end