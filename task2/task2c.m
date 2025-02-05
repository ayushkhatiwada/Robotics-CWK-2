%% Configuration Section (Experiment Parameters)
% Dataset Parameters
dataFilename = 'IrisData.txt';
trainRatio = 0.7;
seed = 42; % Random seed for reproducibility

% Network Architecture
inputLayerSize = 4;
hiddenLayer1Size = 5;
hiddenLayer2Size = 3;
outputLayerSize = 3;

% --- Experiment Parameters ---
learningRatesToTest_ADAM = [0.1, 0.01, 0.001, 0.0001]; % For ADAM LR sensitivity
learningRate_SGD = 0.01;                               % For SGD in ADAM vs SGD comparison
batchSizesToTest_ADAM = [1, 8, 16, 32, 64, 128];       % For ADAM Batch Size sensitivity
learningRate_ADAM_batchSizeExp = 0.01;                 % LR for Batch Size experiment

numEpochs = 1000;
printInterval = 100;
batchSize = 16;

% ADAM Hyperparameters
beta1 = 0.9;
beta2 = 0.999;
epsilon = 1e-8;

% --- Enhanced Plot Settings for Readability at Small Sizes ---
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontSize', 16);
set(0, 'DefaultLineLineWidth', 2);
% --------------------------------------------------------------

% Results Storage
results = struct(); % Structure to store results for each experiment
experimentCounter = 1; % Counter to keep track of experiments

%% --- Data Loading and Preprocessing ---
% Function to load the Iris dataset from a text file and convert class labels
% to one-hot encoded vectors.

function [features, labels] = loadIrisData(filename)
    fileID = fopen(filename, 'r');
    if fileID == -1
        error('Could not open file: %s', filename);
    end

    features = [];
    labels = [];

    while ~feof(fileID)
        line = fgetl(fileID); % Read a line from the file
        if ischar(line) && ~isempty(line)
            parts = strsplit(line, ','); % Split the line by commas
            irisFeatures = str2double(parts(1:4)); % Extract the first 4 parts as numerical features
            features = [features; irisFeatures];
            
            species = parts{5}; % The 5th part is the species name
            
            % Convert species name to a one-hot encoded label
            switch species
                case 'Iris-setosa',      oneHotLabel = [0.6, -0.6, -0.6];
                case 'Iris-versicolor',  oneHotLabel = [-0.6, 0.6, -0.6];
                case 'Iris-virginica',   oneHotLabel = [-0.6, -0.6, 0.6];
                otherwise, error('Unknown Iris species: %s', species);
            end
            labels = [labels; oneHotLabel];
        end
    end
    fclose(fileID);
end

%% --- Experiment Setup and Loop ---

% --- 2.1: ADAM Learning Rate Sensitivity Experiments ---
fprintf('\n--- Section 2.1: ADAM Learning Rate Sensitivity Experiments ---\n'); 
for lr_adam = learningRatesToTest_ADAM
    expName_base = sprintf('Exp%d_ADAM_LR_%g', experimentCounter, lr_adam);
    expName = strrep(strrep(expName_base, '.', '_'), '-', '_'); 
    fprintf('\n--- Starting Experiment: %s ---\n', expName_base);
    fprintf('  Optimizer: ADAM, Learning Rate: %g\n', lr_adam);
    results = runExperiment(results, expName, 'adam', lr_adam, batchSize, beta1, beta2, epsilon, numEpochs, printInterval, dataFilename, trainRatio, seed);
    experimentCounter = experimentCounter + 1;
end

% --- 2.2: ADAM vs SGD Comparison Experiments ---
fprintf('\n--- Section 2.2: ADAM vs SGD Comparison Experiments ---\n'); 
optimizersForComparison = {'adam', 'sgd'};
for optimizer_str = optimizersForComparison
    if strcmp(optimizer_str{1}, 'sgd')
        current_lr = learningRate_SGD; % Use pre-defined LR for SGD
    else % ADAM
        current_lr = 0.01; % Fixed LR for ADAM vs SGD comparison
    end
    expName_base = sprintf('Exp%d_%s_CompareSGD_LR_%g', experimentCounter, upper(optimizer_str{1}), current_lr); 
    expName = strrep(strrep(expName_base, '.', '_'), '-', '_'); 
    fprintf('\n--- Starting Experiment: %s ---\n', expName_base);
    fprintf('  Optimizer: %s, Learning Rate: %g\n', upper(optimizer_str{1}), current_lr);
    results = runExperiment(results, expName, optimizer_str{1}, current_lr, batchSize, beta1, beta2, epsilon, numEpochs, printInterval, dataFilename, trainRatio, seed);
    experimentCounter = experimentCounter + 1;
end

% --- 2.3: Batch Size Sensitivity Experiments for ADAM ---
fprintf('\n--- Section 2.3: Batch Size Sensitivity Experiments for ADAM ---\n');
for currentBatchSize = batchSizesToTest_ADAM
    expName_base = sprintf('Exp%d_ADAM_BatchSize_%d', experimentCounter, currentBatchSize);
    expName = strrep(strrep(expName_base, '.', '_'), '-', '_'); 
    fprintf('\n--- Starting Experiment: %s ---\n', expName_base);
    fprintf('  Optimizer: ADAM, Batch Size: %d, Learning Rate: %g\n', currentBatchSize, learningRate_ADAM_batchSizeExp);
    results = runExperiment(results, expName, 'adam', learningRate_ADAM_batchSizeExp, currentBatchSize, beta1, beta2, epsilon, numEpochs, printInterval, dataFilename, trainRatio, seed);
    experimentCounter = experimentCounter + 1;
end

%% --- Plotting and Analysis - Focused and Efficient ---
% --- 3.1: Create Combined Figures ---
% 1. ADAM Learning Rate Sensitivity
figure_lr_adam_accuracy = figure('Name', 'ADAM LR Sensitivity - Accuracy', 'Position', [100, 100, 800, 600]); 
hold on; 
title('Training Accuracy: ADAM Learning Rate Sensitivity'); 
xlabel('Epoch'); 
ylabel('Accuracy'); 
ylim([0, 1.05]); 
grid on;

figure_lr_adam_loss = figure('Name', 'ADAM LR Sensitivity - Loss', 'Position', [100, 100, 800, 600]); 
hold on; 
title('Training Loss: ADAM Learning Rate Sensitivity'); 
xlabel('Epoch'); 
ylabel('Loss'); 
grid on;

% 2. ADAM vs SGD Comparison
figure_adam_sgd_accuracy = figure('Name', 'ADAM vs SGD - Accuracy', 'Position', [100, 100, 800, 600]); 
hold on; 
title('Training Accuracy: ADAM vs SGD Comparison'); 
xlabel('Epoch'); 
ylabel('Accuracy'); 
ylim([0, 1.05]); 
grid on;

figure_adam_sgd_loss = figure('Name', 'ADAM vs SGD - Loss', 'Position', [100, 100, 800, 600]); 
hold on; 
title('Training Loss: ADAM vs SGD Comparison'); 
xlabel('Epoch'); 
ylabel('Loss'); 
grid on;

% 3. ADAM Batch Size Sensitivity
figure_batchSize_adam_accuracy = figure('Name', 'ADAM Batch Size Sensitivity - Accuracy', 'Position', [100, 100, 800, 600]); 
hold on; 
title('Training Accuracy: ADAM Batch Size Sensitivity'); 
xlabel('Epoch'); 
ylabel('Accuracy'); 
ylim([0, 1.05]); 
grid on;

figure_batchSize_adam_loss = figure('Name', 'ADAM Batch Size Sensitivity - Loss', 'Position', [100, 100, 800, 600]); 
hold on; 
title('Training Loss: ADAM Batch Size Sensitivity'); 
xlabel('Epoch'); 
ylabel('Loss'); 
grid on;

% --- 3.2: Plotting Loop (Efficient and Focused) ---
experimentNames = fieldnames(results);
for i = 1:length(experimentNames)
    expName = experimentNames{i}; expResult = results.(expName);

    % 1. ADAM Learning Rate Sensitivity Plots
    if contains(expName, 'ADAM_LR_') && ~contains(expName, 'CompareSGD') && ~contains(expName, 'BatchSize')
        lr_val = expResult.learningRate;
        figure(figure_lr_adam_accuracy); 
        plot(1:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', sprintf('LR = %g', lr_val));
        
        figure(figure_lr_adam_loss); 
        plot(1:numEpochs, expResult.trainLossHistory, 'DisplayName', sprintf('LR = %g', lr_val));
    end

    % 2. ADAM vs SGD Comparison Plots
    if contains(expName, 'CompareSGD')
        optimizer_name = expResult.optimizer;
        figure(figure_adam_sgd_accuracy); 
        plot(1:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', upper(optimizer_name));
        
        figure(figure_adam_sgd_loss); 
        plot(1:numEpochs, expResult.trainLossHistory, 'DisplayName', upper(optimizer_name));
    end

    % 3. ADAM Batch Size Sensitivity Plots
    if contains(expName, 'ADAM_BatchSize_')
        batch_size_val = expResult.batchSize;
        figure(figure_batchSize_adam_accuracy); 
        plot(1:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', sprintf('Batch Size = %d', batch_size_val));
        
        figure(figure_batchSize_adam_loss); 
        plot(1:numEpochs, expResult.trainLossHistory, 'DisplayName', sprintf('Batch Size = %d', batch_size_val));
    end

    % --- Reduce the Number of Ticks for Clarity ---
    ax = findall(gcf, 'Type', 'axes'); % Find all axes objects in the current figure
    for j = 1:length(ax)
        ax(j).XLim = [0, numEpochs];
        ax(j).XTick = 0:100:numEpochs; 
    end
end

% --- 3.3: Add Legends to Combined Plots ---
figure(figure_lr_adam_accuracy); legend('Location', 'best'); 
figure(figure_lr_adam_loss); legend('Location', 'best');
figure(figure_adam_sgd_accuracy); legend('Location', 'best'); 
figure(figure_adam_sgd_loss); legend('Location', 'best');
figure(figure_batchSize_adam_accuracy); legend('Location', 'best'); 
figure(figure_batchSize_adam_loss); legend('Location', 'best');

%% Section 4: Validation Accuracy Table - Comprehensive Output
fprintf('\n--- Section 4: Validation Accuracy Comparison ---\n');
fprintf('Experiment | Optimizer | Learning Rate | Batch Size | Validation Accuracy\n');
fprintf('-----------|-----------|---------------|------------|---------------------\n');
experimentNames = fieldnames(results);
for i = 1:length(experimentNames)
    expName = experimentNames{i};
    fprintf('   %s    |    %s     |      %g     |      %g    |        %.2f%%\n', ...
        expName, results.(expName).optimizer, results.(expName).learningRate, results.(expName).batchSize, results.(expName).validationAccuracy * 100);
end

%% --- Helper Function: runExperiment ---
function results = runExperiment(results, expName, optimizer, learningRate, batchSize, beta1, beta2, epsilon, numEpochs, printInterval, dataFilename, trainRatio, seed)
    % --- Section 1: Data Loading and Preprocessing ---
    [X, y] = loadIrisData(dataFilename);
    X = (X - mean(X)) ./ std(X);
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

    % --- Section 2: Network Initialization ---
    inputLayerSize = size(X_train, 2);
    hiddenLayer1Size = 5;
    hiddenLayer2Size = 3;
    outputLayerSize = size(y_train, 2);
    W1 = randn(inputLayerSize, hiddenLayer1Size) * sqrt(2 / (inputLayerSize + hiddenLayer1Size));
    b1 = zeros(1, hiddenLayer1Size);
    W2 = randn(hiddenLayer1Size, hiddenLayer2Size) * sqrt(2 / (hiddenLayer1Size + hiddenLayer2Size));
    b2 = zeros(1, hiddenLayer2Size);
    W3 = randn(hiddenLayer2Size, outputLayerSize) * sqrt(2 / (hiddenLayer2Size + outputLayerSize));
    b3 = zeros(1, outputLayerSize);

    % Initialize moment vectors for ADAM
    m_W1 = zeros(size(W1)); v_W1 = zeros(size(W1));
    m_b1 = zeros(size(b1)); v_b1 = zeros(size(b1));
    m_W2 = zeros(size(W2)); v_W2 = zeros(size(W2));
    m_b2 = zeros(size(b2)); v_b2 = zeros(size(b2));
    m_W3 = zeros(size(W3)); v_W3 = zeros(size(W3));
    m_b3 = zeros(size(b3)); v_b3 = zeros(size(b3));

    trainLossHistory = zeros(numEpochs, 1);
    trainAccuracyHistory = zeros(numEpochs, 1);
    numBatches = floor(size(X_train, 1) / batchSize);
    t = 0; % Initialize time step for ADAM

    % --- Section 3: Training Loop (ADAM or SGD) ---
    for epoch = 1:numEpochs
        shuffleIdx = randperm(size(X_train, 1));
        X_train = X_train(shuffleIdx, :);
        y_train = y_train(shuffleIdx, :);
        epochLoss = 0; correctPredictions = 0;

        for batch = 1:numBatches
            t = t + 1;
            batchStart = (batch - 1) * batchSize + 1;
            batchEnd = min(batch * batchSize, size(X_train, 1));
            X_batch = X_train(batchStart:batchEnd, :);
            y_batch = y_train(batchStart:batchEnd, :);

            % Forward propagation
            z1 = X_batch * W1 + b1;
            a1 = tanh(z1);
            z2 = a1 * W2 + b2;
            a2 = tanh(z2);
            z3 = a2 * W3 + b3;
            a3 = tanh(z3);
            batchLoss = mean(sum((a3 - y_batch).^2, 2));
            epochLoss = epochLoss + batchLoss;
            [~, predictedLabels] = max(a3, [], 2);
            [~, trueLabels] = max(y_batch, [], 2);
            correctPredictions = correctPredictions + sum(predictedLabels == trueLabels);

            % Backpropagation
            delta3 = (a3 - y_batch) .* (1 - a3.^2);
            delta2 = (delta3 * W3') .* (1 - a2.^2);
            delta1 = (delta2 * W2') .* (1 - a1.^2);
            
            dW1 = X_batch' * delta1;
            db1 = sum(delta1, 1);
            dW2 = a1' * delta2;
            db2 = sum(delta2, 1);
            dW3 = a2' * delta3;
            db3 = sum(delta3, 1);

            % --- Optimizer Update ---
            if strcmp(optimizer, 'adam')
                % ADAM updates
                m_W1 = beta1 * m_W1 + (1 - beta1) * dW1;
                v_W1 = beta2 * v_W1 + (1 - beta2) * dW1.^2;
                m_hat_W1 = m_W1 / (1 - beta1^t);
                v_hat_W1 = v_W1 / (1 - beta2^t);
                W1 = W1 - learningRate * m_hat_W1 ./ (sqrt(v_hat_W1) + epsilon);

                m_b1 = beta1 * m_b1 + (1 - beta1) * db1;
                v_b1 = beta2 * v_b1 + (1 - beta2) * db1.^2;
                m_hat_b1 = m_b1 / (1 - beta1^t);
                v_hat_b1 = v_b1 / (1 - beta2^t);
                b1 = b1 - learningRate * m_hat_b1 ./ (sqrt(v_hat_b1) + epsilon);

                m_W2 = beta1 * m_W2 + (1 - beta1) * dW2;
                v_W2 = beta2 * v_W2 + (1 - beta2) * dW2.^2;
                m_hat_W2 = m_W2 / (1 - beta1^t);
                v_hat_W2 = v_W2 / (1 - beta2^t);
                W2 = W2 - learningRate * m_hat_W2 ./ (sqrt(v_hat_W2) + epsilon);

                m_b2 = beta1 * m_b2 + (1 - beta1) * db2;
                v_b2 = beta2 * v_b2 + (1 - beta2) * db2.^2;
                m_hat_b2 = m_b2 / (1 - beta1^t);
                v_hat_b2 = v_b2 / (1 - beta2^t);
                b2 = b2 - learningRate * m_hat_b2 ./ (sqrt(v_hat_b2) + epsilon);

                m_W3 = beta1 * m_W3 + (1 - beta1) * dW3;
                v_W3 = beta2 * v_W3 + (1 - beta2) * dW3.^2;
                m_hat_W3 = m_W3 / (1 - beta1^t);
                v_hat_W3 = v_W3 / (1 - beta2^t);
                W3 = W3 - learningRate * m_hat_W3 ./ (sqrt(v_hat_W3) + epsilon);

                m_b3 = beta1 * m_b3 + (1 - beta1) * db3;
                v_b3 = beta2 * v_b3 + (1 - beta2) * db3.^2;
                m_hat_b3 = m_b3 / (1 - beta1^t);
                v_hat_b3 = v_b3 / (1 - beta2^t);
                b3 = b3 - learningRate * m_hat_b3 ./ (sqrt(v_hat_b3) + epsilon);

            elseif strcmp(optimizer, 'sgd')
                % SGD updates
                W1 = W1 - learningRate * dW1;
                b1 = b1 - learningRate * db1;
                W2 = W2 - learningRate * dW2;
                b2 = b2 - learningRate * db2;
                W3 = W3 - learningRate * dW3;
                b3 = b3 - learningRate * db3;
            end
        end

        trainLossHistory(epoch) = epochLoss / numBatches;
        trainAccuracyHistory(epoch) = correctPredictions / size(X_train, 1);
        if mod(epoch, printInterval) == 0
            fprintf('Epoch %d: Loss = %.4f, Training Accuracy = %.2f%%\n', epoch, trainLossHistory(epoch), trainAccuracyHistory(epoch) * 100);
        end
    end

    % --- Section 4: Validation ---
    z1_val = X_val * W1 + b1;
    a1_val = tanh(z1_val);
    z2_val = a1_val * W2 + b2;
    a2_val = tanh(z2_val);
    z3_val = a2_val * W3 + b3;
    a3_val = tanh(z3_val);
    [~, predictedValLabels] = max(a3_val, [], 2);
    [~, trueValLabels] = max(y_val, [], 2);
    validationAccuracy = mean(predictedValLabels == trueValLabels);

    % Store Results
    results.(expName).optimizer = optimizer;
    results.(expName).learningRate = learningRate;
    results.(expName).batchSize = batchSize;
    results.(expName).trainLossHistory = trainLossHistory;
    results.(expName).trainAccuracyHistory = trainAccuracyHistory;
    results.(expName).validationAccuracy = validationAccuracy;
end
