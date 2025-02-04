%% --- Configuration Section (Hard-coded values and experiment parameters) ---

% Dataset Parameters
dataFilename = 'IrisData.txt';      % Filename of the Iris dataset
trainRatio = 0.7;                 % Ratio of data for training
randomSeed = 42;                  % Random seed for reproducibility

% Network Architecture Parameters
inputLayerSize = 4;               % Number of features in the Iris dataset
hiddenLayer1Size = 5;             % Number of neurons in the first hidden layer
hiddenLayer2Size = 3;             % Number of neurons in the second hidden layer
outputLayerSize = 3;              % Number of output classes (Iris species)

% Training Hyperparameters (Experiment Parameters)
learningRatesToTest = [0.1, 0.01, 0.001, 0.0001]; % Learning rates to explore
numEpochs = 1000;                                 % Number of training epochs (fixed)
printInterval = 100;                              % Print training progress every 'printInterval' epochs
useNormalization = [true, false];                 % Boolean array to control normalization (ON/OFF)
initializationMethods = {'xavier', 'random'};     % Initialization methods: 'xavier' or 'random'

% Activation Function
activationFunction = @tanh;
activationDerivative = @(x) 1 - x.^2;

% Results Storage
results = struct(); % Structure to store results for each experiment

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

%% --- Experiment Loop ---
experimentCounter = 1; % Initialize experiment counter

% Loop through all combinations of initialization methods, normalization options, and learning rates
for initMethod_str = initializationMethods
    for norm_flag = useNormalization
        for lr = learningRatesToTest

            % Display experiment settings
            fprintf('\n--- Starting Experiment %d ---\n', experimentCounter);
            fprintf('Initialization: %s, Normalization: %s, Learning Rate: %f\n', ...
                initMethod_str{1}, mat2str(norm_flag), lr);

            % Create a unique name for storing the results of this experiment
            lr_str_for_name = strrep(sprintf('%g', lr), '.', '_'); % Replace '.' with '_' for valid field names
            currentExperimentName = sprintf('Exp%d_Init_%s_Norm_%s_LR_%s', experimentCounter, initMethod_str{1}, mat2str(norm_flag), lr_str_for_name);
            results.(currentExperimentName) = struct(); % Initialize a structure to store results

            % --- Data Loading and Preprocessing ---
            [features, labels] = loadIrisData(dataFilename);

            % --- Data Normalization (Z-score) ---
            if norm_flag
                featureMeans = mean(features);
                featureStDevs = std(features);
                normalizedFeatures = (features - featureMeans) ./ featureStDevs;
                X = normalizedFeatures;
                fprintf('  Z-score Normalization: ON\n');
            else
                X = features; % No normalization
                fprintf('  Z-score Normalization: OFF\n');
            end

            % --- Data Shuffling and Splitting ---
            rng(randomSeed); % Set random seed for reproducibility
            numSamples = size(X, 1);
            randomIndexOrder = randperm(numSamples); % Generate a random permutation of indices
            shuffledFeatures = X(randomIndexOrder, :);
            shuffledLabels = labels(randomIndexOrder, :);

            % Split data into training and validation sets
            splitPoint = floor(trainRatio * numSamples);
            X_train = shuffledFeatures(1:splitPoint, :);
            y_train = shuffledLabels(1:splitPoint, :);
            X_val = shuffledFeatures(splitPoint + 1:end, :);
            y_val = shuffledLabels(splitPoint + 1:end, :);

            % --- Network Initialization ---
            % Initialize weights and biases based on the selected initialization method
            if strcmp(initMethod_str{1}, 'xavier')
                % Xavier initialization (Glorot initialization)
                W1 = randn(inputLayerSize, hiddenLayer1Size) * sqrt(2 / (inputLayerSize + hiddenLayer1Size));
                W2 = randn(hiddenLayer1Size, hiddenLayer2Size) * sqrt(2 / (hiddenLayer1Size + hiddenLayer2Size));
                W3 = randn(hiddenLayer2Size, outputLayerSize) * sqrt(2 / (hiddenLayer2Size + outputLayerSize));
            elseif strcmp(initMethod_str{1}, 'random')
                % Small random initialization
                W1 = randn(inputLayerSize, hiddenLayer1Size) * 0.01;
                W2 = randn(hiddenLayer1Size, hiddenLayer2Size) * 0.01;
                W3 = randn(hiddenLayer2Size, outputLayerSize) * 0.01;
            else
                error('Invalid initialization method.');
            end
            
            % Initialize biases to zero
            b1 = zeros(1, hiddenLayer1Size);
            b2 = zeros(1, hiddenLayer2Size);
            b3 = zeros(1, outputLayerSize);

            % Initialize arrays to store training loss and accuracy history
            trainLossHistory = zeros(1, numEpochs / printInterval);
            trainAccuracyHistory = zeros(1, numEpochs / printInterval);

            % --- Training Loop (Stochastic Gradient Descent - SGD) ---
            currentLearningRate = lr; % Use the current learning rate from the loop

            for epoch = 1:numEpochs
                % Shuffle the training data for each epoch (SGD)
                shuffleIndicesEpoch = randperm(size(X_train, 1));
                X_train = X_train(shuffleIndicesEpoch, :);
                y_train = y_train(shuffleIndicesEpoch, :);

                % Iterate through each training sample
                for i = 1:size(X_train, 1)
                    sample_X = X_train(i, :);
                    sample_y = y_train(i, :);

                    % --- Forward Propagation ---
                    z1 = sample_X * W1 + b1;
                    a1 = activationFunction(z1);
                    z2 = a1 * W2 + b2;
                    a2 = activationFunction(z2);
                    z3 = a2 * W3 + b3;
                    a3 = activationFunction(z3);

                    % --- Backpropagation ---
                    delta3 = (sample_y - a3) .* activationDerivative(a3);
                    delta2 = (delta3 * W3') .* activationDerivative(a2);
                    delta1 = (delta2 * W2') .* activationDerivative(a1);

                    % --- Gradient Descent Update ---
                    W3 = W3 + currentLearningRate * (a2' * delta3);
                    b3 = b3 + currentLearningRate * delta3;
                    W2 = W2 + currentLearningRate * (a1' * delta2);
                    b2 = b2 + currentLearningRate * delta2;
                    W1 = W1 + currentLearningRate * (sample_X' * delta1);
                    b1 = b1 + currentLearningRate * delta1;
                end

                % --- Calculate and Print Training Progress ---
                if mod(epoch, printInterval) == 0
                    % Calculate training accuracy and loss
                    z1_train = X_train * W1 + repmat(b1, size(X_train, 1), 1);
                    a1_train = activationFunction(z1_train);
                    z2_train = a1_train * W2 + repmat(b2, size(X_train, 1), 1);
                    a2_train = activationFunction(z2_train);
                    z3_train = a2_train * W3 + repmat(b3, size(X_train, 1), 1);
                    a3_train = activationFunction(z3_train);

                    [~, predictedTrainLabels] = max(a3_train, [], 2);
                    [~, trueTrainLabels] = max(y_train, [], 2);
                    trainingAccuracy = mean(predictedTrainLabels == trueTrainLabels);
                    trainLoss = mean(sum((y_train - a3_train).^2, 2));

                    % Store training loss and accuracy
                    historyIndex = epoch / printInterval;
                    trainLossHistory(historyIndex) = trainLoss;
                    trainAccuracyHistory(historyIndex) = trainingAccuracy;

                    fprintf('  Epoch %d: Training Accuracy = %.2f%%, Loss = %.4f\n', epoch, trainingAccuracy * 100, trainLoss);
                end
            end

            % --- Validation ---
            % Perform forward propagation on the validation set
            z1_val = X_val * W1 + repmat(b1, size(X_val, 1), 1);
            a1_val = activationFunction(z1_val);
            z2_val = a1_val * W2 + repmat(b2, size(X_val, 1), 1);
            a2_val = activationFunction(z2_val);
            z3_val = a2_val * W3 + repmat(b3, size(X_val, 1), 1);
            a3_val = activationFunction(z3_val);

            % Calculate validation accuracy
            [~, predictedValLabels] = max(a3_val, [], 2);
            [~, trueValLabels] = max(y_val, [], 2);
            validationAccuracy = mean(predictedValLabels == trueValLabels);
            fprintf('  Validation Accuracy (Experiment %d): %.2f%%\n', experimentCounter, validationAccuracy * 100);

            % --- Store Results for this Experiment ---
            results.(currentExperimentName).initializationMethod = initMethod_str{1};
            results.(currentExperimentName).useNormalization = norm_flag;
            results.(currentExperimentName).learningRate = currentLearningRate;
            results.(currentExperimentName).trainAccuracyHistory = trainAccuracyHistory;
            results.(currentExperimentName).trainLossHistory = trainLossHistory;
            results.(currentExperimentName).validationAccuracy = validationAccuracy;
            results.(currentExperimentName).predictedValLabels = predictedValLabels;
            results.(currentExperimentName).trueValLabels = trueValLabels;

            experimentCounter = experimentCounter + 1; % Increment experiment counter
        end
    end
end

%% --- Plotting and Analysis of Experiments ---

% --- Enhanced Plot Settings for Readability at Small Sizes ---
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontSize', 16);
set(0, 'DefaultLineLineWidth', 3);
set(0, 'DefaultLineMarkerSize', 10);
% --------------------------------------------------------------

% 1. Learning Rate Sensitivity (with Xavier Initialization and Normalization ON)
figure1_lr_normOn_xavier_loss = figure('Name', 'Training Loss: Learning Rate Sensitivity (Norm=On, Init=Xavier)', 'Position', [100, 100, 800, 600]);
subplot(1, 1, 1); hold on;
title('Training Loss: Learning Rate Sensitivity (Xavier Init, Norm=On)');
xlabel('Epoch');
ylabel('Loss');
grid on;

figure2_lr_normOn_xavier_accuracy = figure('Name', 'Training Accuracy: Learning Rate Sensitivity (Norm=On, Init=Xavier)', 'Position', [100, 100, 800, 600]);
subplot(1, 1, 1); hold on;
title('Training Accuracy: Learning Rate Sensitivity (Xavier Init, Norm=On)');
xlabel('Epoch');
ylabel('Accuracy');
ylim([0, 1.05]);
grid on;

% 2. Normalization Effect (with Xavier Initialization and LR=0.01)
figure3_norm_xavier_loss = figure('Name', 'Training Loss: Normalization Effect (Init=Xavier, LR=0.01)', 'Position', [100, 100, 800, 600]);
subplot(1, 1, 1); hold on;
title('Training Loss: Normalization Effect (Xavier Init, LR=0.01)');
xlabel('Epoch');
ylabel('Loss');
grid on;

figure4_norm_xavier_accuracy = figure('Name', 'Training Accuracy: Normalization Effect (Init=Xavier, LR=0.01)', 'Position', [100, 100, 800, 600]);
subplot(1, 1, 1); hold on;
title('Training Accuracy: Normalization Effect (Xavier Init, LR=0.01)');
xlabel('Epoch');
ylabel('Accuracy');
ylim([0, 1.05]);
grid on;

% 3. Initialization Method Comparison (LR=0.001, Norm ON & OFF)
figure5_init_lr001_normOn_loss = figure('Name', 'Training Loss: Init. Comparison (LR=0.001, Norm=On)', 'Position', [100, 100, 800, 600]);
subplot(1, 1, 1); hold on;
title('Training Loss: Initialization Comparison (LR=0.001, Norm=On)');
xlabel('Epoch');
ylabel('Accuracy');
ylim([0, 1.05]);
grid on;

figure6_init_lr001_normOff_accuracy = figure('Name', 'Training Accuracy: Init. Comparison (LR=0.001, Norm=Off)', 'Position', [100, 100, 800, 600]);
subplot(1, 1, 1); hold on;
title('Training Accuracy: Initialization Comparison (LR=0.001, Norm=Off)');
xlabel('Epoch');
ylabel('Accuracy');
ylim([0, 1.05]);
grid on;

% --- Plotting Learning Curves on Combined Figures and Selective Confusion Matrices ---
experimentNames = fieldnames(results); % Get all experiment names

for i = 1:length(experimentNames)
    expName = experimentNames{i};
    expResult = results.(expName);

    % --- Populate plot lines for combined Learning Rate Sensitivity (Xavier, Norm ON) ---
    if contains(expName, 'Norm_true') && contains(expName, 'Init_xavier')
        if contains(expName, 'LR_')
            figure(figure1_lr_normOn_xavier_loss);
            plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate), 'Marker', 'o');
            figure(figure2_lr_normOn_xavier_accuracy);
            plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate), 'Marker', 'o');
        end
    end

    % --- Populate plot lines for combined Normalization Effect (Xavier Init, LR=0.01) ---
    if contains(expName, 'LR_0_01') && contains(expName, 'Init_xavier')
        figure(figure3_norm_xavier_loss);
        if contains(expName, 'Norm_true')
            plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', 'With Norm', 'Marker', 'o');
        else
            plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', 'Without Norm', 'Marker', 'x');
        end
        
        figure(figure4_norm_xavier_accuracy);
        if contains(expName, 'Norm_true')
            plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'With Norm', 'Marker', 'o');
        else
            plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'Without Norm', 'Marker', 'x');
        end
    end

    % --- Populate plot lines for combined Initialization Comparison (LR=0.001) ---
    if contains(expName, 'LR_0_001')
        figure(figure5_init_lr001_normOn_loss);
        if contains(expName, 'Norm_true')
            if contains(expName, 'Init_xavier')
                plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', 'Xavier Init', 'Marker', 'o');
            elseif contains(expName, 'Init_random')
                plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', 'Random Init', 'Marker', 'x');
            end
        end
        
        figure(figure6_init_lr001_normOff_accuracy);
        if contains(expName, 'Norm_false')
            if contains(expName, 'Init_xavier')
                plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'Xavier Init', 'Marker', 'o');
            elseif contains(expName, 'Init_random')
                plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'Random Init', 'Marker', 'x');
            end
        end
    end

    % --- Training Accuracy, Training Loss, and Confusion Matrix for baseline SGD---
    if strcmp(expName, 'Exp1_Init_xavier_Norm_true_LR_0_1') 
        % Training Accuracy and Loss
        figure_exp1_train = figure('Name', 'Exp1: Training Accuracy and Loss', 'Position', [100, 100, 600, 600]);
        
        subplot(2, 1, 1);
        plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'Marker', 'o');
        title('Exp1: Training Accuracy');
        xlabel('Epoch');
        ylabel('Accuracy');
        ylim([0, 1.05]);
        grid on;

        subplot(2, 1, 2);
        plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'Marker', 'o');
        title('Exp1: Training Loss');
        xlabel('Epoch');
        ylabel('Loss');
        grid on;

        % Generate only confusion matrix and increase the font size and make it bigger
        figure_exp1_confusion = figure('Name', 'Exp1: Confusion Matrix', 'Position', [100, 100, 800, 600]);
        confusionchart(expResult.trueValLabels, expResult.predictedValLabels);
        set(findall(gcf,'-property','FontSize'),'FontSize',30);
        set(findall(gcf,'-property','FontWeight'),'FontWeight','bold');
    end
end

% --- Add legends to combined plots AFTER the loop ---
figure(figure1_lr_normOn_xavier_loss); legend('Location', 'best');
figure(figure2_lr_normOn_xavier_accuracy); legend('Location', 'best');
figure(figure3_norm_xavier_loss); legend('Location', 'best');
figure(figure4_norm_xavier_accuracy); legend('Location', 'best');
figure(figure5_init_lr001_normOn_loss); legend('Location', 'best');
figure(figure6_init_lr001_normOff_accuracy); legend('Location', 'best');

% --- Reduce Ticks for Combined Plots ---
allAxes = findall(0, 'Type', 'axes');
for i = 1:numel(allAxes)
    allAxes(i).XTick = 0:200:numEpochs;
end

%% --- Validation Accuracy Comparison Table ---

fprintf('\n--- Validation Accuracy Comparison ---\n');
fprintf('Experiment | Init. Method | Normalization | Learning Rate | Validation Accuracy\n');
fprintf('-----------|---------------|---------------|---------------|---------------------\n');
experimentNames = fieldnames(results);
for i = 1:length(experimentNames)
    expName = experimentNames{i};
    fprintf('   %s    |    %s     |      %s     |      %g     |        %.2f%%\n', ...
        expName, results.(expName).initializationMethod, mat2str(results.(expName).useNormalization), ...
        results.(expName).learningRate, results.(expName).validationAccuracy * 100);
end