
%% ELEC0144 - Machine Learning for Robotics - Assignment 2
% Task 2b: Classification - Focused Visualizations (Initialization, Normalization, Learning Rate) - CORRECTED

%% Configuration Section (Hard-coded values and experiment parameters)
% --------------------------------------------------------------------------
% Dataset Parameters
dataFilename = 'IrisData.txt';      % Filename of the Iris dataset
trainRatio = 0.7;                 % Ratio of data for training
randomSeed = 42;                  % Random seed for reproducibility

% Network Architecture Parameters (Fixed for Task 2b)
inputLayerSize = 4;
hiddenLayer1Size = 5;
hiddenLayer2Size = 3;
outputLayerSize = 3;

% Training Hyperparameters (Experiment Parameters)
learningRatesToTest = [0.1, 0.01, 0.001, 0.0001]; % Learning rates to explore
numEpochs = 1000;                 % Number of training epochs (fixed for comparison)
printInterval = 100;               % Print frequency
useNormalization = [true, false];     % true: with normalization, false: without
initializationMethods = {'xavier', 'random'}; % Initialization methods to explore

% Activation Function (tanh - fixed as per task)
activationFunction = @tanh;
activationDerivative = @(x) 1 - x.^2;

% Results Storage
results = struct(); % Structure to store results for each experiment

% --------------------------------------------------------------------------
%% Section 1: Data Loading and Preprocessing (Function remains the same)
% --------------------------------------------------------------------------
function [features, labels] = loadIrisData(filename)
    fileID = fopen(filename, 'r');
    if fileID == -1, error('Could not open file: %s', filename); end
    features = []; labels = [];
    while ~feof(fileID)
        line = fgetl(fileID);
        if ischar(line) && ~isempty(line)
            parts = strsplit(line, ',');
            irisFeatures = str2double(parts(1:4));
            features = [features; irisFeatures];
            species = parts{5};
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
experimentCounter = 1; % Counter for experiments

for initMethod_str = initializationMethods % Loop through initialization methods
    for norm_flag = useNormalization     % Loop through normalization options
        for lr = learningRatesToTest       % Loop through learning rates

            fprintf('\n--- Starting Experiment %d ---\n', experimentCounter);
            fprintf('Initialization: %s, Normalization: %s, Learning Rate: %f\n', ...
                initMethod_str{1}, mat2str(norm_flag), lr);

            lr_str_for_name = strrep(sprintf('%g', lr), '.', '_'); % Replace dot with underscore for field name
            currentExperimentName = sprintf('Exp%d_Init_%s_Norm_%s_LR_%s', experimentCounter, initMethod_str{1}, mat2str(norm_flag), lr_str_for_name);
            results.(currentExperimentName) = struct(); % Initialize struct for current experiment

            %% Section 1: Data Loading and Preprocessing (Inside Experiment Loop)
            [features, labels] = loadIrisData(dataFilename);

            % Normalization Conditional
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

            rng(randomSeed);
            numSamples = size(X, 1);
            randomIndexOrder = randperm(numSamples);
            shuffledFeatures = X(randomIndexOrder, :);
            shuffledLabels = labels(randomIndexOrder, :);

            splitPoint = floor(trainRatio * numSamples);
            X_train = shuffledFeatures(1:splitPoint, :);
            y_train = shuffledLabels(1:splitPoint, :);
            X_val = shuffledFeatures(splitPoint + 1:end, :);
            y_val = shuffledLabels(splitPoint + 1:end, :);


            %% Section 2: Network Initialization (Inside Experiment Loop)
            if strcmp(initMethod_str{1}, 'xavier')
                W1 = randn(inputLayerSize, hiddenLayer1Size) * sqrt(2 / (inputLayerSize + hiddenLayer1Size));
                W2 = randn(hiddenLayer1Size, hiddenLayer2Size) * sqrt(2 / (hiddenLayer1Size + hiddenLayer2Size));
                W3 = randn(hiddenLayer2Size, outputLayerSize) * sqrt(2 / (hiddenLayer2Size + outputLayerSize));
            elseif strcmp(initMethod_str{1}, 'random')
                W1 = randn(inputLayerSize, hiddenLayer1Size) * 0.01;
                W2 = randn(hiddenLayer1Size, hiddenLayer2Size) * 0.01;
                W3 = randn(hiddenLayer2Size, outputLayerSize) * 0.01;
            else
                error('Invalid initialization method.');
            end
            b1 = zeros(1, hiddenLayer1Size);
            b2 = zeros(1, hiddenLayer2Size);
            b3 = zeros(1, outputLayerSize);


            trainLossHistory = zeros(1, numEpochs / printInterval);
            trainAccuracyHistory = zeros(1, numEpochs / printInterval);


            %% Section 3: Training Loop (SGD) (Inside Experiment Loop)
            currentLearningRate = lr; % Use the current learning rate from the loop

            for epoch = 1:numEpochs
                shuffleIndicesEpoch = randperm(size(X_train, 1));
                X_train = X_train(shuffleIndicesEpoch, :);
                y_train = y_train(shuffleIndicesEpoch, :);

                for i = 1:size(X_train, 1)
                    sample_X = X_train(i, :);
                    sample_y = y_train(i, :);

                    % Forward Propagation
                    z1 = sample_X * W1 + b1;         a1 = activationFunction(z1);
                    z2 = a1 * W2 + b2;              a2 = activationFunction(z2);
                    z3 = a2 * W3 + b3;              a3 = activationFunction(z3);

                    % Backpropagation
                    delta3 = (sample_y - a3) .* activationDerivative(a3);
                    delta2 = (delta3 * W3') .* activationDerivative(a2);
                    delta1 = (delta2 * W2') .* activationDerivative(a1);

                    % Gradient Descent Update (using currentLearningRate)
                    W3 = W3 + currentLearningRate * (a2' * delta3);
                    b3 = b3 + currentLearningRate * delta3;
                    W2 = W2 + currentLearningRate * (a1' * delta2);
                    b2 = b2 + currentLearningRate * delta2;
                    W1 = W1 + currentLearningRate * (sample_X' * delta1);
                    b1 = b1 + currentLearningRate * delta1;
                end

                if mod(epoch, printInterval) == 0
                    % Training Accuracy and Loss Calculation
                    z1_train = X_train * W1 + repmat(b1, size(X_train, 1), 1); a1_train = activationFunction(z1_train);
                    z2_train = a1_train * W2 + repmat(b2, size(X_train, 1), 1); a2_train = activationFunction(z2_train);
                    z3_train = a2_train * W3 + repmat(b3, size(X_train, 1), 1); a3_train = activationFunction(z3_train);

                    [~, predictedTrainLabels] = max(a3_train, [], 2);
                    [~, trueTrainLabels] = max(y_train, [], 2);
                    trainingAccuracy = mean(predictedTrainLabels == trueTrainLabels);
                    trainLoss = mean(sum((y_train - a3_train).^2, 2));

                    historyIndex = epoch / printInterval;
                    trainLossHistory(historyIndex) = trainLoss;
                    trainAccuracyHistory(historyIndex) = trainingAccuracy;

                    fprintf('  Epoch %d: Training Accuracy = %.2f%%, Loss = %.4f\n', epoch, trainingAccuracy * 100, trainLoss);
                end
            end

            %% Section 4: Validation (Inside Experiment Loop)
            z1_val = X_val * W1 + repmat(b1, size(X_val, 1), 1); a1_val = activationFunction(z1_val);
            z2_val = a1_val * W2 + repmat(b2, size(X_val, 1), 1); a2_val = activationFunction(z2_val);
            z3_val = a2_val * W3 + repmat(b3, size(X_val, 1), 1); a3_val = activationFunction(z3_val);

            [~, predictedValLabels] = max(a3_val, [], 2);
            [~, trueValLabels] = max(y_val, [], 2);
            validationAccuracy = mean(predictedValLabels == trueValLabels);
            fprintf('  Validation Accuracy (Experiment %d): %.2f%%\n', experimentCounter, validationAccuracy * 100);


            %% Store Results for this Experiment
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

%% Section 5: Plotting and Analysis of Experiments - REDUCED & FOCUSED PLOTS

% --- Create Essential Combined Figures BEFORE Experiment Loop ---
% Learning Rate Sensitivity (with Xavier Initialization and Normalization ON) - ESSENTIAL
figure1_lr_normOn_xavier_loss = figure('Name', 'Training Loss: Learning Rate Sensitivity (Norm=On, Init=Xavier)', 'Position', [100, 100, 800, 600]); subplot(1, 1, 1); hold on; title('Training Loss: Learning Rate Sensitivity (Xavier Init, Norm=On)'); xlabel('Epoch'); ylabel('Loss'); grid on;
figure2_lr_normOn_xavier_accuracy = figure('Name', 'Training Accuracy: Learning Rate Sensitivity (Norm=On, Init=Xavier)', 'Position', [100, 100, 800, 600]); subplot(1, 1, 1); hold on; title('Training Accuracy: Learning Rate Sensitivity (Xavier Init, Norm=On)'); xlabel('Epoch'); ylabel('Accuracy'); ylim([0, 1.05]); grid on;

% Normalization Effect (with Xavier Initialization and LR=0.01) - ESSENTIAL
figure3_norm_xavier_loss = figure('Name', 'Training Loss: Normalization Effect (Init=Xavier, LR=0.01)', 'Position', [100, 100, 800, 600]); subplot(1, 1, 1); hold on; title('Training Loss: Normalization Effect (Xavier Init, LR=0.01)'); xlabel('Epoch'); ylabel('Loss'); grid on;
figure4_norm_xavier_accuracy = figure('Name', 'Training Accuracy: Normalization Effect (Init=Xavier, LR=0.01)', 'Position', [100, 100, 800, 600]); subplot(1, 1, 1); hold on; title('Training Accuracy: Normalization Effect (Xavier Init, LR=0.01)'); xlabel('Epoch'); ylabel('Accuracy'); ylim([0, 1.05]); grid on;

% --- NEW: Initialization Method Comparison (LR=0.01, Norm ON & OFF) ---
figure5_init_lr001_normOn_accuracy = figure('Name', 'Training Accuracy: Init. Comparison (LR=0.01, Norm=On)', 'Position', [100, 100, 800, 600]); subplot(1, 1, 1); hold on; title('Training Accuracy: Initialization Comparison (LR=0.01, Norm=On)'); xlabel('Epoch'); ylabel('Accuracy'); ylim([0, 1.05]); grid on;
figure6_init_lr001_normOff_accuracy = figure('Name', 'Training Accuracy: Init. Comparison (LR=0.01, Norm=Off)', 'Position', [100, 100, 800, 600]); subplot(1, 1, 1); hold on; title('Training Accuracy: Initialization Comparison (LR=0.01, Norm=Off)'); xlabel('Epoch'); ylabel('Accuracy'); ylim([0, 1.05]); grid on;


experimentNames = fieldnames(results); % Get all experiment names

% --- Plotting Learning Curves on Combined Figures and Selective Confusion Matrices ---
for i = 1:length(experimentNames)
    expName = experimentNames{i};
    expResult = results.(expName);

    % --- Populate plot lines for combined Learning Rate Sensitivity (Xavier, Norm ON) ---
    if contains(expName, 'Norm_true') && contains(expName, 'Init_xavier')
        if contains(expName, 'LR_')
            if contains(expName, 'LR_0_1')     figure(figure1_lr_normOn_xavier_loss); plot_handle = plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate));  figure(figure2_lr_normOn_xavier_accuracy); plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate)); end
            if contains(expName, 'LR_0_01')    figure(figure1_lr_normOn_xavier_loss); plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate));  figure(figure2_lr_normOn_xavier_accuracy); plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate)); end
            if contains(expName, 'LR_0_001')   figure(figure1_lr_normOn_xavier_loss); plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate));  figure(figure2_lr_normOn_xavier_accuracy); plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate)); end
            if contains(expName, 'LR_0_0001')  figure(figure1_lr_normOn_xavier_loss); plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate));  figure(figure2_lr_normOn_xavier_accuracy); plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', sprintf('LR = %g', expResult.learningRate)); end
        end
    end

    % --- Populate plot lines for combined Normalization Effect (Xavier Init, LR=0.01) ---
    if contains(expName, 'LR_0_01') && contains(expName, 'Init_xavier')
        if contains(expName, 'Norm_true')  figure(figure3_norm_xavier_loss); plot_handle = plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', 'With Norm');  figure(figure4_norm_xavier_accuracy); plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'With Norm'); end
        if contains(expName, 'Norm_false') figure(figure3_norm_xavier_loss); plot_handle = plot(printInterval:printInterval:numEpochs, expResult.trainLossHistory, 'DisplayName', 'Without Norm'); figure(figure4_norm_xavier_accuracy); plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'Without Norm'); end
    end

     % --- NEW: Populate plot lines for combined Initialization Comparison (LR=0.01) ---
    if contains(expName, 'LR_0_001')
        if contains(expName, 'Norm_true')
            if contains(expName, 'Init_xavier') figure(figure5_init_lr001_normOn_accuracy); plot_handle = plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'Xavier Init'); end
            if contains(expName, 'Init_random') figure(figure5_init_lr001_normOn_accuracy); plot_handle = plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'Random Init'); end
        end
        if contains(expName, 'Norm_false')
            if contains(expName, 'Init_xavier') figure(figure6_init_lr001_normOff_accuracy); plot_handle = plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'Xavier Init'); end
            if contains(expName, 'Init_random') figure(figure6_init_lr001_normOff_accuracy); plot_handle = plot(printInterval:printInterval:numEpochs, expResult.trainAccuracyHistory, 'DisplayName', 'Random Init'); end
        end
    end


    % --- Selective Confusion Matrix Plots (Best and Worst cases - examples) ---
    if strcmp(expName, 'Exp6_Init_xavier_Norm_true_LR_0_01') % Example: Best case - adjust based on your results
        figure_confusion_matrix_best = figure('Name', ['Confusion Matrix - Best Case: ' expName], 'Position', [700, 100, 600, 400]);
        confusionMatrix_bestCase = confusionmat(expResult.trueValLabels, expResult.predictedValLabels);
        confusionchart(confusionMatrix_bestCase, {'Setosa', 'Versicolor', 'Virginica'}, ...
            'Title', ['Validation Confusion Matrix - Best Case: ' expName], ...
            'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
        results.(expName).confusionMatrixFigureHandle_best = figure_confusion_matrix_best; % Store handle
    elseif strcmp(expName, 'Exp16_Init_random_Norm_false_LR_0_0001') % Example: Worst case - adjust
        figure_confusion_matrix_worst = figure('Name', ['Confusion Matrix - Worst Case: ' expName], 'Position', [700, 500, 600, 400]);
        confusionMatrix_worstCase = confusionmat(expResult.trueValLabels, expResult.predictedValLabels);
        confusionchart(confusionMatrix_worstCase, {'Setosa', 'Versicolor', 'Virginica'}, ...
            'Title', ['Validation Confusion Matrix - Worst Case: ' expName], ...
            'RowSummary', 'row-normalized', 'ColumnSummary', 'column-normalized');
        results.(expName).confusionMatrixFigureHandle_worst = figure_confusion_matrix_worst; % Store handle
    end

    % --- (Optional) Keep individual learning curve data stored in results, but don't plot all ---
    results.(expName).trainAccuracyHistory = expResult.trainAccuracyHistory; % Data is still stored
    results.(expName).trainLossHistory = expResult.trainLossHistory;         % Data is still stored

end % Experiment Loop End


% Add legends to combined plots AFTER the loop
figure(figure1_lr_normOn_xavier_loss); legend('Location', 'best');
figure(figure2_lr_normOn_xavier_accuracy); legend('Location', 'best');
figure(figure3_norm_xavier_loss); legend('Location', 'best');
figure(figure4_norm_xavier_accuracy); legend('Location', 'best');
figure(figure5_init_lr001_normOn_accuracy); legend('Location', 'best');
figure(figure6_init_lr001_normOff_accuracy); legend('Location', 'best');


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


fprintf('\nFocused experiments and essential plots generated.\n');