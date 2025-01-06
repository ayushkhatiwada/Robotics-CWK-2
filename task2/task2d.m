%% ELEC0144 - Machine Learning for Robotics
% Task 2: Classification
% Part (d): Exploring Network Architectures and Activation Functions with MATLAB Toolbox

%% Configuration Section (All hard-coded values here)
% Dataset Parameters
dataFilename = 'IrisData.txt';
trainRatio = 0.7; % This will be varied in the experiment
seed = 42; % Random seed for reproducibility

% General Training Parameters
maxEpochs = 100;
performanceGoal = 1e-6; % Mean squared error goal

%% Section 1: Data Loading and Preprocessing

% Load Iris data from file (function defined below)
[X, y] = loadIrisData(dataFilename);

% Normalize features using z-score normalization
X = (X - mean(X,1)) ./ std(X,0,1); %Corrected Normalization

% Randomize data order
rng(seed);
n = size(X, 1); % Corrected to use the number of rows of X
randomIdx = randperm(n);
X = X(randomIdx, :);
y = y(randomIdx, :);

% Split data into training and validation sets
splitPoint = floor(trainRatio * n);
X_train = X(1:splitPoint, :);
y_train = y(1:splitPoint, :);
X_val = X(splitPoint + 1:end, :);
y_val = y(splitPoint + 1:end, :);

% Transpose data for the Neural Network Toolbox
X_train = X_train';
y_train = y_train';
X_val = X_val';
y_val = y_val';

%% Section 2: Experiment Setup

% Define network architectures and training parameters to explore
% Each element in the cell array represents a different experiment
architectures = {
    % Original architecture (4-5-3-3) with tanh
    {patternnet([5 3]), ...
    {'trainFcn', 'traingdm', 'goal', performanceGoal, 'epochs', maxEpochs, 'showWindow', false}, ...
    {'tansig', 'tansig', 'softmax'}, ...
    'Original (4-5-3-3), Tanh'}, ...

    % Smaller architecture (4-3-3) with ReLU
    {patternnet(3), ...
    {'trainFcn', 'traingdm', 'goal', performanceGoal, 'epochs', maxEpochs, 'showWindow', false}, ...
    {'poslin', 'softmax'}, ...
    'Smaller (4-3-3), ReLU'}, ...

    % Larger architecture (4-10-5-3) with mixed activation functions
    {patternnet([10 5]), ...
    {'trainFcn', 'traingdm', 'goal', performanceGoal, 'epochs', maxEpochs, 'showWindow', false}, ...
    {'poslin', 'tansig', 'softmax'}, ...
    'Larger (4-10-5-3), Mixed'}, ...

    % Deeper architecture (4-5-5-5-3) with tanh
    {patternnet([5 5 5]), ...
    {'trainFcn', 'traingdm', 'goal', performanceGoal, 'epochs', maxEpochs, 'showWindow', false}, ...
    {'tansig', 'tansig', 'tansig', 'softmax'}, ...
    'Deeper (4-5-5-5-3), Tanh'}, ...

    % Architecture with more neurons in the first hidden layer (4-20-3) with ReLU
    {patternnet([20 3]), ...
    {'trainFcn', 'traingdm', 'goal', performanceGoal, 'epochs', maxEpochs, 'showWindow', false}, ...
    {'poslin', 'poslin', 'softmax'}, ...
    'Wider First Layer (4-20-3), ReLU'},...

    % Deeper architecture (4-5-5-5-3) with trainscg
    {patternnet([5 5 5]), ...
    {'trainFcn', 'trainscg', 'goal', performanceGoal, 'epochs', maxEpochs, 'showWindow', false}, ...
    {'tansig', 'tansig', 'tansig', 'softmax'}, ...
    'Deeper (4-5-5-5-3), Tanh, trainscg'}, ...
};

numExperiments = length(architectures);

%% Section 3: Training and Evaluation

% Create a figure for plotting training performance
figure('Position', [100, 100, 900, 600]);

% Store validation accuracies for each experiment
validationAccuracies = zeros(1, numExperiments);

for i = 1:numExperiments
    % Get network, training parameters, activation functions, and name
    net = architectures{i}{1};
    trainParams = architectures{i}{2};
    activationFcns = architectures{i}{3};
    experimentName = architectures{i}{4};

    % Set training function
    net.trainFcn = trainParams{2};

    % Set other training parameters
    for j = 3:2:length(trainParams)
        paramName = trainParams{j};
        paramValue = trainParams{j+1};
        net.trainParam.(paramName) = paramValue;
    end

    % Set activation functions
    for layer = 1:length(net.layers)
        if layer <= length(activationFcns)
            net.layers{layer}.transferFcn = activationFcns{layer};
        end
    end

    % Configure and train the network
    net = configure(net, X_train, y_train);
    [net, tr] = train(net, X_train, y_train);

    % Plot training performance (e.g., MSE)
    subplot(2, ceil(numExperiments/2), i);
    plot(tr.epoch, tr.perf, 'LineWidth', 2);
    xlabel('Epoch');
    ylabel('Performance (MSE)');
    title(experimentName);

    % Test the network on the validation set
    y_pred = net(X_val);
    validationAccuracy = mean(vec2ind(y_pred) == vec2ind(y_val));
    fprintf('%s - Validation Accuracy: %.2f%%\n', experimentName, validationAccuracy * 100);
    validationAccuracies(i) = validationAccuracy;
end

% Adjust figure layout
sgtitle('Neural Network Training Performance Comparison');

% Correctly extract experiment names from the cell array
experimentNames = cell(1, numExperiments);
for i = 1:numExperiments
    experimentNames{i} = architectures{i}{4};
end

% Display validation accuracies in a table
accuracyTable = table(experimentNames', validationAccuracies', 'VariableNames', {'Architecture', 'Validation Accuracy'});
disp(accuracyTable);

%% Section 4: Experiment with Varying trainRatio
trainRatios = 0.5:0.1:0.9; % Test with train ratios from 0.5 to 0.9
numRatios = length(trainRatios);

% Store results for each train ratio and each architecture
trainRatioResults = cell(numExperiments, numRatios);

for i = 1:numExperiments
    net = architectures{i}{1};
    trainParams = architectures{i}{2};
    activationFcns = architectures{i}{3};
    experimentName = architectures{i}{4};
    net.trainFcn = trainParams{2};
    for j = 3:2:length(trainParams)
        paramName = trainParams{j};
        paramValue = trainParams{j+1};
        net.trainParam.(paramName) = paramValue;
    end
    for layer = 1:length(net.layers)
        if layer <= length(activationFcns)
            net.layers{layer}.transferFcn = activationFcns{layer};
        end
    end

    for r = 1:numRatios
        currentTrainRatio = trainRatios(r);
        
        % Split data based on the current train ratio
        splitPoint = floor(currentTrainRatio * n);
        X_train = X(1:splitPoint, :);
        y_train = y(1:splitPoint, :);
        X_val = X(splitPoint + 1:end, :);
        y_val = y(splitPoint + 1:end, :);

        % Transpose data
        X_train = X_train';
        y_train = y_train';
        X_val = X_val';
        y_val = y_val';
        
        % Train the network
        net = configure(net, X_train, y_train);
        [net, tr] = train(net, X_train, y_train);

        % Evaluate the network
        y_train_pred = net(X_train);
        y_val_pred = net(X_val);
        
        trainAccuracy = mean(vec2ind(y_train_pred) == vec2ind(y_train));
        valAccuracy = mean(vec2ind(y_val_pred) == vec2ind(y_val));
        trainLoss = tr.perf(end); % Get the final training loss
        
        % Store the results
        trainRatioResults{i, r} = {trainAccuracy, valAccuracy, trainLoss};
        
        fprintf('Architecture: %s, Train Ratio: %.1f, Train Accuracy: %.2f%%, Validation Accuracy: %.2f%%, Loss: %.4f\n', ...
                experimentName, currentTrainRatio, trainAccuracy * 100, valAccuracy * 100, trainLoss);
    end
end

% Display results in a table for each architecture
for i = 1:numExperiments
    experimentName = architectures{i}{4};
    fprintf('\nResults for %s:\n', experimentName);
    
    % Prepare data for table
    ratioTable = table('Size', [numRatios 4], ...
                       'VariableTypes', {'double', 'double', 'double', 'double'}, ...
                       'VariableNames', {'Train Ratio', 'Train Accuracy', 'Validation Accuracy', 'Train Loss'});
    
    for r = 1:numRatios
        ratioTable(r, :) = {trainRatios(r), trainRatioResults{i, r}{1}, trainRatioResults{i, r}{2}, trainRatioResults{i, r}{3}};
    end
    
    disp(ratioTable);
end

%% Helper Function: loadIrisData
% Loads Iris data from a file, converts class labels to one-hot encoded vectors.
function [X, y] = loadIrisData(filename)
    % Open the file
    fileID = fopen(filename, 'r');
    if fileID == -1
        error('Could not open file %s', filename);
    end

    % Initialize data and label matrices
    X = []; % Iris Data Features
    y = []; % One-hot encoded labels

    % Read the file line by line
    while ~feof(fileID)
        currentLine = fgetl(fileID);
        if ischar(currentLine) && ~isempty(currentLine)
            % Split the line by commas
            lineParts = strsplit(currentLine, ',');

            % Convert the first 4 parts to numerical features
            irisFeatures = str2double(lineParts(1:4));

            % Append features to the data matrix
            X = [X; irisFeatures];

            % Create one-hot encoded vector based on the class label
            if contains(lineParts{5}, 'setosa')
                oneHotLabel = [1; 0; 0]; % One-hot encoding
            elseif contains(lineParts{5}, 'versicolor')
                oneHotLabel = [0; 1; 0]; % One-hot encoding
            else % virginica
                oneHotLabel = [0; 0; 1]; % One-hot encoding
            end

            % Append the one-hot encoded label to the label matrix
            y = [y; oneHotLabel];
        end
    end

    % Close the file
    fclose(fileID);
end