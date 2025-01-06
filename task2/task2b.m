%% Configuration Section (All hard-coded values here)
% Dataset Parameters
dataFilename = 'IrisData.txt';
trainRatio = 0.7;
seed = 42;       % Random seed for reproducibility

% Network Architecture
inputLayerSize = 4;
hiddenLayer1Size = 5;
hiddenLayer2Size = 3;
outputLayerSize = 3;

% Training Parameters
learningRate = 0.01;
numEpochs = 1000;
printInterval = 100; % Print accuracy every this many epochs

%% Section 1: Data Loading and Preprocessing

% Load Iris data from file (function defined below)
[X, y] = loadIrisData(dataFilename);

% Normalize features using z-score normalization
X = (X - mean(X)) ./ std(X);

% Randomize data order
rng(seed);
n = size(X, 1);
randomIdx = randperm(n);
X = X(randomIdx, :);
y = y(randomIdx, :);

% Split data into training and validation sets
splitPoint = floor(trainRatio * n);
X_train = X(1:splitPoint, :);
y_train = y(1:splitPoint, :);
X_val = X(splitPoint + 1:end, :);
y_val = y(splitPoint + 1:end, :);

%% Section 2: Network Initialization

% Initialize weights and biases using Xavier initialization
% This helps to prevent vanishing/exploding gradients during training
W1 = randn(inputLayerSize, hiddenLayer1Size) * sqrt(2 / (inputLayerSize + hiddenLayer1Size));
b1 = zeros(1, hiddenLayer1Size);
W2 = randn(hiddenLayer1Size, hiddenLayer2Size) * sqrt(2 / (hiddenLayer1Size + hiddenLayer2Size));
b2 = zeros(1, hiddenLayer2Size);
W3 = randn(hiddenLayer2Size, outputLayerSize) * sqrt(2 / (hiddenLayer2Size + outputLayerSize));
b3 = zeros(1, outputLayerSize);

% Initialize arrays to store loss and accuracy for plotting (for learning curve)
trainLossHistory = zeros(1, numEpochs / printInterval);
trainAccuracyHistory = zeros(1, numEpochs / printInterval);

%% Section 3: Training Loop (Stochastic Gradient Descent)

for epoch = 1:numEpochs
    % Shuffle training data for each epoch (important for SGD)
    shuffleIdx = randperm(size(X_train, 1));
    X_train = X_train(shuffleIdx, :);
    y_train = y_train(shuffleIdx, :);

    % Iterate over each training example (SGD)
    for i = 1:size(X_train, 1)
        % Select a single training example
        batch_X = X_train(i, :); % 1x4 vector
        batch_y = y_train(i, :); % 1x3 vector

        % Forward propagation
        % First hidden layer
        z1 = batch_X * W1 + b1;         % 1x5 vector
        a1 = tanh(z1);                  % 1x5 vector, activation output
        % Second hidden layer
        z2 = a1 * W2 + b2;              % 1x3 vector
        a2 = tanh(z2);                  % 1x3 vector, activation output
        % Output layer
        z3 = a2 * W3 + b3;              % 1x3 vector
        a3 = tanh(z3);                  % 1x3 vector, activation output

        % Backpropagation
        % Output layer delta
        delta3 = (batch_y - a3) .* (1 - a3.^2); % 1x3 vector
        % Second hidden layer delta
        delta2 = (delta3 * W3') .* (1 - a2.^2); % 1x3 vector
        % First hidden layer delta
        delta1 = (delta2 * W2') .* (1 - a1.^2); % 1x5 vector

        % Update weights and biases (using learning rate)
        W3 = W3 + learningRate * (a2' * delta3); % 3x3 matrix
        b3 = b3 + learningRate * delta3;         % 1x3 vector
        W2 = W2 + learningRate * (a1' * delta2); % 5x3 matrix
        b2 = b2 + learningRate * delta2;         % 1x3 vector
        W1 = W1 + learningRate * (batch_X' * delta1); % 4x5 matrix
        b1 = b1 + learningRate * delta1;             % 1x5 vector
    end

    % Calculate and print training accuracy at specified intervals
    if mod(epoch, printInterval) == 0
        % Forward pass on the entire training data (for accuracy calculation)
        % Replicate bias vectors for matrix operations
        z1_train = X_train * W1 + repmat(b1, size(X_train, 1), 1);
        a1_train = tanh(z1_train);
        z2_train = a1_train * W2 + repmat(b2, size(a1_train, 1), 1);
        a2_train = tanh(z2_train);
        z3_train = a2_train * W3 + repmat(b3, size(a2_train, 1), 1);
        a3_train = tanh(z3_train);

        % Calculate training accuracy
        [~, predictedLabels] = max(a3_train, [], 2);
        [~, trueLabels] = max(y_train, [], 2);
        trainingAccuracy = mean(predictedLabels == trueLabels);
        
        % Calculate training loss (mean squared error)
        trainLoss = mean(sum((y_train - a3_train).^2, 2));

        % Store loss and accuracy for plotting
        historyIndex = epoch / printInterval;
        trainLossHistory(historyIndex) = trainLoss;
        trainAccuracyHistory(historyIndex) = trainingAccuracy;

        fprintf('Epoch %d: Training Accuracy = %.2f%%, Loss = %.4f\n', epoch, trainingAccuracy * 100, trainLoss);
    end
end

%% Section 4: Validation

% Forward pass on validation data
% Replicate bias vectors for matrix operations
z1_val = X_val * W1 + repmat(b1, size(X_val, 1), 1);
a1_val = tanh(z1_val);
z2_val = a1_val * W2 + repmat(b2, size(a1_val, 1), 1);
a2_val = tanh(z2_val);
z3_val = a2_val * W3 + repmat(b3, size(a2_val, 1), 1);
a3_val = tanh(z3_val);

% Calculate validation accuracy
[~, predictedValLabels] = max(a3_val, [], 2);
[~, trueValLabels] = max(y_val, [], 2);
validationAccuracy = mean(predictedValLabels == trueValLabels);
fprintf('Validation Accuracy: %.2f%%\n', validationAccuracy * 100);

%% Section 5: Plotting Learning Curve
figure;
subplot(2, 1, 1);
plot(printInterval:printInterval:numEpochs, trainLossHistory, 'b-');
title('Training Loss');
xlabel('Epoch');
ylabel('Loss');

subplot(2, 1, 2);
plot(printInterval:printInterval:numEpochs, trainAccuracyHistory, 'r-');
title('Training Accuracy');
xlabel('Epoch');
ylabel('Accuracy');

%% Section 6: Confusion Matrix (Validation)

% Generate confusion matrix for the validation set
confusionMatrix = confusionmat(trueValLabels, predictedValLabels);

% Display the confusion matrix
disp("Confusion Matrix (Validation):");
disp(confusionMatrix);

% Plot the confusion matrix with class labels
figure;
confusionchart(confusionMatrix, {'Setosa', 'Versicolor', 'Virginica'}, 'Title', 'Confusion Matrix (Validation)');

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
                oneHotLabel = [0.6, -0.6, -0.6];
            elseif contains(lineParts{5}, 'versicolor')
                oneHotLabel = [-0.6, 0.6, -0.6];
            else % virginica
                oneHotLabel = [-0.6, -0.6, 0.6];
            end

            % Append the one-hot encoded label to the label matrix
            y = [y; oneHotLabel];
        end
    end
    
    % Close the file
    fclose(fileID);
end