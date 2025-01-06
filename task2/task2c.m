%% Configuration Section
% Dataset Parameters
dataFilename = 'IrisData.txt';
trainRatio = 0.7;
seed = 42; % Random seed for reproducibility

% Network Architecture
inputLayerSize = 4;
hiddenLayer1Size = 5;
hiddenLayer2Size = 3;
outputLayerSize = 3;

% Training Parameters
learningRate = 0.01;
beta1 = 0.9;    % Exponential decay rate for the first moment estimate
beta2 = 0.999;  % Exponential decay rate for the second moment estimate
epsilon = 1e-8; % Small constant to prevent division by zero
numEpochs = 1000;
batchSize = 16;
printInterval = 100; % Print loss and accuracy every this many epochs

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

% Initialize weights with Xavier initialization
W1 = randn(inputLayerSize, hiddenLayer1Size) * sqrt(2 / (inputLayerSize + hiddenLayer1Size));
b1 = zeros(1, hiddenLayer1Size);
W2 = randn(hiddenLayer1Size, hiddenLayer2Size) * sqrt(2 / (hiddenLayer1Size + hiddenLayer2Size));
b2 = zeros(1, hiddenLayer2Size);
W3 = randn(hiddenLayer2Size, outputLayerSize) * sqrt(2 / (hiddenLayer2Size + outputLayerSize));
b3 = zeros(1, outputLayerSize);

% Initialize ADAM momentum and velocity variables
m_W1 = zeros(size(W1)); v_W1 = zeros(size(W1));
m_b1 = zeros(size(b1)); v_b1 = zeros(size(b1));
m_W2 = zeros(size(W2)); v_W2 = zeros(size(W2));
m_b2 = zeros(size(b2)); v_b2 = zeros(size(b2));
m_W3 = zeros(size(W3)); v_W3 = zeros(size(W3));
m_b3 = zeros(size(b3)); v_b3 = zeros(size(b3));

%% Section 3: Training Loop (ADAM Optimization)

numBatches = floor(size(X_train, 1) / batchSize);

% Storage for plotting
trainLossHistory = zeros(numEpochs, 1);
trainAccuracyHistory = zeros(numEpochs, 1);

% Training loop
t = 0; % Initialize time step for ADAM
for epoch = 1:numEpochs
    % Shuffle training data for each epoch
    shuffleIdx = randperm(size(X_train, 1));
    X_train = X_train(shuffleIdx, :);
    y_train = y_train(shuffleIdx, :);

    epochLoss = 0;
    correctPredictions = 0;

    for batch = 1:numBatches
        t = t + 1; % Increment time step

        % Get mini-batch
        batchStart = (batch - 1) * batchSize + 1;
        batchEnd = min(batch * batchSize, size(X_train, 1));
        X_batch = X_train(batchStart:batchEnd, :);
        y_batch = y_train(batchStart:batchEnd, :);

        % Forward propagation
        % First hidden layer
        z1 = X_batch * W1 + repmat(b1, size(X_batch, 1), 1);
        a1 = tanh(z1);

        % Second hidden layer
        z2 = a1 * W2 + repmat(b2, size(a1, 1), 1);
        a2 = tanh(z2);

        % Output layer
        z3 = a2 * W3 + repmat(b3, size(a2, 1), 1);
        a3 = tanh(z3);

        % Calculate loss (Mean Squared Error - MSE)
        batchLoss = mean(sum((a3 - y_batch).^2, 2));
        epochLoss = epochLoss + batchLoss;

        % Calculate accuracy
        [~, predictedLabels] = max(a3, [], 2);
        [~, trueLabels] = max(y_batch, [], 2);
        correctPredictions = correctPredictions + sum(predictedLabels == trueLabels);

        % Backpropagation
        % Output layer
        delta3 = (a3 - y_batch) .* (1 - a3.^2);

        % Second hidden layer
        delta2 = (delta3 * W3') .* (1 - a2.^2);

        % First hidden layer
        delta1 = (delta2 * W2') .* (1 - a1.^2);

        % Calculate gradients
        dW3 = a2' * delta3;
        db3 = sum(delta3, 1);
        dW2 = a1' * delta2;
        db2 = sum(delta2, 1);
        dW1 = X_batch' * delta1;
        db1 = sum(delta1, 1);

        % ADAM updates
        % W3 and b3
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

        % W2 and b2
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

        % W1 and b1
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
    end

    % Store training metrics
    trainLossHistory(epoch) = epochLoss / numBatches;
    trainAccuracyHistory(epoch) = correctPredictions / size(X_train, 1);

    % Print progress
    if mod(epoch, printInterval) == 0
        fprintf('Epoch %d: Loss = %.4f, Training Accuracy = %.2f%%\n', ...
            epoch, trainLossHistory(epoch), trainAccuracyHistory(epoch) * 100);
    end
end

%% Section 4: Validation

% Forward pass on validation data
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

%% Section 5: Plotting Training Progress

figure;
subplot(2, 1, 1);
plot(1:numEpochs, trainLossHistory, 'b-');
title('Training Loss (ADAM)');
xlabel('Epoch');
ylabel('Loss');
grid on;

subplot(2, 1, 2);
plot(1:numEpochs, trainAccuracyHistory, 'r-');
title('Training Accuracy (ADAM)');
xlabel('Epoch');
ylabel('Accuracy');
grid on;

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