function [X, y] = loadIrisData(filename)
    % Read the raw data line by line
    fid = fopen(filename, 'r');
    if fid == -1
        error('Could not open file %s', filename);
    end
    
    % Initialize arrays
    X = [];
    y = [];
    
    % Read line by line
    while ~feof(fid)
        line = fgetl(fid);
        if ischar(line) && ~isempty(line)
            % Split the line by commas
            parts = strsplit(line, ',');
            
            % Convert first 4 values to numbers
            features = str2double(parts(1:4));
            
            % Add to X matrix
            X = [X; features];
            
            % Create one-hot encoding based on the class
            if contains(parts{5}, 'setosa')
                y = [y; [0.6 -0.6 -0.6]];
            elseif contains(parts{5}, 'versicolor')
                y = [y; [-0.6 0.6 -0.6]];
            else % virginica
                y = [y; [-0.6 -0.6 0.6]];
            end
        end
    end
    fclose(fid);
end

% Main script
% Load and preprocess data
[X, y] = loadIrisData('IrisData.txt');

% Normalize features (this often helps with neural network training)
X = (X - mean(X)) ./ std(X);

% Randomize and split data
rng(42); % For reproducibility
n = size(X,1);
idx = randperm(n);
X = X(idx,:);
y = y(idx,:);

% Split data
split = floor(0.7 * n);
X_train = X(1:split,:);
y_train = y(1:split,:);
X_val = X(split+1:end,:);
y_val = y(split+1:end,:);

% Initialize network parameters
input_size = 4;
hidden1_size = 5;
hidden2_size = 3;
output_size = 3;

% Initialize weights with Xavier initialization (https://www.pinecone.io/learn/weight-initialization/)
W1 = randn(input_size, hidden1_size) * sqrt(2/(input_size + hidden1_size));
b1 = zeros(1, hidden1_size);
W2 = randn(hidden1_size, hidden2_size) * sqrt(2/(hidden1_size + hidden2_size));
b2 = zeros(1, hidden2_size);
W3 = randn(hidden2_size, output_size) * sqrt(2/(hidden2_size + output_size));
b3 = zeros(1, output_size);

% Training parameters
learning_rate = 0.01;
epochs = 1000;
batch_size = 1; % was 32

% Training loop
for epoch = 1:epochs
    % Shuffle training data
    idx = randperm(size(X_train,1));
    X_train = X_train(idx,:);
    y_train = y_train(idx,:);
    
    % Mini-batch training
    for i = 1:batch_size:size(X_train,1)
        % Get current batch
        batch_end = min(i + batch_size - 1, size(X_train,1));
        batch_X = X_train(i:batch_end,:);
        batch_y = y_train(i:batch_end,:);
        
        % Forward propagation
        % First hidden layer
        z1 = batch_X * W1 + repmat(b1, size(batch_X,1), 1);
        a1 = tanh(z1);
        % Second hidden layer
        z2 = a1 * W2 + repmat(b2, size(a1,1), 1);
        a2 = tanh(z2);
        % Output layer
        z3 = a2 * W3 + repmat(b3, size(a2,1), 1);
        a3 = tanh(z3);
        
        % Backpropagation
        % Output layer
        delta3 = (a3 - batch_y) .* (1 - a3.^2);
        % Second hidden layer
        delta2 = (delta3 * W3') .* (1 - a2.^2);
        % First hidden layer
        delta1 = (delta2 * W2') .* (1 - a1.^2);
        
        % Update weights and biases using mini-batch averages
        batch_size_actual = size(batch_X,1);
        W3 = W3 - learning_rate * (a2' * delta3) / batch_size_actual;
        b3 = b3 - learning_rate * sum(delta3,1) / batch_size_actual;
        W2 = W2 - learning_rate * (a1' * delta2) / batch_size_actual;
        b2 = b2 - learning_rate * sum(delta2,1) / batch_size_actual;
        W1 = W1 - learning_rate * (batch_X' * delta1) / batch_size_actual;
        b1 = b1 - learning_rate * sum(delta1,1) / batch_size_actual;
    end
    
    % Calculate and print training accuracy every 100 epochs
    if mod(epoch, 100) == 0
        % Forward pass on training data
        z1 = X_train * W1 + repmat(b1, size(X_train,1), 1);
        a1 = tanh(z1);
        z2 = a1 * W2 + repmat(b2, size(a1,1), 1);
        a2 = tanh(z2);
        z3 = a2 * W3 + repmat(b3, size(a2,1), 1);
        a3 = tanh(z3);
        [~, pred] = max(a3, [], 2);
        [~, true_labels] = max(y_train, [], 2);
        accuracy = mean(pred == true_labels);
        fprintf('Epoch %d: Training Accuracy = %.2f%%\n', epoch, accuracy*100);
    end
end

% Validation
z1 = X_val * W1 + repmat(b1, size(X_val,1), 1);
a1 = tanh(z1);
z2 = a1 * W2 + repmat(b2, size(a1,1), 1);
a2 = tanh(z2);
z3 = a2 * W3 + repmat(b3, size(a2,1), 1);
a3 = tanh(z3);
[~, pred] = max(a3, [], 2);
[~, true_labels] = max(y_val, [], 2);
val_accuracy = mean(pred == true_labels);
fprintf('Validation Accuracy: %.2f%%\n', val_accuracy*100);