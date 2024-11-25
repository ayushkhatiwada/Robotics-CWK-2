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
            features = str2double(parts(1:4))';
            % Add to X matrix
            X = [X features];
            % Create one-hot encoding based on the class
            if contains(parts{5}, 'setosa')
                y = [y [1; 0; 0]];
            elseif contains(parts{5}, 'versicolor')
                y = [y [0; 1; 0]];
            else % virginica
                y = [y [0; 0; 1]];
            end
        end
    end
    fclose(fid);
    
    % Transpose matrices to match the second implementation's format
    X = X';
    y = y';
end

% Main script
% Load and preprocess data
[X, y] = loadIrisData('IrisData.txt');

% Normalize features
X = (X - mean(X)) ./ std(X);

% Randomize and split data
rng(42); % For reproducibility
n = size(X,1);
idx = randperm(n);
X = X(idx,:);
y = y(idx,:);

trainRatio = 0.7;
split = floor(trainRatio * n);
X_train = X(1:split,:);
y_train = y(1:split,:);
X_val = X(split+1:end,:);
y_val = y(split+1:end,:);

% Initialize network parameters with improved architecture
input_size = 4;
hidden1_size = 5;
hidden2_size = 3;
output_size = 3;

% Initialize weights with Xavier initialization
W1 = randn(input_size, hidden1_size) * sqrt(2/(input_size + hidden1_size));
b1 = zeros(1, hidden1_size);
W2 = randn(hidden1_size, hidden2_size) * sqrt(2/(hidden1_size + hidden2_size));
b2 = zeros(1, hidden2_size);
W3 = randn(hidden2_size, output_size) * sqrt(2/(hidden2_size + output_size));
b3 = zeros(1, output_size);

% ADAM parameters
learning_rate = 0.001;
beta1 = 0.9;      % Exponential decay rate for first moment
beta2 = 0.999;    % Exponential decay rate for second moment
epsilon = 1e-8;   % Small number to prevent division by zero

% Initialize ADAM momentum variables
m_W1 = zeros(size(W1)); v_W1 = zeros(size(W1));
m_b1 = zeros(size(b1)); v_b1 = zeros(size(b1));
m_W2 = zeros(size(W2)); v_W2 = zeros(size(W2));
m_b2 = zeros(size(b2)); v_b2 = zeros(size(b2));
m_W3 = zeros(size(W3)); v_W3 = zeros(size(W3));
m_b3 = zeros(size(b3)); v_b3 = zeros(size(b3));

% Training parameters
epochs = 1000;
batch_size = 32;
num_batches = floor(size(X_train, 1) / batch_size);

% Storage for plotting
train_loss = zeros(epochs, 1);
train_acc = zeros(epochs, 1);

% Training loop
t = 0;  % Initialize time step for ADAM
for epoch = 1:epochs
    % Shuffle training data
    idx = randperm(size(X_train,1));
    X_train = X_train(idx,:);
    y_train = y_train(idx,:);
    
    epoch_loss = 0;
    correct_predictions = 0;
    
    for batch = 1:num_batches
        t = t + 1;  % Increment time step
        
        % Get mini-batch
        batch_start = (batch-1)*batch_size + 1;
        batch_end = min(batch*batch_size, size(X_train,1));
        X_batch = X_train(batch_start:batch_end, :);
        y_batch = y_train(batch_start:batch_end, :);
        
        % Forward propagation
        % First hidden layer
        z1 = X_batch * W1 + repmat(b1, size(X_batch,1), 1);
        a1 = tanh(z1);
        
        % Second hidden layer
        z2 = a1 * W2 + repmat(b2, size(a1,1), 1);
        a2 = tanh(z2);
        
        % Output layer
        z3 = a2 * W3 + repmat(b3, size(a2,1), 1);
        a3 = tanh(z3);
        
        % Calculate loss
        batch_loss = mean(sum((a3 - y_batch).^2, 2));
        epoch_loss = epoch_loss + batch_loss;
        
        % Calculate accuracy
        [~, pred] = max(a3, [], 2);
        [~, true_labels] = max(y_batch, [], 2);
        correct_predictions = correct_predictions + sum(pred == true_labels);
        
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
        W3 = W3 - learning_rate * m_hat_W3 ./ (sqrt(v_hat_W3) + epsilon);
        
        m_b3 = beta1 * m_b3 + (1 - beta1) * db3;
        v_b3 = beta2 * v_b3 + (1 - beta2) * db3.^2;
        m_hat_b3 = m_b3 / (1 - beta1^t);
        v_hat_b3 = v_b3 / (1 - beta2^t);
        b3 = b3 - learning_rate * m_hat_b3 ./ (sqrt(v_hat_b3) + epsilon);
        
        % W2 and b2
        m_W2 = beta1 * m_W2 + (1 - beta1) * dW2;
        v_W2 = beta2 * v_W2 + (1 - beta2) * dW2.^2;
        m_hat_W2 = m_W2 / (1 - beta1^t);
        v_hat_W2 = v_W2 / (1 - beta2^t);
        W2 = W2 - learning_rate * m_hat_W2 ./ (sqrt(v_hat_W2) + epsilon);
        
        m_b2 = beta1 * m_b2 + (1 - beta1) * db2;
        v_b2 = beta2 * v_b2 + (1 - beta2) * db2.^2;
        m_hat_b2 = m_b2 / (1 - beta1^t);
        v_hat_b2 = v_b2 / (1 - beta2^t);
        b2 = b2 - learning_rate * m_hat_b2 ./ (sqrt(v_hat_b2) + epsilon);
        
        % W1 and b1
        m_W1 = beta1 * m_W1 + (1 - beta1) * dW1;
        v_W1 = beta2 * v_W1 + (1 - beta2) * dW1.^2;
        m_hat_W1 = m_W1 / (1 - beta1^t);
        v_hat_W1 = v_W1 / (1 - beta2^t);
        W1 = W1 - learning_rate * m_hat_W1 ./ (sqrt(v_hat_W1) + epsilon);
        
        m_b1 = beta1 * m_b1 + (1 - beta1) * db1;
        v_b1 = beta2 * v_b1 + (1 - beta2) * db1.^2;
        m_hat_b1 = m_b1 / (1 - beta1^t);
        v_hat_b1 = v_b1 / (1 - beta2^t);
        b1 = b1 - learning_rate * m_hat_b1 ./ (sqrt(v_hat_b1) + epsilon);
    end
    
    % Store training metrics
    train_loss(epoch) = epoch_loss / num_batches;
    train_acc(epoch) = correct_predictions / size(X_train,1);
    
    % Print progress every 100 epochs
    if mod(epoch, 100) == 0
        fprintf('Epoch %d: Loss = %.4f, Training Accuracy = %.2f%%\n', ...
            epoch, train_loss(epoch), train_acc(epoch)*100);
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

% Plot training progress
figure;
subplot(2,1,1);
plot(1:epochs, train_loss);
title('Training Loss');
xlabel('Epoch');
ylabel('Loss');
grid on;

subplot(2,1,2);
plot(1:epochs, train_acc);
title('Training Accuracy');
xlabel('Epoch');
ylabel('Accuracy');
grid on;