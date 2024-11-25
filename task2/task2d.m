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
end

% Main script
% Load and preprocess data
[X, y] = loadIrisData('IrisData.txt');

% Normalize features
X = (X - mean(X, 2)) ./ std(X, 0, 2);

% Randomize and split data
rng(42); % For reproducibility
n = size(X, 2);
idx = randperm(n);
X = X(:, idx);
y = y(:, idx);

trainRatio = 0.7;
split = floor(trainRatio * n);
X_train = X(:, 1:split);
y_train = y(:, 1:split);
X_val = X(:, split+1:end);
y_val = y(:, split+1:end);

% Create figure for plotting training progress
figure('Position', [100, 100, 800, 600]);
subplot(3,1,1);
hold on;
title('Training Performance');

% 1. Original architecture (4-5-3-3) with tanh
net1 = patternnet([5 3]);
net1.layers{1}.transferFcn = 'tansig';
net1.layers{2}.transferFcn = 'tansig';
net1.layers{3}.transferFcn = 'softmax';
net1.trainParam.showWindow = false;  % Disable default training window
net1.trainParam.epochs = 100;        % Set maximum epochs
net1 = configure(net1, X_train, y_train);

% Custom training with progress tracking
[net1, tr1] = train(net1, X_train, y_train);
plot(tr1.epoch, tr1.perf, 'b-', 'LineWidth', 2);
xlabel('Epoch');
ylabel('MSE');
title('Original Architecture Performance');

% Test performance
y_pred1 = net1(X_val);
accuracy1 = mean(vec2ind(y_pred1) == vec2ind(y_val));
fprintf('Original architecture accuracy: %.2f%%\n', accuracy1*100);

% 2. Smaller architecture with ReLU
subplot(3,1,2);
hold on;
net2 = patternnet(3);
net2.layers{1}.transferFcn = 'poslin';
net2.layers{2}.transferFcn = 'softmax';
net2.trainParam.showWindow = false;
net2.trainParam.epochs = 100;
net2 = configure(net2, X_train, y_train);

[net2, tr2] = train(net2, X_train, y_train);
plot(tr2.epoch, tr2.perf, 'r-', 'LineWidth', 2);
xlabel('Epoch');
ylabel('MSE');
title('Smaller Architecture Performance');

y_pred2 = net2(X_val);
accuracy2 = mean(vec2ind(y_pred2) == vec2ind(y_val));
fprintf('Smaller architecture with ReLU accuracy: %.2f%%\n', accuracy2*100);

% 3. Larger architecture with mixed activation functions
subplot(3,1,3);
hold on;
net3 = patternnet([10 5]);
net3.layers{1}.transferFcn = 'poslin';
net3.layers{2}.transferFcn = 'tansig';
net3.layers{3}.transferFcn = 'softmax';
net3.trainParam.showWindow = false;
net3.trainParam.epochs = 100;
net3 = configure(net3, X_train, y_train);

[net3, tr3] = train(net3, X_train, y_train);
plot(tr3.epoch, tr3.perf, 'g-', 'LineWidth', 2);
xlabel('Epoch');
ylabel('MSE');
title('Larger Architecture Performance');

y_pred3 = net3(X_val);
accuracy3 = mean(vec2ind(y_pred3) == vec2ind(y_val));
fprintf('Larger architecture with mixed activations accuracy: %.2f%%\n', accuracy3*100);

% Adjust figure layout
sgtitle('Neural Network Training Performance Comparison');