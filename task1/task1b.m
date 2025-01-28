% =====================================================================
% Task 1(b): Regression using a 1-3-1 Neural Network with Backpropagation
% =====================================================================

clear; clc; close all;

% -----------------------
% (I) Create Training Data
% -----------------------
baseSeed = 42;  % For reproducibility
rng(baseSeed, 'twister');

% x: input data
x = -1:0.05:1;  
len = length(x);

% d: desired output with noise 
%    underlying function = 0.8x^3 + 0.3x^2 - 0.4x, plus small noise
d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x + normrnd(0, 0.02, [1, len]);

% Prepare test data (used later to plot final predictions)
xtest = -0.97:0.1:0.93;
nTest = length(xtest);

% -----------------------
% (II) Define Learning Rates to Test
% -----------------------
learning_rates = [0.1, 0.01, 0.001];

% Number of training epochs
epochs = 30000;

% -----------------------
% (III) We'll store results for combined plotting after all runs
% -----------------------
% 1) MSE vs. epoch for each LR => an (epochs x numLR) matrix
mse_storage = zeros(epochs, length(learning_rates));

% 2) Final predictions on xtest => an (nTest x numLR) matrix
ytest_storage = zeros(nTest, length(learning_rates));

% -----------------------
% (IV) Plot training data once, for reference
% -----------------------
figure;
plot(x, d, 'k+');
xlabel('x');
ylabel('d (target)');
title('Training Data (Common for All LR Trials)');
grid on;

% -----------------------
% (V) Loop Over Each Learning Rate
% -----------------------
for iLR = 1:length(learning_rates)
    
    lr = learning_rates(iLR);
    
    fprintf('\n===================================\n');
    fprintf(' Starting new run with LR = %g\n', lr);
    fprintf('===================================\n');
    
    % -------------------------------------------------
    % (A) Re-initialize the network (weights/biases)
    % -------------------------------------------------
    % We re-set the random seed so each LR run has identical initial weights
    rng(baseSeed, 'twister');
    
    % Hidden layer: w1 is (1x3), b1 is (1x3)
    w1 = rand(1,3) - 0.5;
    b1 = rand(1,3) - 0.5;
    % Output layer: w2 is (3x1), b2 is scalar
    w2 = rand(3,1) - 0.5;
    b2 = rand(1)   - 0.5;
    
    % We'll use the same LR for hidden and output updates
    eta1 = lr;  
    eta2 = lr;  
    
    % Prepare array for MSE each epoch
    mse_array = zeros(epochs,1);
    
    % -----------------------
    % (B) SGD Training Loop
    % -----------------------
    for epoch = 1:epochs
        
        % Shuffle the data indices
        idx = randperm(len);
        x_shuffled = x(idx);
        d_shuffled = d(idx);
        
        total_error = 0;
        
        for n = 1:len
            % Current sample and target
            x_n = x_shuffled(n);
            d_n = d_shuffled(n);
            
            % ------------------
            % (1) Forward pass
            % ------------------
            % Hidden layer net input: v1 (1x3)
            v1 = w1 * x_n + b1;  
            % Hidden output: h = tanh(v1)
            h  = tanh(v1);       % (1x3)
            
            % Output layer net input: v2 (scalar)
            v2 = w2' * h' + b2;  % w2'*(3x1) => scalar
            % y = linear output
            y  = v2;
            
            % Error = d_n - y
            e = d_n - y;
            
            % Accumulate squared error
            total_error = total_error + e^2;
            
            % ------------------
            % (2) Backward pass
            % ------------------
            % Output layer local gradient: delta2 = e
            delta2 = e; 
            
            % Hidden layer gradients (1x3)
            %   delta1_j = delta2 * w2_j * (1 - h_j^2)
            delta1 = (1 - h.^2) .* (delta2 * w2');
            
            % ------------------
            % (3) Weight updates
            % ------------------
            % Output layer:
            w2 = w2 + eta2 * (h' * delta2);  % (3x1) + (3x1)
            b2 = b2 + eta2 * delta2;         % scalar
            
            % Hidden layer:
            w1 = w1 + eta1 * (x_n * delta1); % (1x3) + (1x3)
            b1 = b1 + eta1 * delta1;         % (1x3)
            
        end % end of training samples
        
        % MSE for this epoch
        mse_array(epoch) = total_error / len;
        
        % Print progress every 1000 epochs
        if mod(epoch,1000)==0
            fprintf('Epoch %5d / %5d, LR=%g, MSE=%.6f\n', ...
                epoch, epochs, lr, mse_array(epoch));
        end
        
    end % end epoch loop
    
    % -------------------------------------------------
    % (C) Store final MSE curve for combined plotting
    % -------------------------------------------------
    mse_storage(:, iLR) = mse_array;
    
    % -------------------------------------------------
    % (D) Evaluate on xtest for final predictions
    % -------------------------------------------------
    ytest = zeros(size(xtest));
    for iT = 1:nTest
        x_t = xtest(iT);
        % Forward pass
        v1_t   = w1 * x_t + b1;
        h_t    = tanh(v1_t);
        v2_t   = w2' * h_t' + b2;
        ytest(iT) = v2_t;  % linear output
    end
    
    % Save these final predictions for combined plot
    ytest_storage(:, iLR) = ytest;
    
    % Print final MSE
    fprintf('*** Final MSE after %d epochs (LR=%.4f): %.6f\n', ...
        epochs, lr, mse_array(end));

end % end for each learning rate

% -----------------------------------------------------
% (VI) Combine All Results in Two Plots
% -----------------------------------------------------

% -- (1) MSE Curves 
figure;
hold on;
colors = {'r','g','b'};  
for iLR = 1:length(learning_rates)
    plot(1:epochs, mse_storage(:, iLR), ...
         'Color', colors{iLR}, 'LineWidth', 1.2, ...
         'DisplayName', ['LR=', num2str(learning_rates(iLR))]);
end
xlabel('Epoch');
ylabel('Mean Squared Error');
title('MSE vs. Epoch for Different Learning Rates (SGD)');
legend('Location','best');
grid on;
hold off;

% -- (2) Final Regression Curves 
figure;
plot(x, d, 'k+', 'DisplayName','Training Data'); 
hold on;
for iLR = 1:length(learning_rates)
    plot(xtest, ytest_storage(:, iLR), ...
         'Color', colors{iLR}, 'LineWidth',1.2, ...
         'DisplayName', ['LR=', num2str(learning_rates(iLR))]);
end
xlabel('x');
ylabel('y (prediction)');
title('Final Regression Curves for Different Learning Rates');
legend('Location','best');
grid on;
hold off;
