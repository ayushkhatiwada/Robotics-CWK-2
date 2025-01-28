% =========================================================================
% Task 1(c): Regression using a 1-3-1 Neural Network with ADAM Optimizer
% =========================================================================
clear; clc; close all;

% -----------------------
% (I) Generate Training Data
% -----------------------
baseSeed = 42;
rng(baseSeed, 'twister');

% x: from -1 to +1 in increments of 0.05
x = -1:0.05:1;
len = length(x);

% d: desired output with noise
d = 0.8*x.^3 + 0.3*x.^2 - 0.4*x + normrnd(0, 0.02, [1, len]);

% Prepare some test inputs for final plotting
xtest = -0.97:0.1:0.93;
nTest = length(xtest);

% -----------------------
% (II) Define ADAM parameters + Learning Rates to test
% -----------------------
learning_rates = [0.01, 0.001, 0.0001];  % We'll try these 3 LR values
epochs = 5000;                           % Number of epochs per run

% ADAM hyper-parameters
beta1   = 0.9;   
beta2   = 0.999;
epsilon = 1e-8;

% We'll store MSE vs. epoch for each LR in a matrix (epochs x numLR)
mse_storage = zeros(epochs, length(learning_rates));

% We'll store final test predictions in a matrix (nTest x numLR)
ytest_storage = zeros(nTest, length(learning_rates));

% -----------------------
% (III) Plot the training data (for reference)
% -----------------------
figure;
plot(x, d, 'k+');
xlabel('x'); ylabel('d');
title('Training Data (Used for All ADAM Runs)');
grid on;

% -----------------------
% (IV) Loop Over Each Learning Rate
% -----------------------
for iLR = 1:length(learning_rates)
    
    eta = learning_rates(iLR);  % Current learning rate
    
    fprintf('\n====================================\n');
    fprintf(' ADAM run with learning rate = %.4f\n', eta);
    fprintf('====================================\n');
    
    % ----------------------------------------------------
    % (A) Re-initialize weights, biases, and ADAM moments
    % ----------------------------------------------------
    rng(baseSeed, 'twister');  % so each LR has same initial weights
    
    % Hidden layer weights (w1) and biases (b1)
    %  w1 = w_{1j}^{(1)} => (1x3)
    %  b1 = w_{0j}^{(1)} => (1x3)
    w1 = rand(1,3) - 0.5;
    b1 = rand(1,3) - 0.5;
    
    % Output layer weights (w2) and bias (b2)
    %  w2 = w_{j1}^{(2)} => (3x1)
    %  b2 = w_{0,1}^{(2)} => scalar
    w2 = rand(3,1) - 0.5;
    b2 = rand(1)   - 0.5;
    
    % Initialize ADAM moment estimates (m = first moment, v = second moment)
    % for w1, b1
    m_w1 = zeros(size(w1));
    v_w1 = zeros(size(w1));
    m_b1 = zeros(size(b1));
    v_b1 = zeros(size(b1));
    
    % for w2, b2
    m_w2 = zeros(size(w2));
    v_w2 = zeros(size(w2));
    m_b2 = 0;
    v_b2 = 0;
    
    % Array for MSE each epoch
    mse_array = zeros(epochs,1);
    
    % ----------------------------------------------------
    % (B) ADAM Training Loop
    % ----------------------------------------------------
  
    
    global_step = 0;  % ADAM "time step" (t)
    
    for epoch = 1:epochs
        
        % Shuffle training data
        idx = randperm(len);
        x_shuffled = x(idx);
        d_shuffled = d(idx);
        
        total_error = 0;
        
        for n = 1:len
            
            global_step = global_step + 1;  % increment ADAM time step
            t = global_step;               % for clarity
            
            % ------------------
            % (1) Forward pass
            % ------------------
            x_n = x_shuffled(n);
            d_n = d_shuffled(n);
            
            % Hidden layer net input: v_j^{(1)} = w1*x_n + b1
            v1 = w1 * x_n + b1;     % (1x3)
            % Hidden output: h_j = tanh(v1_j)
            h  = tanh(v1);         % (1x3)
            
            % Output layer net input: v^{(2)} = sum_j( w2_j * h_j ) + b2
            v2 = w2' * h' + b2;    % scalar
            y  = v2;               % linear output => y = v2
            
            % Error e = d_n - y
            e  = d_n - y;
            total_error = total_error + e^2;
            
            % ------------------
            % (2) Backward pass
            % ------------------
            % For a MSE loss E ~ (d - y)^2, the gradient wrt y is -(d-y) => -e
            % So we define:
            %   delta2 = derivative wrt v2 = -e (since linear act. derivative = 1)
          
            
            delta2 = -e;    % negative gradient for minimization
            
            % Gradients for output layer parameters:
            %   grad_w2 = dE/dw2 => delta2 * h
            %   grad_b2 = dE/db2 => delta2
            grad_w2 = delta2 * h';   % (3x1)
            grad_b2 = delta2;        % scalar
            
            % Hidden layer local gradients:
            %   delta1_j = delta2 * w2_j * (1 - tanh^2(v1_j))
            delta1 = (1 - h.^2) .* (delta2 * w2');
            
            % Gradients wrt hidden parameters:
            %   grad_w1 => shape (3x1), so we do delta1' * x_n => (3x1)
            %   grad_b1 => shape (1x3)
            grad_w1 = (delta1' * x_n); 
            grad_b1 = delta1;
            
            % ------------------
            % (3) ADAM Updates
            % ------------------
            % ADAM formula references:
            %   m_t = beta1*m_{t-1} + (1-beta1)*grad
            %   v_t = beta2*v_{t-1} + (1-beta2)*(grad^2)
            %   m_hat = m_t / (1 - beta1^t)
            %   v_hat = v_t / (1 - beta2^t)
            %   w <- w - eta * (m_hat / ( sqrt(v_hat) + eps ))
            
            % ---- w2
            m_w2 = beta1 * m_w2 + (1 - beta1) * grad_w2; 
            v_w2 = beta2 * v_w2 + (1 - beta2) * (grad_w2.^2);
            m_w2_hat = m_w2 / (1 - beta1^t);
            v_w2_hat = v_w2 / (1 - beta2^t);
            w2 = w2 - eta * (m_w2_hat ./ (sqrt(v_w2_hat) + epsilon));
            
            % ---- b2
            m_b2 = beta1 * m_b2 + (1 - beta1) * grad_b2;
            v_b2 = beta2 * v_b2 + (1 - beta2) * (grad_b2^2);
            m_b2_hat = m_b2 / (1 - beta1^t);
            v_b2_hat = v_b2 / (1 - beta2^t);
            b2 = b2 - eta * (m_b2_hat / (sqrt(v_b2_hat) + epsilon));
            
            % ---- w1
            m_w1 = beta1 * m_w1 + (1 - beta1) * grad_w1';
            v_w1 = beta2 * v_w1 + (1 - beta2) * (grad_w1'.^2);
            m_w1_hat = m_w1 / (1 - beta1^t);
            v_w1_hat = v_w1 / (1 - beta2^t);
            w1 = w1 - eta * (m_w1_hat ./ (sqrt(v_w1_hat) + epsilon));
            
            % ---- b1
            m_b1 = beta1 * m_b1 + (1 - beta1) * grad_b1;
            v_b1 = beta2 * v_b1 + (1 - beta2) * (grad_b1.^2);
            m_b1_hat = m_b1 / (1 - beta1^t);
            v_b1_hat = v_b1 / (1 - beta2^t);
            b1 = b1 - eta * (m_b1_hat ./ (sqrt(v_b1_hat) + epsilon));
            
        end % end for each sample
        
        % Mean squared error for this epoch
        mse_array(epoch) = total_error / len;
        
        % Display progress every 100 epochs
        if mod(epoch, 100) == 0
            fprintf('Epoch %4d / %4d, LR=%.4f, MSE=%.6f\n', ...
                epoch, epochs, eta, mse_array(epoch));
        end
        
    end % end epoch loop
    
    % Store the final MSE curve for combined plot
    mse_storage(:, iLR) = mse_array;
    
    % Evaluate on xtest
    ytest = zeros(nTest,1);
    for iT = 1:nTest
        x_t = xtest(iT);
        
        v1_t   = w1 * x_t + b1; 
        h_t    = tanh(v1_t);
        v2_t   = w2' * h_t' + b2;
        ytest(iT) = v2_t;  
    end
    
    % Store final predictions
    ytest_storage(:, iLR) = ytest;
    
    fprintf('>>> Final MSE after %d epochs (LR=%.4f) = %.6f\n', ...
        epochs, eta, mse_array(end));
end

% -----------------------
% (V) Combine All Results in Single Plots
% -----------------------
colors = {'r','g','b'};  % Distinguish lines for each LR

% 1) Plot MSE vs. Epoch for all LRs
figure;
hold on;
for iLR = 1:length(learning_rates)
    plot(1:epochs, mse_storage(:, iLR), ...
         'Color', colors{iLR}, 'LineWidth',1.2, ...
         'DisplayName', ['LR=', num2str(learning_rates(iLR))]);
end
xlabel('Epoch');
ylabel('Mean Squared Error');
title('ADAM: MSE vs. Epoch for Different Learning Rates');
legend('Location','best');
grid on;
hold off;

% 2) Plot final regression curves vs. training data
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
title('ADAM: Final Regression Curves for Different LRs');
legend('Location','best');
grid on;
hold off;
