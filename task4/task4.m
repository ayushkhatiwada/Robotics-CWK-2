%% Configuration Section

% Set random seed for reproducibility
rng(11);

% Hyperparameters struct for easy experimentation
params = struct();
params.learningRate = 0.1;       % Learning rate alpha
params.discountFactor = 0.9;     % Discount factor gamma
params.initialEpsilon = 1.0;      % Initial exploration rate epsilon
params.minEpsilon = 0.1;         % Minimum exploration rate epsilon
params.epsilonDecayRate = 0.9994; % Decay rate for epsilon
params.maxEpisodes = 5000;       % Maximum number of episodes to train
params.maxSteps = 50;            % Maximum steps per episode
params.convergenceTolerance = 1e-6; % Tolerance for convergence check (Q-value change)
params.convergenceEpisodes = 500;  % Number of consecutive episodes for convergence

% Environment parameters
env = struct();
env.numStates = 12;          % Number of states in the grid world
env.numActions = 4;         % Number of possible actions (Up, Right, Down, Left)
env.obstacleState = 6;       % State number of the obstacle
env.terminalStates = [8, 12]; % State numbers of terminal states (8: -10 reward, 12: +10 reward)
env.rewards = ones(1, env.numStates) * -1; % Initialize rewards to -1 for all states (living reward)
env.rewards(8) = -10;                     % Reward for entering terminal state 8
env.rewards(12) = 10;                    % Reward for entering terminal state 12

% Define global ACTIONS struct for action names and indices for better readability
ACTIONS = struct();
ACTIONS.names = ["Up", "Right", "Down", "Left"];
ACTIONS.UP = 1;
ACTIONS.RIGHT = 2;
ACTIONS.DOWN = 3;
ACTIONS.LEFT = 4;

% Initialize Q-table with zeros.
Q = zeros(env.numStates, env.numActions);

% --- Enhanced Plot Settings for Readability at Small Sizes ---
set(0, 'DefaultAxesFontSize', 16);
set(0, 'DefaultTextFontSize', 16);
set(0, 'DefaultLineLineWidth', 2);
% --------------------------------------------------------------

%% Helper Functions

function [alpha, action] = selectAction(Q, state, epsilon)
    % Implements epsilon-greedy action selection.
    % With probability epsilon, choose a random action (exploration).
    % Otherwise, choose the action with the highest Q-value for the current state (exploitation).

    alpha = rand(); % Generate a random number from a uniform distribution [0, 1)
    if alpha < epsilon
        action = randi(size(Q, 2)); % Explore: Choose a random action
    else
        [max_vals, ~] = max(Q(state, :)); % Find the maximum Q-value for the current state
        % Find all actions with max value in case of ties, and break ties randomly
        best_actions = find(Q(state, :) == max_vals);
        action = best_actions(randi(length(best_actions))); % Exploit: Choose the best action (tie-breaking)
    end
end


function [nextState, reward] = getNextState(currentState, action, rewards, obstacleState, actions)
    % Determines the next state and reward based on the current state and action.
    % Takes into account grid boundaries and the obstacle.
    % Grid dimensions are fixed as 3 rows and 4 columns as per the problem description.
    rows = 3;
    cols = 4;

    % Convert state number to row and column indices for easier grid manipulation.
    row = ceil(currentState/cols); % Calculate row number
    col = mod(currentState-1, cols) + 1; % Calculate column number

    % Calculate next position based on the chosen action.
    switch action
        case actions.UP
            row = row + 1;
        case actions.RIGHT
            col = col + 1;
        case actions.DOWN
            row = row - 1;
        case actions.LEFT
            col = col - 1;
    end

    % Check for boundary conditions: if the agent tries to move out of the grid, it stays in the current state.
    if row < 1 || row > rows || col < 1 || col > cols
        nextState = currentState; % Stay in current state if action leads out of grid
    else
        nextState = (row-1)*cols + col; % Calculate next state number from row and column
        if nextState == obstacleState
            nextState = currentState; % Stay in current state if action leads to obstacle (cell 6)
        end
    end

    % Get the reward associated with the next state.
    reward = rewards(nextState);
end


function visualizeGridWorld(optimalPolicy, obstacleState, terminalStates)
    % Displays the optimal policy on the grid world.
    % Uses arrow symbols to represent actions and 'X' for obstacle, 'T' for terminal states.
    symbols = ['↑', '→', '↓', '←']; % Symbols for Up, Right, Down, Left actions

    % Create a cell array to represent the grid for display.
    grid = cell(3, 4);
    for state = 1:12
        row = ceil(state/4);
        col = mod(state-1, 4) + 1;

        if state == obstacleState
            grid{row, col} = 'X'; % Mark obstacle cell with 'X'
        elseif ismember(state, terminalStates)
            grid{row, col} = sprintf('T%d', find(terminalStates == state)); % Mark terminal states as T1, T2.
        else
            grid{row, col} = symbols(optimalPolicy(state)); % Use action symbol for other states based on optimal policy
        end
    end

    % Display the grid world policy in the command window.
    disp('Grid World Policy:');
    for row = 3:-1:1 % Iterate rows from top to bottom for display
        for col = 1:4 % Iterate columns
            fprintf('%2s ', grid{row, col}); % Print grid cell content with formatting
        end
        fprintf('\n');
    end
end

%% Training Loop with Enhanced Monitoring
episodeHistory = struct('states', {}, 'actions', {}, 'rewards', {}, ...
    'steps', {}, 'epsilon', {}, 'maxQChange', {}, 'avgQValue', {}); % Structure to store episode-wise data for analysis

epsilon = params.initialEpsilon; % Initialize epsilon for exploration-exploitation balance
previousQ = Q; % Store Q-table from the previous iteration to check for convergence
convergenceCounter = 0; % Counter for consecutive episodes with small Q-value change
converged = false; % Flag to indicate if training has converged

% Track Q-value statistics over episodes for monitoring training progress
qValueStats = struct('mean', [], 'std', [], 'max', [], 'min', []);

fprintf('Starting Q-Learning Training...\n');

for episode = 1:params.maxEpisodes
    currentState = 1; % Start each episode from state 1
    totalReward = 0; % Initialize total reward for the episode
    stepCount = 0; % Initialize step count for the episode
    episodeStates = currentState; % Record states visited in the episode
    episodeActions = []; % Record actions taken in the episode
    episodeRewards = []; % Record rewards received in the episode

    % Episode loop: runs until maximum steps per episode or terminal state is reached
    for step = 1:params.maxSteps
        stepCount = step;

        % Select action using epsilon-greedy policy
        [alpha, action] = selectAction(Q, currentState, epsilon);

        % Take action and observe next state and reward
        [nextState, reward] = getNextState(currentState, action, env.rewards, env.obstacleState, ACTIONS);

        % Q-Learning Update Rule: Update Q-value for the (currentState, action) pair
        nextMaxQ = max(Q(nextState, :)); % Estimate of optimal future value in next state
        Q(currentState, action) = Q(currentState, action) + ...
            params.learningRate * (reward + params.discountFactor * nextMaxQ - Q(currentState, action)); % Q-update formula

        % Print iteration details for the first 3 episodes and first 3 steps for report explanation and the final step of episode 1 and 2 before the start of step 1 of episode 2 and 3
        if episode <= 3 && step <= 3
            if step == 1
                fprintf("Initial Q-table:\n");
                disp(previousQ); % Display the entire Q-table to show updates after each of the first 9 steps
            end
            fprintf('\nEpisode: %d, Step: %d, State: %d, Alpha: %d, Epsilon: %d, Action: %s, Next State: %d, Reward: %d\n', ...
                episode, step, currentState, alpha, epsilon, ACTIONS.names(action), nextState, reward);
            fprintf('Q-table after update (State %d, Action %s):\n', currentState, ACTIONS.names(action));
            disp(Q); % Display the entire Q-table to show updates after each of the first 9 steps
        end

        % Store trajectory of the episode for analysis
        episodeStates = [episodeStates, nextState];
        episodeActions = [episodeActions, action];
        episodeRewards = [episodeRewards, reward];

        currentState = nextState; % Move to the next state
        totalReward = totalReward + reward; % Accumulate episode reward

        if ismember(currentState, env.terminalStates) % Check if terminal state is reached
            break; % End episode if terminal state is reached
        end
    end

    % Update epsilon for next episode - reduce exploration over time
    epsilon = max(params.minEpsilon, ...
        params.initialEpsilon * (params.epsilonDecayRate^episode)); % Exponential decay of epsilon, clamped at minEpsilon

    % Store episode data for later analysis and plotting
    episodeHistory(episode).states = episodeStates;
    episodeHistory(episode).actions = episodeActions;
    episodeHistory(episode).rewards = episodeRewards;
    episodeHistory(episode).steps = stepCount;
    episodeHistory(episode).epsilon = epsilon;

    % Track Q-value statistics for convergence monitoring
    qValueStats.mean(episode) = mean(Q(:));
    qValueStats.std(episode) = std(Q(:));
    qValueStats.max(episode) = max(Q(:));
    qValueStats.min(episode) = min(Q(:));

    % Convergence check: if Q-value changes are very small for several episodes, assume convergence
    maxQChange = max(abs(Q - previousQ), [], 'all'); % Maximum absolute change in Q-values in this episode
    episodeHistory(episode).maxQChange = maxQChange;
    episodeHistory(episode).avgQValue = mean(Q(:));

    if maxQChange < params.convergenceTolerance % Check if max Q-value change is below tolerance
        convergenceCounter = convergenceCounter + 1; % Increment convergence counter
        if convergenceCounter >= params.convergenceEpisodes % Check if converged for enough consecutive episodes
            converged = true;
            fprintf('Converged after %d episodes\n', episode);
            break; % Break out of training loop if converged
        end
    else
        convergenceCounter = 0; % Reset counter if not converged in this episode
    end

    previousQ = Q; % Update previous Q-table for convergence check in the next episode
end

fprintf('\nTraining Finished with %d episodes.\n', episode);

%% Analysis and Visualization

% Plot Q-value statistics over episodes to visualize training progress
figure('Name', 'Mean Q-value Over Episodes');
plot(1:length(qValueStats.mean), qValueStats.mean);
title('Mean Q-value Over Episodes');
xlabel('Episode'); ylabel('Mean Q-value');
grid on;

figure('Name', 'Q-value Standard Deviation');
plot(1:length(qValueStats.std), qValueStats.std);
title('Q-value Standard Deviation');
xlabel('Episode'); ylabel('Std Dev');
grid on;

figure('Name', 'Max Q-value Over Episodes');
plot(1:length([episodeHistory.maxQChange]), [episodeHistory.maxQChange]);
title('Max Q-value Change per Episode');
xlabel('Episode'); ylabel('Max Change');
grid on;

figure('Name', 'Epsilon Decay');
plot([episodeHistory.epsilon]);
title('Epsilon Decay');
xlabel('Episode'); ylabel('Epsilon');
grid on;

% Visualize the optimal policy derived from the learned Q-values
[~, optimalPolicy] = max(Q, [], 2); % Get the action with the highest Q-value for each state
fprintf('\nOptimal Policy Visualization:\n');
visualizeGridWorld(optimalPolicy, env.obstacleState, env.terminalStates); % Display policy on grid

% Print the final learned Q-table to the command window
fprintf('\nFinal Q-table:\n');
disp(Q);