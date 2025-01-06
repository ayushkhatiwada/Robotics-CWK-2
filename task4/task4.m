%% ELEC0144 - Machine Learning for Robotics
% Task 4: Tabular Q-Learning

%% Configuration Section (All hard-coded values here)

% Q-learning parameters
learningRate = 0.1;   % Alpha
discountFactor = 0.9; % Gamma
initialEpsilon = 0.9;
epsilonDecayRate = 0.006;
maxEpisodes = 1000;
maxSteps = 100;

% Grid world definition
numStates = 12;
numActions = 4;
obstacleState = 6;
terminalStates = [8, 12];
rewards = zeros(1, numStates);
rewards(8) = -10;
rewards(12) = 10;
livingReward = -1;

%% Section 1: Environment Setup

% Action indices and labels for better readability
actions = ["UP", "RIGHT", "DOWN", "LEFT"];
actionIndex.UP = 1;
actionIndex.RIGHT = 2;
actionIndex.DOWN = 3;
actionIndex.LEFT = 4;

% Q-table initialization
Q = zeros(numStates, numActions);

%% Section 2: Q-Learning Algorithm

% Function to determine next state and reward
function [nextState, reward] = takeAction(state, action, rewards, livingReward, obstacleState, actionIndex)
    switch action
        case actionIndex.UP
            nextState = state + 4;
            if state >= 9 || nextState == obstacleState
                nextState = state; % Stay in the same state
            end
        case actionIndex.RIGHT
            nextState = state + 1;
            if mod(state, 4) == 0 || nextState == obstacleState
                nextState = state;
            end
        case actionIndex.DOWN
            nextState = state - 4;
            if state <= 4 || nextState == obstacleState
                nextState = state;
            end
        case actionIndex.LEFT
            nextState = state - 1;
            if mod(state, 4) == 1 || nextState == obstacleState
                nextState = state;
            end
    end
    reward = rewards(nextState) + livingReward;
end

% Q-learning algorithm
epsilon = initialEpsilon;
trainingLossHistory = []; % for storing the training loss of each episode
convergence = false;

for episode = 1:maxEpisodes
    state = 1; % Start from state 1 in each episode
    totalReward = 0; % to calculate loss

    for step = 1:maxSteps
        % Choose action using epsilon-greedy policy
        if rand() < epsilon
            action = randi(numActions); % Explore: choose a random action
        else
            [~, action] = max(Q(state, :)); % Exploit: choose the best action based on Q-table
        end

        % Take action and observe the next state and reward
        [nextState, reward] = takeAction(state, action, rewards, livingReward, obstacleState, actionIndex);

        % Q-learning update rule
        bestNextActionValue = max(Q(nextState, :));
        Q(state, action) = Q(state, action) + learningRate * (reward + discountFactor * bestNextActionValue - Q(state, action));

        % Update state and total reward
        state = nextState;
        totalReward = totalReward + reward;

        % Check if the terminal state is reached
        if ismember(state, terminalStates)
            break;
        end

        % Print the first 3 iterations of the first 3 episodes
        if episode <= 3
            if step <= 3
                disp("------------------------");
                disp("Episode: " + num2str(episode) + ", Step: " + num2str(step));
                fprintf('State: %d, Action: %s, Next State: %d, Reward: %.2f\n', ...
                        state-4*(state>8)*(state~=12), actions(action), nextState-4*(nextState>8)*(nextState~=12), reward);
                disp("Q-table (partially):");
                disp(Q(max(1,state-4):min(numStates,state+4),:)); % only show a part of Q table for better format
            end
        end
    end

    % Store the loss of this episode
    trainingLossHistory = [trainingLossHistory totalReward];

    % Epsilon decay
    epsilon = initialEpsilon * exp(-epsilonDecayRate * episode);

    % Check for convergence
    if episode > 4
        if abs(trainingLossHistory(episode) - trainingLossHistory(episode-1)) < 1e-4 && ...
           abs(trainingLossHistory(episode-1) - trainingLossHistory(episode-2)) < 1e-4 && ...
           abs(trainingLossHistory(episode-2) - trainingLossHistory(episode-3)) < 1e-4 && ...
           abs(trainingLossHistory(episode-3) - trainingLossHistory(episode-4)) < 1e-4
            disp("Converged at episode " + num2str(episode));
            convergence = true;
            break;
        end
    end
end

%% Section 3: Results and Visualization

% Determine the best action for each state
[~, bestActions] = max(Q, [], 2);

% Display the Q-table
disp("Q-table:");
disp(Q);

% Function to visualize the grid world and policy
function visualizePolicy(bestActions, obstacleState, terminalStates)
    gridSymbols = ["9" "10" "11" "12"; "5" "6" "7" "8"; "1" "2" "3" "4"];
    policySymbols = ["↑", "→", "↓", "←"];

    % Mark obstacle and terminal states
    gridSymbols(obstacleState) = "X";
    gridSymbols(terminalStates(1)) = "T1";
    gridSymbols(terminalStates(2)) = "T2";

    % Create the policy grid
    policyGrid = strings(3, 4);
    for row = 1:3
        for col = 1:4
            state = (3 - row) * 4 + col;
            if ismember(state, [obstacleState, terminalStates])
                policyGrid(row, col) = gridSymbols(state);
            else
                policyGrid(row, col) = policySymbols(bestActions(state));
            end
        end
    end

    % Display the policy grid
    disp(policyGrid);
end

% Visualize the learned policy
visualizePolicy(bestActions, obstacleState, terminalStates);

% Plot the training loss curve
figure;
plot(1:length(trainingLossHistory), trainingLossHistory);
title('Training Loss Over Episodes');
xlabel('Episode');
ylabel('Total Reward (Loss)');
grid on;

% Display a message if the algorithm did not converge within maxEpisodes
if ~convergence
    disp("Did not converge within " + num2str(maxEpisodes) + " episodes.");
end