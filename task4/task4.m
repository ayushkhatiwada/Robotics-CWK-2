% Q-learning parameters
alpha = 0.1;
gamma = 0.9;
epsilon_start = 0.9;
epsilon = epsilon_start;
max_episodes = 1000;
max_steps = 100;
lambda = 0.006;

% Grid world definition
num_states = 12;
num_actions = 4;
obstacle = 6;
terminal_states = [8, 12];
rewards = zeros(1, num_states);
rewards(8) = -10;
rewards(12) = 10;
living_reward = -1;

% Q-table initialization
Q = zeros(num_states, num_actions);
Q_prev = Q;

% Action indices
UP = 1;
RIGHT = 2;
DOWN = 3;
LEFT = 4;

% Function to determine next state and reward
function [next_state, reward] = take_action(state, action, rewards, living_reward, obstacle)
    switch action
        case 1 % UP
            if state >= 9
                next_state = state;
            elseif state + 4 == obstacle
                next_state = state;
            else
                next_state = state + 4;
            end
        case 2 % RIGHT
            if mod(state, 4) == 0
                next_state = state;
            elseif state + 1 == obstacle
                next_state = state;
            else
                next_state = state + 1;
            end
        case 3 % DOWN
            if state <= 4
                next_state = state;
            elseif state - 4 == obstacle
                next_state = state;
            else
                next_state = state - 4;
            end
        case 4 % LEFT
            if mod(state, 4) == 1
                next_state = state;
            elseif state - 1 == obstacle
                next_state = state;
            else
                next_state = state - 1;
            end
    end
    reward = rewards(next_state) + living_reward;
end

% Q-learning algorithm
for episode = 1:max_episodes
    state = 1;
    epsilon = epsilon_start; % Epsilon for this episode
    
    for step = 1:max_steps
        % Choose action using epsilon-greedy policy
        a = rand();
        if a < epsilon
            action = randi(num_actions); % Random action
        else
            [~, action] = max(Q(state, :)); % Best action
        end

        % Take action and observe next state and reward
        [next_state, reward] = take_action(state, action, rewards, living_reward, obstacle);

        % Update Q-value
        best_next_action = max(Q(next_state, :));
        Q(state, action) = Q(state, action) + alpha * (reward + gamma * best_next_action - Q(state, action));

        % Move to next state
        state = next_state;

        % Check if terminal state is reached
        if ismember(state, terminal_states)
            break;
        end

        if episode <= 3
            if step <= 3
                disp("Episode: " + num2str(episode) + " Step: " + num2str(step));
                disp("Current State: " + num2str(state));
                disp("a: " + num2str(a));
                disp("Action: " + num2str(action));
                disp("Next State: " + num2str(next_state));
                disp("Q table");
                disp(Q);
            end
        end
    end

    if Q == Q_prev
        disp("Converged at episode " + num2str(episode));
        break;
    end

    Q_prev = Q;

    epsilon = epsilon_start * exp(-lambda * episode); % Decay epsilon

    if episode <= 3
        disp("Q-table for episode " + num2str(episode));
        disp(Q);
    end
end

% Determine best action for each state
[~, best_actions] = max(Q, [], 2);

% Display results
disp("Q-table");
disp(Q);
disp("Best actions:");
disp(best_actions);

% Function to visualize the grid world and policy
function visualize_policy(best_actions, obstacle)
    grid = ["9" "10" "11" "12"; "5" "6" "7" "8"; "1" "2" "3" "4"];  % Still a string array initially
    policy = ["↑" "→" "↓" "←"];
    
    for i = 1:3
        for j = 1:4
            state = (3-i)*4 + j;
            if state == obstacle
                continue;
            end
            grid(i,j) = {strcat(num2str(grid{i,j}), " ", policy(best_actions(state)))}; % Key change here!
        end
    end
    disp(grid)
end

% Visualize the learned policy
visualize_policy(best_actions, obstacle);