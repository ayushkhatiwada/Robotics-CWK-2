% Choose which fruit picture set to pick (cut, uncut, mix of both)
folderName = 'FruitPicsMix';

% Configurable Parameters
optimizerType = "sgdm";                    % Optimizer ("sgdm", "adam", "rmsprop")
initialLearnRate = 1e-4;                   % Initial learning rate
maxEpochs = 6;                             % Maximum number of epochs
miniBatchSize = 10;                        % Mini-batch size
validationFrequency = 3;                   % Validation frequency (in iterations)
shuffleOption = "every-epoch";             % Shuffle data ("every-epoch", "once", "never")
finalLayerWeightLearnRateFactor = 20;      % Learning rate factor for final layer weights
finalLayerBiasLearnRateFactor = 20;        % Learning rate factor for final layer biases
metrics = "accuracy";                      % Metric to monitor during training
numTrials = 1;                             % Number of trials for training

% Initialize variables to store results across all trials
accuracies = zeros(1, numTrials);          % Store accuracy for each trial
allYPred = [];                             % Store predictions across trials
allYValidation = [];                       % Store true labels across trials
allValidationFiles = {};                   % Store image file paths for all validation images

% Loop through multiple training trials
for trial = 1:numTrials
    fprintf('Training trial %d of %d...\n', trial, numTrials);

    % Load and preprocess training and validation datasets
    [imdsTrain, imdsValidation, numClasses, classNames] = prepareData(folderName);

    % Store validation file paths for all trials
    allValidationFiles = [allValidationFiles; imdsValidation.Files];

    % Load AlexNet and configure it for the new classification task
    net = imagePretrainedNetwork("alexnet", NumClasses=numClasses);

    % Adjust learning rates for the final classification layer
    net = setLearnRateFactor(net, "fc8/Weights", finalLayerWeightLearnRateFactor);
    net = setLearnRateFactor(net, "fc8/Bias", finalLayerBiasLearnRateFactor);

    % Apply data augmentation to enhance training and validation datasets
    [augimdsTrain, augimdsValidation] = augmentData(net, imdsTrain, imdsValidation);

    % Set training options
    options = trainingOptions(optimizerType, ...
        MiniBatchSize=miniBatchSize, ...
        MaxEpochs=maxEpochs, ...
        Metrics=metrics, ...
        InitialLearnRate=initialLearnRate, ...
        Shuffle=shuffleOption, ...
        ValidationData=augimdsValidation, ...
        ValidationFrequency=validationFrequency, ...
        Verbose=false, ...
        Plots="training-progress");

    % Train the network using cross-entropy loss
    trainedNet = trainnet(augimdsTrain, net, "crossentropy", options);

    % Predict labels on the validation set
    scores = minibatchpredict(trainedNet, augimdsValidation);
    YPred = scores2label(scores, classNames);
    YValidation = imdsValidation.Labels;

    % Aggregate predictions and true labels
    allYPred = [allYPred; YPred];
    allYValidation = [allYValidation; YValidation];

    % Calculate and store accuracy for the current trial
    accuracy = mean(YPred == YValidation);
    accuracies(trial) = accuracy;
end

% Calculate mean and standard deviation of accuracies
meanAccuracy = mean(accuracies);
stdAccuracy = std(accuracies);

% Display results
fprintf('Mean validation accuracy across %d trials: %.2f%%\n', numTrials, meanAccuracy * 100);
fprintf('Standard deviation of validation accuracies: %.2f%%\n', stdAccuracy * 100);

% Display cumulative confusion matrix
confusionchart(allYValidation, allYPred);
title(sprintf('Cumulative Confusion Matrix - %s', folderName));

% Display classification results
displayClassificationResults(allYPred, allYValidation, allValidationFiles);
