function plotConfusionMatrix(net, augimdsValidation, imdsValidation, classNames)
    % Function to create and display a confusion matrix for validation data
    %
    % Parameters:
    %   net - Trained neural network
    %   augimdsValidation - Augmented image datastore for validation
    %   imdsValidation - Original image datastore for validation labels
    %   classNames - List of class names in the dataset
    %
    % This function computes the confusion matrix and displays it as a chart.

    % Predict classes for validation images
    scores = minibatchpredict(net, augimdsValidation);
    YPred = scores2label(scores, classNames);

    % Get the true labels from the validation datastore
    YValidation = imdsValidation.Labels;

    % Create the confusion matrix
    confusionMatrix = confusionmat(YValidation, YPred);

    % Display the confusion matrix as a chart
    figure;
    confusionchart(confusionMatrix, categories(YValidation));
    title('Confusion Matrix');
end
