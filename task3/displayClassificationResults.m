function displayClassificationResults(allYPred, allYValidation, allValidationFiles)
    % Visualizes classification results by displaying correctly and 
    % incorrectly classified images.
    %
    % Parameters:
    %   allYPred - A vector of predicted labels for the validation dataset.
    %   allYValidation - A vector of true labels for the validation dataset.
    %   allValidationFiles - A cell array of file paths corresponding to 
    %                        the validation images.

    
    % Find indices of correct and incorrect predictions
    correctIdx = find(allYPred == allYValidation);
    incorrectIdx = find(allYPred ~= allYValidation);

    % Plot incorrect predictions
    if ~isempty(incorrectIdx)
        figure('Name', 'Incorrect Predictions');
        numIncorrect = numel(incorrectIdx);
        for i = 1:numIncorrect
            % Read the incorrectly classified image
            I = imread(allValidationFiles{incorrectIdx(i)});

            % Create a subplot for each incorrect image
            subplot(ceil(sqrt(numIncorrect)), ceil(sqrt(numIncorrect)), i);

            % Display the image
            imshow(I);

            % Show the incorrect label and the correct label
            title(sprintf('Predicted: %s\nTrue: %s', ...
                string(allYPred(incorrectIdx(i))), ...
                string(allYValidation(incorrectIdx(i)))));
        end
    else
        disp('No incorrect predictions.');
    end

    % Plot correct predictions
    if ~isempty(correctIdx)
        figure('Name', 'Correct Predictions');
        numCorrect = numel(correctIdx);
        for i = 1:numCorrect
            % Read the correctly classified image
            I = imread(allValidationFiles{correctIdx(i)});

            % Create a subplot for each correct image
            subplot(ceil(sqrt(numCorrect)), ceil(sqrt(numCorrect)), i);

            % Display the image
            imshow(I);

            % Show the correct label
            title(sprintf('Correct: %s', string(allYPred(correctIdx(i)))));
        end
    else
        disp('No correct predictions.');
    end

end
