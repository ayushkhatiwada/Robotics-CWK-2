function [imdsTrain, imdsValidation, numClasses, classNames] = prepareData(folderName)
    % Function to prepare data for training and validation
    %
    % Parameters:
    %   folderName - Name of the folder containing the dataset
    %
    % Returns:
    %   imdsTrain - Image datastore for training
    %   imdsValidation - Image datastore for validation
    %   numClasses - Number of unique classes in the dataset
    %   classNames - Names of the unique classes in the dataset

    % Create an image datastore from the folder
    imds = imageDatastore(folderName, ...
        'IncludeSubfolders', true, ...
        'LabelSource', 'foldernames');
    
    % Split the datastore into training and validation sets
    % 70% Training, 30% Validation
    [imdsTrain, imdsValidation] = splitEachLabel(imds, 0.7, 'randomized');
    
    % Get class names and number of classes
    classNames = categories(imdsTrain.Labels);
    numClasses = numel(classNames);
end
