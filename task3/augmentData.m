function [augimdsTrain, augimdsValidation] = augmentData(net, imdsTrain, imdsValidation)
    % Performs data augmentation and resizing for training and validation data.
    %
    % Parameters:
    %   net - Pretrained network to retrieve input size.
    %   imdsTrain - Training image datastore.
    %   imdsValidation - Validation image datastore.
    %
    % Returns:
    %   augimdsTrain - Augmented training image datastore.
    %   augimdsValidation - Augmented validation image datastore.

    % Retrieve the input size of the network
    inputSize = net.Layers(1).InputSize;

    % Define the range of random translations (±30 pixels)
    pixelRange = [-30 30];
    
    % Create an image augmenter with the specified operations
    imageAugmenter = imageDataAugmenter( ...
        'RandXReflection', true, ...  % Randomly flip images horizontally
        'RandXTranslation', pixelRange, ...  % Random horizontal shift
        'RandYTranslation', pixelRange);     % Random vertical shift

    % Create an augmented image datastore for training data
    % Automatically resizes the images to the required input size of the network
    augimdsTrain = augmentedImageDatastore(inputSize(1:2), imdsTrain, ...
        'DataAugmentation', imageAugmenter);

    % Create an augmented image datastore for validation data
    % Resizes the images but does not apply additional augmentations
    augimdsValidation = augmentedImageDatastore(inputSize(1:2), imdsValidation);
end
