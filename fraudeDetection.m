clc

% Load the Credit Card Fraud Detection dataset from Kaggle
dataset = readtable('creditcard.csv');

% Remove irrelevant information from the dataset
dataset = removevars(dataset, {'Time'});

% Split the dataset into training and testing sets
cv = cvpartition(height(dataset),'HoldOut',0.2);
training_data = dataset(training(cv),:);
testing_data = dataset(test(cv),:);

% Separate fraudulent and non-fraudulent transactions
fraudulent = training_data(training_data.Class == 1,:);
non_fraudulent = training_data(training_data.Class == 0,:);


% Undersample the non-fraudulent transactions
undersampled_non_fraudulent = non_fraudulent(1:height(fraudulent),:);


% Oversample the fraudulent transactions using the Synthetic Minority Oversampling Technique (SMOTE)
mdl = fitcknn(fraudulent(:,1:end-1), fraudulent.Class,'NumNeighbors',5);
synth_fraudulent = predict(mdl, fraudulent(:,1:end-1));
synth_fraudulent_table = array2table(synth_fraudulent, 'VariableNames', {'Class'});
synth_fraudulent_table = horzcat(fraudulent(:,1:end-1), synth_fraudulent_table);


% Combine the undersampled non-fraudulent transactions and oversampled fraudulent transactions
training_data_resampled = vertcat(synth_fraudulent_table, undersampled_non_fraudulent);
