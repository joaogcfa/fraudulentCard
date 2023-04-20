clc

% Load the Credit Card Fraud Detection dataset from Kaggle
dataset = readtable('creditcard.csv');

% Remove irrelevant information from the dataset
dataset = removevars(dataset, {'Time'});

% Split the dataset into training and testing sets
cv = cvpartition(height(dataset),'HoldOut',0.2);
training_data = dataset(training(cv),:);
testing_data = dataset(test(cv),:);

