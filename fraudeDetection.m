clc
clear

% Load the Credit Card Fraud Detection dataset from Kaggle
dataset = readtable('creditcard.csv');

% Define the features and target variables for the model
features = dataset.Properties.VariableNames(1:end-1);
target = dataset.Properties.VariableNames(end);

% Split the dataset into training and testing sets
cv = cvpartition(height(dataset),'HoldOut',0.2);
training_data = dataset(training(cv),:);
testing_data = dataset(test(cv),:);

%------------------------------------------------TRAINING-----------------------------------------------

% Train a random forest model on the training set
mdl_rf = TreeBagger(3, training_data{:,features}, training_data{:,target}, 'Method', 'classification');
pred_target_rf = predict(mdl_rf, testing_data{:,features});

% Convert pred_target_rf to a numeric array
pred_target_rf = str2double(pred_target_rf);

% Train a logistic regression model on the training set
mdl_lr = fitglm(training_data, 'linear', 'Distribution', 'binomial');
pred_target_lr = predict(mdl_lr, testing_data{:,features});

% Convert pred_target_lr to a numeric array
pred_target_lr = round(pred_target_lr);

% Train a support vector machine (SVM) model on the training set
subset_idx = randperm(size(training_data, 1), round(0.2*size(training_data, 1)));
mdl_svm = fitcsvm(training_data{subset_idx,features}, training_data{subset_idx,target}, 'KernelFunction', 'linear', 'BoxConstraint', 1, 'Standardize', true);
pred_target_svm = predict(mdl_svm, testing_data{:,features});



%---------------------------------EVALUATING----------------------------------

% Evaluate the performance of the models using precision, recall, and F1 score
conf_mat_rf = confusionmat(testing_data{:,target}, pred_target_rf);
precision_rf = conf_mat_rf(2,2)/(conf_mat_rf(2,2)+conf_mat_rf(1,2));
recall_rf = conf_mat_rf(2,2)/(conf_mat_rf(2,2)+conf_mat_rf(2,1));
F1_score_rf = 2*(precision_rf*recall_rf)/(precision_rf+recall_rf);


conf_mat_lr = confusionmat(testing_data{:,target}, pred_target_lr);
precision_lr = conf_mat_lr(2,2)/(conf_mat_lr(2,2)+conf_mat_lr(1,2));
recall_lr = conf_mat_lr(2,2)/(conf_mat_lr(2,2)+conf_mat_lr(2,1));
F1_score_lr = 2*(precision_lr*recall_lr)/(precision_lr+recall_lr);

conf_mat_svm = confusionmat(testing_data{:,target}, pred_target_svm);
precision_svm = conf_mat_svm(2,2)/(conf_mat_svm(2,2)+conf_mat_svm(1,2));
recall_svm = conf_mat_svm(2,2)/(conf_mat_svm(2,2)+conf_mat_svm(2,1));
F1_score_svm = 2*(precision_svm*recall_svm)/(precision_svm+recall_svm);


%-----------------------------------------DISPLAYING------------------------------------------

% Display the performance metrics for Random Forrest
fprintf('Random Forest:\n');
fprintf('Accuracy: %0.2f%%\n', (conf_mat_rf(1,1)+conf_mat_rf(2,2))/sum(sum(conf_mat_rf))*100);
fprintf('Precision: %0.2f%%\n', precision_rf*100);
fprintf('Recall: %0.2f%%\n', recall_rf*100);
fprintf('F1 Score: %0.2f%%\n', F1_score_rf*100);
fprintf('\n');

% Display the performance metrics Logistic Regression
fprintf('Logistic Regression:\n');
fprintf('Accuracy: %0.2f%%\n', (conf_mat_lr(1,1)+conf_mat_lr(2,2))/sum(sum(conf_mat_lr))*100);
fprintf('Precision: %0.2f%%\n', precision_lr*100);
fprintf('Recall: %0.2f%%\n', recall_lr*100);
fprintf('F1 Score: %0.2f%%\n', F1_score_lr*100);

fprintf('\n');
% Display the performance metrics Support Vector Machines
fprintf('Support Vector Machines:\n');
fprintf('Accuracy: %0.2f%%\n', (conf_mat_svm(1,1)+conf_mat_svm(2,2))/sum(sum(conf_mat_svm))*100);
fprintf('Precision: %0.2f%%\n', precision_svm*100);
fprintf('Recall: %0.2f%%\n', recall_svm*100);
fprintf('F1 Score: %0.2f%%\n', F1_score_svm*100);


%------------------------------------PLOTTING---------------------------------------------


% Plot the ROC curve for Random Forest
[~,score_rf] = predict(mdl_rf,testing_data{:,features});
figure;
plotroc(testing_data{:,target}',score_rf(:,2)');
title('Random Forest ROC Curve');


% Plot the ROC curve for Logistic Regression
[~,score_lr] = predict(mdl_lr,testing_data{:,features});
figure;
plotroc(testing_data{:,target}',score_lr(:,2)');
title('Logistic Regression ROC Curve');


% Plot the ROC curve for SVM
[~,score_svm] = predict(mdl_svm,testing_data{:,features});
figure;
plotroc(testing_data{:,target}',score_svm(:,2)');
title('SVM ROC Curve');

