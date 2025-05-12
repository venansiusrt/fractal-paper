clc; clear;

%% IMPORT THE DATASET
datatrain = xlsread("New_Mandelbrot_Dataset_Train_10000_0.xlsx");
datatest = xlsread('Mandelbrot_Dataset_Test.xlsx');

%% TRAIN A RANDOM FOREST MODEL
tail = 8;
Xtrain = datatrain(:,1:tail);
Ytrain = datatrain(:,9);
Xtest = datatest(:,1:tail);
Ytest = datatest(:,9);

%% CLASSIFICATION TEST (TRAINING SET)
tic;
numTrees = 100;
RFModel = TreeBagger(numTrees, XTrain, YTrain, ...
    'Method', 'classification', ...
    'OOBPrediction', 'on', ...
    'OOBVarImp', 'on'); % Enable Out-of-Bag (OOB) error estimation
time = toc;

%% PREDICTION
Yval = str2double(RFModel.predict(XTrain)); % Predictions on training data
Ypred = str2double(RFModel.predict(Xtest)); % Predictions on test data

wrong_label_val = []; wrong_label_pred = [];
for i = 1:length(YTrain)
    if Yval(i) == YTrain(i)
        wrong_label_val(i) = 1;
    else
        wrong_label_val(i) = 0;
    end
end
for i = 1:length(Ytest)
    if Ypred(i) == Ytest(i)
        wrong_label_pred(i) = 1;
    else
        wrong_label_pred(i) = 0;
    end
end

%% EVALUATE THE ACCURACY
train_acc = sum(wrong_label_val)/size(datatrain,1)*100;
testing_acc = sum(wrong_label_pred)/size(datatest,1)*100;