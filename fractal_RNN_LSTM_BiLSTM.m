clc; clear;

%% IMPORT THE DATASET
datatrain = xlsread("New_Mandelbrot_Dataset_Train_10000_0.xlsx");
datatest = xlsread('Mandelbrot_Dataset_Test.xlsx');

%% CONFIGURE THE ATTRIBUTES
tail = 8;
Xtrain = datatrain(:,1:tail);
Ytrain = datatrain(:,9);
Xtest = datatest(:,1:tail);
Ytest = datatest(:,9);

%% CREATE A NEURAL NETWORK AND ITS OPTIMIZER
numFeatures = size(datatrain(:,1:tail+1),2)-1;
numResponses = 2;
neuron = round(numFeatures*(2/3) + numResponses);
hiddenLayerRNN = round((2/3)*(numFeatures + numResponses));
layers = [ sequenceInputLayer(numFeatures) ...
           fullyConnectedLayer(neuron) ...
           lstmLayer(hiddenLayerRN) ...
%            bilstmLayer(hiddenLayerRN) ...
           fullyConnectedLayer(numResponses) ...
           softmaxLayer ...
           classificationLayer
         ];

options = trainingOptions('sgdm', ...
    'MaxEpochs', 200, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', 0.005, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 25, ...
    'LearnRateDropFactor', 0.2, ...
    'Verbose', 1);

%% TRAINING PROCESS
tic;
Xtrain = Xtrain';
Ytrain = categorical(Ytrain');
LSTM = trainNetwork(Xtrain, Ytrain, layers, options);
time = toc;

%% PREDICTION
model = LSTM;
Ytrain = double(Ytrain)-1;

Xval = datatrain(:,1:tail);
val = classify(LSTM, Xtrain); Yval = double(val) - 1;
val_acc = sum(Ytrain == Yval)*100/length(Ytrain);

Xtest = datatest(:,1:tail)';
Ytest = datatest(:,9)';
pred = classify(LSTM, Xtest); Ypred = double(pred) - 1;
pred_acc = sum(Ytest == Ypred)*100/length(Ytest);

wrong_label_val = []; wrong_label_pred = [];
for i = 1:length(Ytrain)
    if Yval(i) == Ytrain(i)
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

Xtrain = Xtrain';
Xtest = Xtest';

if length(unique(Ypred)) == 1
    Ypred(randi(10)) = 0;
end
if length(unique(Yval)) == 1
    Yval(randi(10)) = 0;
end

train_acc = sum(wrong_label_val)/size(datatrain,1)*100;
testing_acc = sum(wrong_label_pred)/size(datatest,1)*100;