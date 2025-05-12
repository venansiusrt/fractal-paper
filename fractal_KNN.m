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

%% TRAINING PROCESS
tic;
KNN_model = fitcknn(Xtrain, Ytrain, 'NumNeighbors', 3, 'Distance', 'Euclidean');
time = toc;

%% PREDICTION
model = KNN_model;

Xval = datatrain(:,1:tail);
Yval = round(predict(model, Xval));
val_acc = sum(Ytrain == Yval)*100/length(Ytrain);

Ypred = round(predict(model, Xtest));
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

%% EVALUATE THE ACCURACY
train_acc = sum(wrong_label_val)/size(datatrain,1)*100;
testing_acc = sum(wrong_label_pred)/size(datatest,1)*100;