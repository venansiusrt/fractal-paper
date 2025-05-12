clc;
clear all;
close all;

% Define dataset directory
datasetDir = 'D:\Mandelbrot Project';

% Initialize results storage
results = struct();

I = 4;

% Loop through all types (0 to 9)
for type = 0:9
    fprintf('Processing type %d:\n', type);
    
    % Load dataset
    filename = sprintf("New_Julia_Dataset_Train_10000_%d.xlsx", type);
    datatrain = xlsread(fullfile(datasetDir, filename));
    
    % Extract predictors (columns 1-8) and target (column 9)
    X = datatrain(:, 1:8);  
    y = datatrain(:, 9);    
    
    %% Gini Impurity Analysis (Columns 1-4)
    gini_scores = zeros(1, I);
    for col = 1:4
        gini_scores(col) = computeGini(X(:, col), y);
    end
    [~, gini_rank] = sort(gini_scores);
    
    %% Chi-Squared Analysis (Columns 1-4)
    % Discretize features into 3 bins
    X_discretized = discretize(X(:, 1:I), 3);  
    
    chi2_scores = zeros(1, I);
    for col = 1:I
        tbl = crosstab(X_discretized(:, col), y);
        [~, chi2_scores(col), ~] = chi2test(tbl);
    end
    [~, chi2_rank] = sort(chi2_scores, 'descend');
    
    %% Store results
    results(type+1).type = type;
    results(type+1).gini_scores = gini_scores;
    results(type+1).gini_rank = gini_rank;
    results(type+1).chi2_scores = chi2_scores;
    results(type+1).chi2_rank = chi2_rank;
    
    %% Display summary
    fprintf('Type %d:\n', type);
    fprintf('  Gini Scores:    [%s]\n', num2str(gini_scores, '%.2f  '));
    fprintf('  Gini Rank:      [%s]\n', num2str(gini_rank));
    fprintf('  Chi2 Scores:    [%s]\n', num2str(chi2_scores, '%.2f  '));
    fprintf('  Chi2 Rank:      [%s]\n\n', num2str(chi2_rank));
end

%% Save results to MAT file
save('feature_analysis_results.mat', 'results');

%% Helper Functions
function gini = computeGini(feature, target)
    values = unique(feature);
    gini = 0;
    for i = 1:length(values)
        mask = (feature == values(i));
        subset = target(mask);
        p0 = sum(subset == 0) / length(subset);
        p1 = sum(subset == 1) / length(subset);
        gini_subset = 1 - (p0^2 + p1^2);
        gini = gini + (length(subset) / length(target)) * gini_subset;
    end
end

function [h, chi2, p] = chi2test(tbl)
    [rows, cols] = size(tbl);
    expected = sum(tbl, 2) * sum(tbl, 1) / sum(tbl(:));
    chi2 = sum((tbl - expected).^2 ./ expected, 'all');
    df = (rows-1)*(cols-1);
    p = 1 - chi2cdf(chi2, df);
    h = p < 0.05;
end