% Define the file path
data_path = '~/s&p500_closing_prices.csv';
spy_data_path = '~/spy_historical_prices.csv';
% Use the 'readmatrix' function with the appropriate options for Excel files
df = readmatrix(data_path);
spy_df = readmatrix(spy_data_path);

% Remove columns with null values in dataframe
df = df(:, ~any(isnan(df)));

% TrainingStocks = df(1:604, 2:end);
% TestStocks = df(605:end, 2:end);
% spy_train = spy_df(1:604, 2:end);
% spy_test = spy_df(605:end, 2:end);

stocks = df(:, 2:end);
spy = spy_df(:, 2:end);

%%
% Calculate returns for all stocks
returnsAll = price2ret(stocks);
spy_returns = price2ret(spy);

% Perform PCA
[coeffAll, ~, ~, ~, explained, ~] = pca(returnsAll);

% Select the first principal component (eigenportfolio) for all stocks
eigenportfolioAll = coeffAll(:, 1);

%%
% Normalize the eigenportfolio weights to sum to 1
weightsAll = eigenportfolioAll / sum(abs(eigenportfolioAll));

% Set the sliding window to our desired length
windowLength = 30;  % Adjust as needed do other lengths to test (10, 30, and 50)

% Initialize matrices to store alpha, beta, and residuals for the portfolio
alphaPortfolioAll = zeros(size(returnsAll, 1) - windowLength + 1, 1);
betaPortfolioAll = zeros(size(returnsAll, 1) - windowLength + 1, 1);
residualsPortfolioAll = zeros(size(returnsAll, 1) - windowLength + 1, 1);

% Perform sliding window regression for the portfolio
for t = 1:size(returnsAll, 1) - windowLength + 1
    windowReturnsAll = returnsAll(t : t + windowLength - 1, :);
    
    % Calculate the portfolio return
    portfolioReturnAll = windowReturnsAll * weightsAll;
    
    % Extract returns of the market (or any other benchmark)
    marketReturnAll = spy_returns(t : t + windowLength - 1);
    
    % Perform linear regression
    XPortfolioAll = [ones(windowLength, 1) marketReturnAll];
    betaAlphaPortfolioAll = XPortfolioAll \ portfolioReturnAll;  % Coefficients (beta, alpha)
    
    % Store coefficients and residuals
    betaPortfolioAll(t) = betaAlphaPortfolioAll(2);
    alphaPortfolioAll(t) = betaAlphaPortfolioAll(1);
    residualsPortfolioAll(t) = portfolioReturnAll(end) - betaAlphaPortfolioAll(2) * marketReturnAll(end) - betaAlphaPortfolioAll(1);
end

% For every dollar long our eigenportfolio, we short Beta * dollar of SPY
% or in other words our Market

%%
% Plot residuals over time for the portfolio
figure
plot(residualsPortfolioAll)
title('Residuals for Eigenportfolio and Market')
xlabel('Time')
ylabel('Residuals')

figure
plot(residualsPortfolioAll)

hold on

% Add the mean line
meanResidual = mean(residualsPortfolioAll);
plot([1, size(residualsPortfolioAll, 1)], [meanResidual, meanResidual], '--', 'LineWidth', 1, 'Color', 'k', 'DisplayName', 'Mean')

% Add z-scores lines
zScores = [1.0, -1.0, 1.25, -1.25];

for z = zScores
    zLine = repmat(meanResidual + z * std(residualsPortfolioAll), size(residualsPortfolioAll));
    plot(zLine, '--', 'LineWidth', 1, 'DisplayName', ['Z = ' num2str(z)]);
end

hold off

title('Residuals for Eigenportfolio and Market')
xlabel('Time')
ylabel('Residuals')
legend('Location', 'best')
grid on

X = cumsum(residualsPortfolioAll);
figure
plot(X)
xlabel('Days')
ylabel('Cumulative PnL')

%%
train_data = transpose(X(1:604, :));
test_data = transpose(X(605:end, :));

numTimeStepsTrain = numel(train_data)-1;

mu = mean(train_data);
sigma = std(train_data);

% Standardize our data using the train data mean and standard deviation
dataStandardized_train = (train_data - mu) / sigma;

% Initialize our independent and dependent variables (time-series)
X_Train = dataStandardized_train(1:end-1);
Y_Train = dataStandardized_train(2:end);

ads_X_Train = arrayDatastore(X_Train);
ads_Y_Train = arrayDatastore(Y_Train);

% Set up for training our LSTM model
cds_Train = combine(ads_X_Train, ads_Y_Train);

%%

% Use different layers, parameters, and hyperparameters to find desired
% model
numFeatures = 1;
numHiddenUnits = 250;
numResponses = 1;

% layers = [ ...
%     sequenceInputLayer(numFeatures)
%     lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
%     dropoutLayer(0.2)
%     lstmLayer(numHiddenUnits/2, 'OutputMode', 'sequence')
%     dropoutLayer(0.2)
%     fullyConnectedLayer(numResponses)
%     reluLayer
%     fullyConnectedLayer(numResponses)
%     regressionLayer];

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits,'OutputMode','sequence')
    dropoutLayer
    fullyConnectedLayer(numResponses)
    regressionLayer];

% layers = [ ...
%     sequenceInputLayer(numFeatures)
%     lstmLayer(numHiddenUnits, 'OutputMode', 'sequence')
%     dropoutLayer(0.2)
%     reluLayer
%     fullyConnectedLayer(numResponses)
%     regressionLayer];


options = trainingOptions("adam", ...
    MaxEpochs=700, ...
    InitialLearnRate=0.005,...
    Shuffle="every-epoch", ...
    GradientThreshold=1, ...
    Verbose=false, ...
    Plots="training-progress");

trainedNetwork_1=trainNetwork(cds_Train,layers,options);

%%
numFeatures = 1;
numHiddenUnits = 250;
numResponses = 1;

layers_gru = [ ...
    sequenceInputLayer(numFeatures)
    gruLayer(numHiddenUnits, 'OutputMode', 'sequence')
    dropoutLayer
    fullyConnectedLayer(numResponses)
    regressionLayer];

options_gru = trainingOptions("adam", ...
    'MaxEpochs', 700, ...
    'InitialLearnRate', 0.005, ...
    'Shuffle', 'every-epoch', ...
    'GradientThreshold', 1, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

trainedNetwork_gru = trainNetwork(cds_Train, layers_gru, options_gru);


%%
% Standardize our test data with our train data mean and standard deviation
dataStandardized_test = (test_data - mu) / sigma;

X_Test = dataStandardized_test(1:end-1);
Y_Test = test_data(2:end);

% Update our network with the training independent data
% network = predictAndUpdateState(trainedNetwork_1, X_Train);
network = predictAndUpdateState(trainedNetwork_gru, X_Train);

%%
% Predict and update our network on our last datapoint of our train data
[network, Y_Pred] = predictAndUpdateState(network, Y_Train(end));

% Predict next day's return based on our predictions of the day before
numTimeStepsTest = numel(X_Test);
for i = 2:numTimeStepsTest
    [network,Y_Pred(:,i)] = predictAndUpdateState(network,Y_Pred(:,i-1), 'ExecutionEnvironment', 'cpu');
end

% De-Standardize our prediction to compare (RMSE for performance
% comparison)
Y_Pred = sigma*Y_Pred + mu;
rmse = sqrt(mean((Y_Pred - Y_Test).^2));

%%
figure
plot(train_data(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[train_data(numTimeStepsTrain) Y_Pred], '.-')
hold off
xlabel('Days')
ylabel('Cumulative PnL')
title("GRU Forecast")
legend(["Observed" "Forecast"])

%%

figure
subplot(2,1,1)
plot(Y_Test)
hold on
plot(Y_Pred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cumulative PnL")
title("GRU Forecast")

subplot(2,1,2)
stem(Y_Pred - Y_Test)
xlabel("Day")
ylabel("Error")
title("RMSE = " + rmse)

%%
% Reset our network state to the initial trained netword and use train data
% to train the model again
network = resetState(network);
network = predictAndUpdateState(network, X_Train);

% Predict and update our model and our dependent test data based on our
% independent train data
Y_Pred = [];
numTimeStepsTest = numel(X_Test);
for i = 1:numTimeStepsTest
    [network, Y_Pred(:, i)] = predictAndUpdateState(network, X_Test(:, i), 'ExecutionEnvironment', 'cpu');
end

% De-Standardize our prediction to compare (EMSE for performance
% comparison)
Y_Pred = sigma*Y_Pred + mu;
rmse = sqrt(mean((Y_Pred - Y_Test).^2));

figure
subplot(2,1,1)
plot(Y_Test)
hold on
plot(Y_Pred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cumulative PnL")
title("Forecast with Updates")

subplot(2,1,2)
stem(Y_Pred - Y_Test)
xlabel("Day")
ylabel("Error")
title("RMSE = " + rmse)


%%
% Backtest our model on the historical data in our training sample
% Still needs to be worked as our backtesting code is not fully functional

% Initialize capital and holdings
initialCapital = 10000000;  % Initial capital in USD
capital = initialCapital;
position = 0;  % Initial position, 0 means no position

% Define thresholds for trading signals
buyThreshold = 0.0005;  % Example threshold for a buy signal
sellThreshold = -0.0005;  % Example threshold for a sell signal

% Initialize arrays to track trading actions and P/L
actions = zeros(size(Y_Pred));  % 1 for buy, -1 for sell, 0 for no action
PnL = zeros(size(Y_Pred));

% Backtesting loop
for i = 2:numel(residualsPortfolioAll(608:end,:))
    % Determine trading actions based on predicted returns
    if Y_Pred(i) > 0.01  % Can adjust the threshold
        actions(i) = 1;  % Buy signal
        position = position + 1;
    elseif Y_Pred(i) < -0.01  % Can adjust the threshold
        actions(i) = -1;  % Sell signal
        position = position - 1;
    else
        actions(i) = 0;  % No action
    end
    
    % Calculate PnL for the day using residuals
    PnL(i) = position * residualsPortfolioAll(i);
    
    % Update capital
    capital = capital + PnL(i);
end

% Visualize the backtest results
figure;
subplot(2,1,1);
plot(residualsPortfolioAll(606:end, :));
title('Price Data');

subplot(2,1,2);
plot(actions, 'o', 'MarkerSize', 5);
title('Trading Actions');

% Display backtest statistics
fprintf('Initial Capital: $%.2f\n', initialCapital);
fprintf('Final Capital: $%.2f\n', capital);
fprintf('Total P/L: $%.2f\n', capital - initialCapital);
fprintf('Total Trades: %d\n', sum(actions ~= 0));
fprintf('Winning Trades: %d\n', sum(PnL > 0));
fprintf('Losing Trades: %d\n', sum(PnL < 0));


