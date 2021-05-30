import pandas as pd
from pandas import DataFrame

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error


import sys
sys.path.append('../')

from consensus.algorithms import *
from tools.predictorsI import *
from tools.predictorsII import *
from tools.predictorsIII import *


def data_prep(df: DataFrame, input_batch_size: int, future_horizon: int) -> [(DataFrame, DataFrame)]:
    '''Takes in data and splits it into an input batch for the individual predictors to perform a prediction on and the the real values observed.

        Parameters:
            df (DataFrame): Whole data set to be divided into prediction batch and real values.
            input_batch_size (int): Length of input batch size.
            future_horizon (int): How many time steps are predicted into the future.

        Returns:
            [(DataFrame, DataFrame)]: Input batch dataframe and real value dataframe.
    '''
    input_b = df[0:input_batch_size]
    real_value = df[input_batch_size:input_batch_size + future_horizon]
    
    return input_b, real_value


def individual_predictors1(training_df: DataFrame, input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Handles the individual predictors by training them and feeding them the data to predict the specified future horizon. The following individual predictors are implemented here:

    1. CNN-LSTM
    2. Bidirectional LSTM
    3. LSTM

        Parameters:
            training_df (DataFrame): Data on which the predictors are trained on.
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''
    one = HybridUnivariatePredictor(training_df,2, len(input_batch), future_horizon)
    one.create_cnnlstm()
    one.fit_model(10)
    one.show_performance()
    
    two = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    two.create_bilstm()
    two.fit_model(10)
    two.show_performance()
    
    
    three = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    three.create_cnn()
    three.fit_model(10)
    three.show_performance()
    
    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three], axis=1) 

    return final_df


def individual_predictors2(training_df: DataFrame, input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Handles the individual predictors by training them and feeding them the data to predict the specified future horizon. The following individual predictors are implemented here:

    1. CNN-LSTM
    2. Bidirectional LSTM
    3. CNN
    4. MLP

        Parameters:
            training_df (DataFrame): Data on which the predictors are trained on.
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''
    one = HybridUnivariatePredictor(training_df,2, len(input_batch), future_horizon)
    one.create_cnnlstm()
    one.fit_model(10)
    one.show_performance()
    
    two = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    two.create_bilstm()
    two.fit_model(10)
    two.show_performance()
     
    three = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    three.create_cnn()
    three.fit_model(10)
    three.show_performance()

    four = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    four.create_mlp()
    four.fit_model(10)
    four.show_performance()
    
    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)
    prediction_four = four.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three, prediction_four], axis=1) 

    return final_df


def individual_predictors3(training_df: DataFrame, input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Handles the individual predictors by training them and feeding them the data to predict the specified future horizon. The following individual predictors are implemented here:

    1. CNN-LSTM
    2. Bidirectional LSTM
    3. CNN
    4. MLP
    5. LSTM

        Parameters:
            training_df (DataFrame): Data on which the predictors are trained on.
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''
    one = HybridUnivariatePredictor(training_df,2, len(input_batch), future_horizon)
    one.create_cnnlstm()
    one.fit_model(10)
    one.show_performance()
    
    two = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    two.create_bilstm()
    two.fit_model(10)
    two.show_performance()
     
    three = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    three.create_cnn()
    three.fit_model(10)
    three.show_performance()

    four = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    four.create_mlp()
    four.fit_model(10)
    four.show_performance()

    five = BasicUnivariatePredictor(training_df, len(input_batch), future_horizon)
    five.create_lstm()
    five.fit_model(10)
    five.show_performance()
    
    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)
    prediction_four = four.predict(input_batch)
    prediction_five = five.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three, prediction_four, prediction_five], axis=1) 

    return final_df

def system_disagreement(df: DataFrame):
    '''Plots the overall system disagreement and the individual disagreement scores of the algorithms.
        
        Parameters:
            df (DataFrame): Containing all individual predictors forecasts.
    '''
    disagreement(df).plot()
    predictor_score(df).plot()



def consensus(df: DataFrame, real: DataFrame) -> DataFrame:
    '''Applies the following consensus algorithm to provide the final system forecast:

    1. Average
    2. No Memory
    3. Memory
    4. Focus
    5. Anchor

        Parameters:
            df (DataFrame): Forecasts of all individual predictors.
            real (DataFrame): The true/actual values.

        Returns:
            (DataFrame): Containing all final consensus values from all algorithms.
    '''
    consensus = pd.DataFrame()
    
    average = average_consolidation(df)
    nomemory = consolidated_predictions(df, real)
    memory = consolidated_predictions_memory(df, real)
    focus = consolidated_predictions_focused(df, real)
    anchor = consolidated_predictions_anchor(df, real, 1.5)
    
    consensus['Average'] = average
    consensus['NoMemory'] = nomemory
    consensus['Memory'] = memory
    consensus['Focus'] = focus
    consensus['Anchor'] = anchor
    
    return consensus

def set_same_index(to_df: DataFrame, from_df: DataFrame) -> DataFrame:
    '''Helper function to transfer the dates of a date-time indexed dataframe to another.

        Parameter:
            to_df (DataFrame): Target DataFrame, application destination.
            from_df (DataFrame): DataFrame containing date-time index. 

        Returns:
            (DataFrame): Target DataFrame with date-time index.
    '''
    to_df = to_df.set_index(from_df.index)

    return to_df

def evaluation_frame(to_df: DataFrame, from_df: DataFrame) -> DataFrame:
    '''Helper function to transfer the dates of a date-time indexed dataframe to another while additionally attaching the Real Value column.
        
        Parameters:
            to_df (DataFrame): Target DataFrame, application destination.
            from_df (DataFrame): DataFrame containing date-time index and Real Value column.

        Returns:
            (DataFrame): Target DataFrame with date-time index and Real Valeu column.
    '''
    to_df = to_df.set_index(from_df.index)
    to_df['Real Value'] = from_df
    
    return to_df

def combined_frame(df1: DataFrame, df2:DataFrame, real: DataFrame) -> DataFrame:
    '''Combining the individual predictors forecast DataFrame with the algorithm consensus values DataFrame. These two DataFrames are divided by the Real Value column.

        Parameters:
            df1 (DataFrame): First DataFrame, for example: DataFrame containing algorithms consensus values.
            df2 (DataFrame): Second DataFrame, for example: DataFrame containing individual predictors forecasts.
            real (DataFrame): DataFrame containing the real values.
            
        Returns:
            (DataFrame): Combined DataFrame with Real Value column in the middle. 
    '''
    df1 = set_same_index(df1, real)
    df2 = evaluation_frame(df2, real)

    combined_frame = pd.concat([df2, df1], axis=1)
    
    return combined_frame



def mse_score(df: DataFrame) -> DataFrame:
    '''Calculates the mean squared error for the individual predictors and consensus algorithms. Plots MSE performences in descending order.

        Parameters:
            df (DataFrame): DataFrame containing individual predictors forecasts and consensus values of algorithms.

        Returns:
            (DataFrame): DataFrame containing mean squared error of individual predictors forecasts and consensus values of algorithms.
    '''
    end = (list(df.columns).index('Real Value'))
    start = (list(df.columns).index('Real Value')) + 1

    y_true = df['Real Value']

    name = []
    mse = []
    for i in range(0, end):
        name.append(df.columns[i])
        mse.append(mean_squared_error(y_true, df.iloc[:, i]))
    
    for i in range(start, df.shape[1]):
        name.append(df.columns[i])
        mse.append(mean_squared_error(y_true, df.iloc[:, i]))

    data = {'Algorithms': name, 'MSE': mse}
    result = pd.DataFrame(data)
    
    to_plot = result.sort_values(by = 'MSE')
    to_plot.plot.bar(x='Algorithms', y='MSE', figsize=(15, 6))

    return result

def mae_score(df: DataFrame) -> DataFrame:
    '''Calculates the mean absolute error for the individual predictors and consensus algorithms. Plots MAE performences in descending order.

        Parameters:
            df (DataFrame): DataFrame containing individual predictors forecasts and consensus values of algorithms.

        Returns:
            (DataFrame): DataFrame containing mean absolute error of individual predictors forecasts and consensus values of algorithms.
    '''
    end = (list(df.columns).index('Real Value'))
    start = (list(df.columns).index('Real Value')) + 1

    y_true = df['Real Value']

    name = []
    mae = []
    for i in range(0, end):
        name.append(df.columns[i])
        mae.append(mean_absolute_error(y_true, df.iloc[:,i]))
    
    for i in range(start, df.shape[1]):
        name.append(df.columns[i])
        mae.append(mean_absolute_error(y_true, df.iloc[:,i]))

    data = {'Algorithms': name, 'MAE': mae}
    result = pd.DataFrame(data)

    to_plot = result.sort_values(by = 'MAE')
    to_plot.plot.bar(x='Algorithms', y='MAE', figsize=(15, 6))

    return result

def mse_log_score(df: DataFrame) -> [(list, list)]:
    '''Calculates the mean squared log error for the individual predictors and consensus algorithms.

        Parameters:
            df (DataFrame): DataFrame containing individual predictors forecasts and consensus values of algorithms.

        Returns:
            [(list, list)]: Lists containing mean squared log error of individual predictors forecasts and consensus values of algorithms.
    '''
    end = (list(df.columns).index('Real Value'))
    start = (list(df.columns).index('Real Value')) + 1

    y_true = df['Real Value']

    mse_log1 = []
    mse_log2 = []
    for i in range(0, end):
        mse_log1.append((df.columns[i], mean_squared_log_error(y_true, df.iloc[:,i])))
    
    for i in range(start, df.shape[1]):
        mse_log2.append((df.columns[i], mean_squared_log_error(y_true, df.iloc[:,i])))

    return mse_log1, mse_log2



def plot_performance(data: DataFrame):
    '''Plots individual predictors forecasts and consensus values of algorithms against the real values.

        Parameters:
            data (DataFrame): DataFrame containing individual predictors forecasts, consensus values and real values.
    '''
    columns = data.columns

    for i in range(len(columns)):
        if columns[i] == 'Real Value':
            continue
        
        plt.figure(figsize=(15,6))
        plt.plot(data['Real Value'])
        plt.plot(data[columns[i]])
        plt.title(columns[i] + ' Algorithm Error')
        plt.ylabel('Absolute Error')
        plt.xlabel('Time')
        plt.legend(['Real Value', 'Prediction', 'Error'], loc='upper right')
        plt.show()

