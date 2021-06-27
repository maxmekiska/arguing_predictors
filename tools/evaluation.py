import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

import sys
sys.path.append('../')

from consensus.algorithms import disagreement, predictor_score  # since disagreement scores are used in this evaluation pipeline set-up

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error

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

def all_stats_frame(combined: DataFrame, predictor_forecasts: DataFrame) -> DataFrame:
    '''Combining the combined DataFrame with disagreement statistics to build a DataFrame containing all system statistics.

        Parameters:
            combined (DataFrame): First DataFrame, containing algorithms, consensus and real value.
            predictor_forecasts (DataFrame): DataFrame containing predictor forecasts.

        Returns:
            (DataFrame): DataFrame containing all system statistics.
    '''
    dis = disagreement(predictor_forecasts)
    score = predictor_score(predictor_forecasts)

    adjusted_dis = set_same_index(dis, combined)
    adjusted_score = set_same_index(score, combined)

    result = pd.concat([combined, adjusted_dis, adjusted_score], axis=1)

    return result

def correlation(df: DataFrame, plot: bool = False) -> DataFrame:
    '''Computation of correlation matrix with Pandas Library corr() function.
        Parameters:
            df (DataFrame): DataFrame to supply data for the correlation matrix.
            plot (bool): Option to plot correlation heatmap when True is passed in.
        Returns:
            (DataFrame): Correlation matrix of supplied DataFrame.
    '''
    corr_matrix = df.corr()

    if plot == True:
        plt.figure(figsize = (15,15))
        sns.heatmap(corr_matrix, annot=True)

    return corr_matrix

def absolute_error_analytics(predictors: DataFrame, algorithms: DataFrame, real: DataFrame) -> DataFrame:
    '''Computes the absolute error values of all individual predictors and consensus algorithms. Additionally adds system disagreement and individual predictors disagreement scores.
    
        Parameters:
            predictors (DataFrame): DataFrame containing individual predictors forecasts.
            algorithms (DataFrame): DataFrame containing consensus algorithm forecasts.
            real (DataFrame): DataFrame containing actual future values.
        
        Returns:
            (DataFrame): DataFrame containing all absolute error values of individual predictors and consenus algorithms togehter with system disagreement and individual disagreement scores.
    '''
    data = evaluation_frame(predictors,real)
    
    data2 = evaluation_frame(algorithms, real)
    
    individual_disagreements = predictor_score(predictors)
    individual_disagreements = set_same_index(individual_disagreements, real)
    
    system_disagreement = disagreement(predictors)
    system_disagreement = set_same_index(system_disagreement, real)

    result = pd.DataFrame() 
    for i in range(len(data.columns)-1): # do not include Real value column
        current_column = data.columns[i]
        result[current_column + ' absolute error'] = abs(data[current_column] - data['Real Value'])
        
    for i in range(len(data2.columns)-1):
        current_column = data2.columns[i]
        result[current_column + ' absolute error'] = abs(data2[current_column] - data2['Real Value'])
    
    result = pd.concat([result, individual_disagreements, system_disagreement], axis=1)
    
    return result

def mse_score(df: DataFrame, plot: bool = False) -> DataFrame:
    '''Calculates the mean squared error for the individual predictors and consensus algorithms. Option to plot MSE performences in descending order.

        Parameters:
            df (DataFrame): DataFrame containing individual predictors forecasts and consensus values of algorithms.
            plot (bool): Option to plot MSE performance chart.

        Returns:
            (DataFrame): DataFrame containing mean squared error of individual predictors forecasts and consensus values of algorithms.
    '''
    # finding start and end index since Real Value column is in the middle
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

    if plot == True: # plotting results if set to True 
        to_plot = result.sort_values(by = 'MSE')
        to_plot.plot.bar(x='Algorithms', y='MSE', figsize=(15, 6))

    return result

def mae_score(df: DataFrame, plot: bool = False) -> DataFrame:
    '''Calculates the mean absolute error for the individual predictors and consensus algorithms. Option to plot MAE performences in descending order.

        Parameters:
            df (DataFrame): DataFrame containing individual predictors forecasts and consensus values of algorithms.
            plot (bool): Option to plot MAE performance chart.

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
    
    if plot == True:
        to_plot = result.sort_values(by = 'MAE')
        to_plot.plot.bar(x='Algorithms', y='MAE', figsize=(15, 6))

    return result

def mse_log_score(df: DataFrame, plot: bool = False) -> DataFrame:
    '''Calculates the mean squared log error for the individual predictors and consensus algorithms. Option to plot MSE log performences in descending order.

        Parameters:
            df (DataFrame): DataFrame containing individual predictors forecasts and consensus values of algorithms.
            plot (bool): Option to plot MSE log performance chart.

        Returns:
            (DataFrame): DataFrame containing mean squared log error of individual predictors forecasts and consensus values of algorithms.
    '''
    end = (list(df.columns).index('Real Value'))
    start = (list(df.columns).index('Real Value')) + 1

    y_true = df['Real Value']

    name = []
    mse_log = []
    for i in range(0, end):
        name.append(df.columns[i])
        mse_log.append(mean_squared_log_error(y_true, df.iloc[:,i]))
    
    for i in range(start, df.shape[1]):
        name.append(df.columns[i])
        mse_log.append(mean_squared_log_error(y_true, df.iloc[:,i]))

    data = {'Algorithms': name, 'MSE Log': mse_log}
    result = pd.DataFrame(data)

    if plot == True:
        to_plot = result.sort_values(by = 'MSE Log')
        to_plot.plot.bar(x='Algorithms', y='MSE Log', figsize=(15, 6))

    return result

def plot_performance(data: DataFrame):
    '''Plots individual predictors forecasts and consensus values of algorithms against the real values.

        Parameters:
            data (DataFrame): DataFrame containing individual predictors forecasts, consensus values and real values.
    '''
    columns = data.columns

    for i in range(len(columns)):
        if columns[i] == 'Real Value':
            continue
        abs_error = abs(data[columns[i]] - data['Real Value']) # absolute error between real and predicted values (subplot)

        fig, ax = plt.subplots(2, 1, figsize=(15, 6))
        ax[0].plot(data['Real Value'])
        ax[0].plot(data[columns[i]])
        ax[0].title.set_text(columns[i] + ' Algorithm Error')
        ax[0].set_ylabel('Stock Price')
        ax[0].set_xlabel('Time')
        ax[0].legend(['Real Value', 'Prediction'], loc='upper right')

        ax[1].plot(abs_error, 'r')
        ax[1].title.set_text(columns[i] + ' Absolute Error')
        ax[1].set_ylabel('Error')
        ax[1].set_xlabel('Time')
        ax[1].legend(['Error'], loc='upper right')

        plt.tight_layout()

