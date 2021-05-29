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


def individual_predictors(training_df: DataFrame, input_batch: DataFrame, future_horizon: int) -> DataFrame:
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
            to_df (DataFrame): 
        

    '''
    to_df = to_df.set_index(from_df.index)

    return to_df

def evaluation_frame(algorithms, real_df):
    algorithms = algorithms.set_index(real_df.index)
    algorithms['Real Value'] = real_df
    
    return algorithms

def combined_frame(df1, df2, real):
    df1 = set_same_index(df1, real)
    df2 = evaluation_frame(df2, real)

    combined_frame = pd.concat([df2, df1], axis=1)
    
    return combined_frame



def mse_score(df):
    end = (list(df.columns).index('Real Value'))
    start = (list(df.columns).index('Real Value')) + 1

    y_true = df['Real Value']

    mse1 = []
    mse2 = []
    for i in range(0, end):
        mse1.append(mean_squared_error(y_true, df.iloc[:,i]))
    
    for i in range(start, df.shape[1]):
        mse2.append(mean_squared_error(y_true, df.iloc[:,i]))

    return mse1, mse2

def mae_score(df):
    end = (list(df.columns).index('Real Value'))
    start = (list(df.columns).index('Real Value')) + 1

    y_true = df['Real Value']

    mae1 = []
    mae2 = []
    for i in range(0, end):
        mae1.append(mean_absolute_error(y_true, df.iloc[:,i]))
    
    for i in range(start, df.shape[1]):
        mae2.append(mean_absolute_error(y_true, df.iloc[:,i]))

    return mae1, mae2

def mse_log_score(df):
    end = (list(df.columns).index('Real Value'))
    start = (list(df.columns).index('Real Value')) + 1

    y_true = df['Real Value']

    mse_log1 = []
    mse_log2 = []
    for i in range(0, end):
        mse_log1.append(mean_squared_log_error(y_true, df.iloc[:,i]))
    
    for i in range(start, df.shape[1]):
        mse_log2.append(mean_squared_log_error(y_true, df.iloc[:,i]))

    return mse_log1, mse_log2



def plot_performance(data):

    plt.figure(figsize=(15,6))
    plt.plot(data['Real Value'])
    plt.plot(data['Average'])
    plt.title('Average Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()
   
    plt.figure(figsize=(15,6))
    plt.plot(data['Real Value'])
    plt.plot(data['NoMemory'])
    plt.title('NoMemory Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()
  
    plt.figure(figsize=(15,6))
    plt.plot(data['Real Value'])
    plt.plot(data['Memory'])
    plt.title('Memory Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()
 
    plt.figure(figsize=(15,6))
    plt.plot(data['Real Value'])
    plt.plot(data['Focus'])
    plt.title('Focus Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()

    plt.figure(figsize=(15,6))
    plt.plot(data['Real Value'])
    plt.plot(data['Anchor'])
    plt.title('Anchor Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()

    plt.figure(figsize=(15,6))
    plt.plot(data['Real Value'])
    plt.plot(data['CNN-LSTM'])
    plt.title('CNN-LSTM Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Individual Predictor estimate'], loc='upper right')
    plt.show()

    plt.figure(figsize=(15,6))
    plt.plot(data['Real Value'])
    plt.plot(data['Bidirectional LSTM'])
    plt.title('Bidirectional LSTM Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Individual Predictor estimate'], loc='upper right')
    plt.show()

    plt.figure(figsize=(15,6))
    plt.plot(data['Real Value'])
    plt.plot(data['CNN'])
    plt.title('CNN Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Individual Predictor estimate'], loc='upper right')
    plt.show()
   

