import pandas as pd
from pandas import DataFrame

from sklearn.metrics import mean_squared_error


import sys
sys.path.append('../')

from consensus.algorithms import *
from tools.predictorsI import *
from tools.predictorsII import *
from tools.predictorsIII import *


def data_prep(df: DataFrame, input_batch_size: int, future_horizon: int):
    input_b = df[0:input_batch_size]
    real_value = df[input_batch_size:input_batch_size + future_horizon]
    
    return input_b, real_value


def individual_predictors(training_df, input_batch, future_horizon: int):
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


def system_disagreement(df):
    disagreement(df).plot()
    predictor_score(df).plot()


def consensus(df, real):
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

def set_same_index(algorithms, real_df):
    algorithms = algorithms.set_index(real_df.index)

    return algorithms

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
