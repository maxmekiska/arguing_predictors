import pandas as pd

import sys
sys.path.append('../')

from consensus.algorithms import *
from tools.predictorsI import *
from tools.predictorsII import *
from tools.predictorsIII import *


def data_prep(df, input_batch_size, future_horizon):
    input_b = df[0:input_batch_size]
    real_value = df[input_batch_size:input_batch_size + future_horizon]
    
    return input_b, real_value


# future_horizon = length -> days into the future predicted
def individual_predictors(training_df, input_batch, future_horizon):
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


def evaluation_frame(algorithms, real_df):
    algorithms = algorithms.set_index(real_df.index)
    algorithms['Real Value'] = real_df
    
    return algorithms


def calculate_error_predictors(data):
    error = data.copy()
    
    error['Error CNN-LSTM'] = abs(error['Real Value'] - error['CNN-LSTM'])
    error['Error Bidirectional LSTM'] = abs(error['Real Value'] - error['Bidirectional LSTM'])
    error['Error CNN'] = abs(error['Real Value'] - error['CNN'])
    
    return error

def calculate_error_algorithms(data):
    error = data.copy()
    
    error['Error Average'] = abs(error['Real Value'] - error['Average'])
    error['Error NoMemory'] = abs(error['Real Value'] - error['NoMemory'])
    error['Error Memory'] = abs(error['Real Value'] - error['Memory'])
    error['Error Focus'] = abs(error['Real Value'] - error['Focus'])
    error['Error Anchor'] = abs(error['Real Value'] - error['Anchor'])
    
    return error

def print_simple_statistics(df):
    start = (list(df.columns).index('Real Value')) + 1

    print('-------SUM-------')
    for i in range(start,df.shape[1]):
        print(df.iloc[:,i].sum())
    print('------AVERAGE----')
    for j in range(start, df.shape[1]):
        print(df.iloc[:,j].mean())
    print('------MEDIAN-----')
    for k in range(start, df.shape[1]):
        print(df.iloc[:,k].median())



def plot_performance(data):

    plt.plot(data['Real Value'])
    plt.plot(data['Average'])
    plt.title('Average Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()
    
    plt.plot(data['Real Value'])
    plt.plot(data['NoMemory'])
    plt.title('NoMemory Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()
    
    plt.plot(data['Real Value'])
    plt.plot(data['Memory'])
    plt.title('Memory Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()
    
    plt.plot(data['Real Value'])
    plt.plot(data['Focus'])
    plt.title('Focus Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()
    
    plt.plot(data['Real Value'])
    plt.plot(data['Anchor'])
    plt.title('Anchor Algorithm Error')
    plt.ylabel('Absolute Error')
    plt.xlabel('Time')
    plt.legend(['Real Value', 'Consensus'], loc='upper right')
    plt.show()
