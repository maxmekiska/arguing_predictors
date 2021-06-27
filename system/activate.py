import pandas as pd
from pandas import DataFrame

import sys
sys.path.append('../')

from consensus.algorithms import *
from tools.predictorsI import *
from tools.predictorsII import *
from tools.predictorsIII import *
from tools.evaluation import *

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

def individual_predictors_template0(training_df: DataFrame, input_batch: DataFrame, future_horizon: int, epochs: int) -> DataFrame:
    '''Handles the individual predictors by training them and feeding them the data to predict the specified future horizon. The following individual predictors are implemented:

    1. CNN-LSTM
    2. Bidirectional LSTM
    3. CNN

        Parameters:
            training_df (DataFrame): Data on which the predictors are trained on.
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.
            epochs (int): Determines for how many epochs each model is trained. 

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''
    one = HybridUnivariatePredictor(2, len(input_batch), future_horizon, training_df)
    one.create_cnnlstm()
    one.fit_model(epochs)
    one.show_performance()
    
    two = BasicUnivariatePredictor(len(input_batch), future_horizon, training_df)
    two.create_bilstm()
    two.fit_model(epochs)
    two.show_performance()
     
    three = BasicUnivariatePredictor(len(input_batch), future_horizon, training_df)
    three.create_cnn()
    three.fit_model(epochs)
    three.show_performance()
    
    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three], axis=1) 

    return final_df

def individual_predictors_template1(training_df: DataFrame, input_batch: DataFrame, future_horizon: int, epochs: int) -> DataFrame:
    '''Handles the individual predictors by training them and feeding them the data to predict the specified future horizon. The following individual predictors are implemented:

    1. CNN-LSTM
    2. Bidirectional LSTM
    3. CNN
    4. MLP
    5. LSTM

        Parameters:
            training_df (DataFrame): Data on which the predictors are trained on.
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.
            epochs (int): Determines for how many epochs each model is trained. 

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''
    one = HybridUnivariatePredictor(2, len(input_batch), future_horizon, training_df)
    one.create_cnnlstm()
    one.fit_model(epochs)
    one.show_performance()
    
    two = BasicUnivariatePredictor(len(input_batch), future_horizon, training_df)
    two.create_bilstm()
    two.fit_model(epochs)
    two.show_performance()
     
    three = BasicUnivariatePredictor(len(input_batch), future_horizon, training_df)
    three.create_cnn()
    three.fit_model(epochs)
    three.show_performance()
    
    four = BasicUnivariatePredictor(len(input_batch), future_horizon, training_df)
    four.create_mlp()
    four.fit_model(epochs)
    four.show_performance()

    five = BasicUnivariatePredictor(len(input_batch), future_horizon, training_df)
    five.create_lstm()
    five.fit_model(epochs)
    five.show_performance()

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)
    prediction_four = four.predict(input_batch)
    prediction_five = five.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three, prediction_four, prediction_five], axis=1) 

    return final_df

def individual_predictors_template3(training_df: DataFrame, future_horizon: int) -> DataFrame:
    '''Handles the individual predictors by training them and feeding them the data to predict the specified future horizon. The following individual predictors are implemented:

    1. Facebook Prophet
    2. Facebook Neural Prophet

        Parameters:
            training_df (DataFrame): Data on which the predictors are trained on.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = UnivariatePredictorII(training_df, future_horizon)
    one.fit_neural_model(150, 'D')
    one.show_performance_neural()

    two = UnivariatePredictorII(training_df, future_horizon)
    two.fit_prophet_model()
    two.show_performance_prophet()

    prediction_one = one.predict_neural()
    prediction_two = two.predict_prophet()

    final_df = pd.concat([prediction_one, prediction_two], axis=1) 

    return final_df

def individual_predictors_pretrained_Ford_5_2(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained Ford stock model (horizon=5) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM') # manual model ID/name setting, necessary since model is pretrained and loaded from a file
    one.load_model('../pretrained/LSTM_Ford_5')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_Ford_5')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two], axis=1) 

    return final_df

def individual_predictors_pretrained_BP_30_2(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained BP stock model (horizon=30) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_BP_30')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_BP_30')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two], axis=1) 

    return final_df

def individual_predictors_pretrained_SP500_40_2(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained SP500 index model (horizon=40) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_SP500_40')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_SP500_40')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two], axis=1) 

    return final_df

def individual_predictors_pretrained_Ford_5_3(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained Ford stock model (horizon=5) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_Ford_5')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_Ford_5')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_Ford_5')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three], axis=1) 

    return final_df

def individual_predictors_pretrained_BP_30_3(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained BP stock model (horizon=30) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_BP_30')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_BP_30')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_BP_30')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three], axis=1) 

    return final_df

def individual_predictors_pretrained_SP500_40_3(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained SP500 index model (horizon=40) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_SP500_40')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_SP500_40')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_SP500_40')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three], axis=1) 

    return final_df

def individual_predictors_pretrained_Ford_5_4(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained Ford stock model (horizon=5) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP
    4. Bidirectional-LSTM

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_Ford_5')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_Ford_5')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_Ford_5')

    four = BasicUnivariatePredictor(len(input_batch), future_horizon)
    four.set_model_id('BI-LSTM')
    four.load_model('../pretrained/BI-LSTM_Ford_5')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)
    prediction_four = four.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three, prediction_four], axis=1) 

    return final_df

def individual_predictors_pretrained_BP_30_4(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained BP stock model (horizon=30) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP
    4. Bidirectional-LSTM

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_BP_30')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_BP_30')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_BP_30')

    four = BasicUnivariatePredictor(len(input_batch), future_horizon)
    four.set_model_id('BI-LSTM')
    four.load_model('../pretrained/BI-LSTM_BP_30')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)
    prediction_four = four.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three, prediction_four], axis=1) 

    return final_df

def individual_predictors_pretrained_SP500_40_4(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained SP500 index model (horizon=40) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP
    4. Bidirectional-LSTM

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_SP500_40')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_SP500_40')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_SP500_40')

    four = BasicUnivariatePredictor(len(input_batch), future_horizon)
    four.set_model_id('BI-LSTM')
    four.load_model('../pretrained/BI-LSTM_SP500_40')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)
    prediction_four = four.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three, prediction_four], axis=1) 

    return final_df

def individual_predictors_pretrained_Ford_5_5(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained Ford stock model (horizon=5) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP
    4. Bidirectional-LSTM
    5. CNN-LSTM

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_Ford_5')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_Ford_5')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_Ford_5')

    four = BasicUnivariatePredictor(len(input_batch), future_horizon)
    four.set_model_id('BI-LSTM')
    four.load_model('../pretrained/BI-LSTM_Ford_5')

    five = HybridUnivariatePredictor(sub_seq = 2, steps_past = len(input_batch), steps_future = future_horizon)
    five.set_model_id('CNN-LSTM')
    five.load_model('../pretrained/CNN-LSTM_Ford_5')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)
    prediction_four = four.predict(input_batch)
    prediction_five = five.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three, prediction_four, prediction_five], axis=1) 

    return final_df

def individual_predictors_pretrained_BP_30_5(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained BP stock model (horizon=30) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP
    4. Bidirectional-LSTM
    5. CNN-LSTM

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_BP_30')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_BP_30')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_BP_30')

    four = BasicUnivariatePredictor(len(input_batch), future_horizon)
    four.set_model_id('BI-LSTM')
    four.load_model('../pretrained/BI-LSTM_BP_30')

    five = HybridUnivariatePredictor(sub_seq = 2, steps_past = len(input_batch), steps_future = future_horizon)
    five.set_model_id('CNN-LSTM')
    five.load_model('../pretrained/CNN-LSTM_BP_30')

    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)
    prediction_four = four.predict(input_batch)
    prediction_five = five.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three, prediction_four, prediction_five], axis=1) 

    return final_df

def individual_predictors_pretrained_SP500_40_5(input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Loades pretrained SP500 index model (horizon=40) and predicts based on the given input batch. The following individual predictors are implemented:

    1. LSTM
    2. CNN
    3. MLP
    4. Bidirectional-LSTM
    5. CNN-LSTM

        Parameters:
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''    
    one = BasicUnivariatePredictor(len(input_batch), future_horizon)
    one.set_model_id('LSTM')
    one.load_model('../pretrained/LSTM_SP500_40')
 
    two = BasicUnivariatePredictor(len(input_batch), future_horizon)
    two.set_model_id('CNN')
    two.load_model('../pretrained/CNN_SP500_40')

    three = BasicUnivariatePredictor(len(input_batch), future_horizon)
    three.set_model_id('MLP')
    three.load_model('../pretrained/MLP_SP500_40')

    four = BasicUnivariatePredictor(len(input_batch), future_horizon)
    four.set_model_id('BI-LSTM')
    four.load_model('../pretrained/BI-LSTM_SP500_40')

    five = HybridUnivariatePredictor(sub_seq = 2, steps_past = len(input_batch), steps_future = future_horizon)
    five.set_model_id('CNN-LSTM')
    five.load_model('../pretrained/CNN-LSTM_SP500_40')

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
    6. Correcting
    7. Correcting Memory

        Parameters:
            df (DataFrame): Forecasts of all individual predictors.
            real (DataFrame): The true/actual values.

        Returns:
            (DataFrame): Containing all final consensus values from all algorithms.
    '''
    consensus = pd.DataFrame() # DataFrame that will hold all consensus values
    
    average = average_consolidation(df)
    nomemory = consolidated_predictions(df, real)
    memory = consolidated_predictions_memory(df, real)
    focus = consolidated_predictions_focused(df, real)
    anchor = consolidated_predictions_anchor(df, real, 3.5)
    correcting = consolidated_predictions_correcting(df, real)
    correcting_mem = consolidated_predictions_memory_correcting(df, real)
    
    consensus['Average'] = average
    consensus['NoMemory'] = nomemory
    consensus['Memory'] = memory
    consensus['Focus'] = focus
    consensus['Anchor'] = anchor
    consensus['Correcting'] = correcting
    consensus['Correcting Memory'] = correcting_mem
    
    return consensus

def consensus_optimal(df: DataFrame, real: DataFrame) -> DataFrame:
    '''Applies the correcting consensus algorithm to provide the final system forecast.

        Parameters:
            df (DataFrame): Forecasts of all individual predictors.
            real (DataFrame): The true/actual values.

        Returns:
            (DataFrame): Containing all final consensus values from all algorithms.
    '''
    consensus = pd.DataFrame() # DataFrame that will hold consensus values

    correcting = consolidated_predictions_correcting(df, real)    
    consensus['Correcting'] = correcting
 
    return consensus
