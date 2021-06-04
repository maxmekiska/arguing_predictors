import pandas as pd
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt

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

def individual_predictors_template(training_df: DataFrame, input_batch: DataFrame, future_horizon: int) -> DataFrame:
    '''Handles the individual predictors by training them and feeding them the data to predict the specified future horizon. The following individual predictors are implemented:

    1. CNN-LSTM
    2. Bidirectional LSTM
    3. CNN

        Parameters:
            training_df (DataFrame): Data on which the predictors are trained on.
            input_batch (DataFrame): Data which is fed to predictors to predict future values.
            future_horizon (int): Length of how far into the future the predictors will predict.

        Returns:
            (DataFrame): Containing all predictions from all individual predictors.
    '''
    one = HybridUnivariatePredictor(2, len(input_batch), future_horizon, training_df)
    one.create_cnnlstm()
    one.fit_model(10)
    one.show_performance()
    
    two = BasicUnivariatePredictor(len(input_batch), future_horizon, training_df)
    two.create_bilstm()
    two.fit_model(10)
    two.show_performance()
     
    three = BasicUnivariatePredictor(len(input_batch), future_horizon, training_df)
    three.create_cnn()
    three.fit_model(10)
    three.show_performance()
    
    prediction_one = one.predict(input_batch)
    prediction_two = two.predict(input_batch)
    prediction_three = three.predict(input_batch)

    final_df = pd.concat([prediction_one, prediction_two, prediction_three], axis=1) 

    return final_df

def individual_predictors_template2(training_df: DataFrame, future_horizon: int) -> DataFrame:
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
    one.fit_neural_model(50, 'D')
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
    one.set_model_id('LSTM')
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
    5. LSTM-CNN

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
    5. LSTM-CNN

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
    5. LSTM-CNN

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
    anchor = consolidated_predictions_anchor(df, real, 3.5)
    
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
        plt.figure(figsize = (10,10))
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

    if plot == True: 
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
        
        plt.figure(figsize=(15,6))
        plt.plot(data['Real Value'])
        plt.plot(data[columns[i]])
        plt.title(columns[i] + ' Algorithm Error')
        plt.ylabel('Absolute Error')
        plt.xlabel('Time')
        plt.legend(['Real Value', 'Prediction', 'Error'], loc='upper right')
        plt.show()
