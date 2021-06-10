import PySimpleGUI as sg
import matplotlib.pyplot as plt

from tools.dataloader import *
from system.activate import *

def main():
    '''Example main function to execute the system without a jupyter notebook as UI.
    '''
    layout_training = [[sg.Button('Start'), sg.Button('Plot Correlation'), sg.Cancel('Continue')]]

    window_training = sg.Window('Train individual predictors', layout_training)


    while True:
        event, values = window_training.read()
        if event in (sg.WIN_CLOSED, 'Continue'):
            break
        elif event == 'Start':            
            predict = DataLoader('BP', '2018-02-01', '2018-05-01')
            predict = predict.get_adjclose()

            training = DataLoader('BP', '2015-01-01', '2018-01-01')
            training = training.get_adjclose()
           
            predict_req, real = data_prep(predict, 20, 30) # dividing data into predictor input and real data

            #individual_predictors_forecasts = individual_predictors_pretrained_BP_30_5(predict_req, 30)

            individual_predictors_forecasts = individual_predictors_template(training, predict_req, 30)

            consensus_forecasts = consensus(individual_predictors_forecasts, real)

            all_forecasts = combined_frame(individual_predictors_forecasts, consensus_forecasts, real)

            prediction_error = absolute_error_analytics(individual_predictors_forecasts, consensus_forecasts, real)


    window_training.close()

    layout_result = [[sg.Button('Plot Disagreement'), sg.Button('Plot Correlation'), sg.Cancel()], 
              [sg.Button('Plot MSE'), sg.Button('Plot MSE Log'), sg.Button('Plot MAE')],
              [sg.Button('Plot Performance')]]

    window_result = sg.Window('Arguing Predictors', layout_result)

    while True:
        event, values = window_result.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        elif event == 'Plot Disagreement':            
            system_disagreement(individual_predictors_forecasts)
            plt.show(block=False)
        elif event == 'Plot Correlation':
            correlation(prediction_error, True)
            plt.show(block=False)
        elif event == 'Plot MSE':
            mse_score(all_forecasts, True)
            plt.show(block=False)
        elif event == 'Plot MSE Log':
            mse_log_score(all_forecasts, True)
            plt.show(block=False)
        elif event == 'Plot MAE':
            mae_score(all_forecasts, True)
            plt.show(block=False)
        elif event == 'Plot Performance':
            plot_performance(all_forecasts)
            plt.show(block=False)

    window_result.close()

if __name__ == "__main__":
    main()
