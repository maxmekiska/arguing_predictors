import PySimpleGUI as sg
#import matplotlib.pyplot as plt

from tools.dataloader import *
from system.activate import *

def main():
    '''Example main function to execute the system without a jupyter notebook as UI.
    '''


    predict = DataLoader('BP', '2018-02-01', '2018-05-01')
    predict = predict.get_adjclose()

    training = DataLoader('BP', '2015-01-01', '2018-01-01')
    training = training.get_adjclose()

    predict_req, real = data_prep(predict, 20, 30) # dividing data into predictor input and real data
    individual_predictors_forecasts = individual_predictors_template(training, predict_req, 30, 5)
    consensus_forecasts = consensus(individual_predictors_forecasts, real)
    all_forecasts = combined_frame(individual_predictors_forecasts, consensus_forecasts, real)
    prediction_error = absolute_error_analytics(individual_predictors_forecasts, consensus_forecasts, real) 

    sg.ChangeLookAndFeel('Dark')      
    sg.SetOptions(element_padding=(1, 1))

    layout = [[sg.Button('Plot Disagreement')], [sg.Button('Plot Correlation')], 
              [sg.Button('Plot MSE'), sg.Button('Plot MSE Log'), sg.Button('Plot MAE')],
              [sg.Button('Plot Performance'), sg.Cancel()]]

    window = sg.Window('Arguing Predictors', layout)

    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, 'Cancel'):
            break
        elif event == 'Plot Disagreement':            
            system_disagreement(individual_predictors_forecasts)
            plt.tight_layout()
            plt.show(block=False)
        elif event == 'Plot Correlation':
            correlation(prediction_error, True)
            plt.tight_layout()
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

    window.close()

if __name__ == "__main__":
    main()
