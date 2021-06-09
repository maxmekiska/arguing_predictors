from tools.dataloader import *
from system.activate import *

def main():
    '''Example main function to execute the system without a jupyter notebook as UI.
    '''

    # data extraction/preparation phase

    predict = DataLoader('BP', '2018-02-01', '2018-05-01')
    predict = predict.get_adjclose()
   
    predict_req, real = data_prep(predict, 20, 30) # dividing data into predictor input and real data

    individual_predictors_forecasts = individual_predictors_pretrained_BP_30_5(predict_req, 30)

    system_disagreement(individual_predictors_forecasts)

    consensus_forecasts = consensus(individual_predictors_forecasts, real)

    all_forecasts = combined_frame(individual_predictors_forecasts, consensus_forecasts, real)

    prediction_error = absolute_error_analytics(individual_predictors_forecasts, consensus_forecasts, real)

    correlation(prediction_error, True)

    mse_score(all_forecasts, True)

    mse_log_score(all_forecasts, True)

    mae = mae_score(all_forecasts, True)

    plot_performance(all_forecasts)

if __name__ == "__main__":
    main()
