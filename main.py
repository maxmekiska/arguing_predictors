from tools.dataloader import *
from system.activate import *

def main(): # experimental main TODO: build loop to verify statistically prediction performance of the system

    # data extraction/preparation phase
    training = DataLoader('aapl', '2009-01-01', '2010-05-01')
    training = training.get_close()
    predict = DataLoader('aapl', '2010-06-01', '2010-09-01')
    predict = predict.get_close()
    predict_req, real = data_prep(predict, 24, 30)
    
    # individual predictors forecasting
    final_df = individual_predictors1(training, predict_req, 30)

    # system disagreement calculation
    system_disagreement(final_df)

    # building of consensus values
    algos = consensus(final_df, real)
    ui = combined_frame(final_df, algos, real)
    plot_performance(ui)

if __name__ == "__main__":
    main()

