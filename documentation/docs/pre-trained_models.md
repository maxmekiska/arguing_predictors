<style>
	.formatting {
		text-align: justify;
	 }
</style>


# Pre-trained models
<div class="formatting">
In the following you can find the training processes of all pre-trained individual predictors.

</div>
## Horizon 5
<div class="formatting">
The first dataset used is the stock price of Ford Motor Company (F). Prices are in USD and listed on NYSE - Nasdaq. The data is extracted via the Yahoo Finance API accessed via the pandas data reader function. The
adjusting closing price was used to train the following predictors. <a href="https://uk.finance.yahoo.com/quote/F/history?p=F">Link</a> to the data.

The data the model was trained on is the historical adjusted close stock price of Ford Motor Company, ranging from 1st of January 2010 to the 1st of January 2018.

</div>
<embed src="/resources/ModelTrainingHorizon5.pdf" type="application/pdf" width="100%" height="620px">

The resulting pre-trained models are:

- BI-LSTM_Ford_5
- CNN-LSTM_Ford_5
- CNN_Ford_5
- LSTM_Ford_5
- MLP_Ford_5


## Horizon 30
<div class="formatting">
The second dataset used is the stock price of BP p.l.c. (BP). Prices are in USD and listed on NYSE - Nasdaq. The data is extracted via the Yahoo Finance API accessed via the pandas data reader function. The
adjusting closing price was used to train the following predictors. <a href="https://uk.finance.yahoo.com/quote/BP/history?p=BP">Link</a> to the data.


The data the model was trained on is the historical adjusted close stock price of Ford BP, ranging from 1st of January 2010 to the 1st of January 2018.

</div>
<embed src="/resources/ModelTraining2Horizon30.pdf" type="application/pdf" width="100%" height="620px">


The resulting pre-trained models are:

- BI-LSTM_BP_30
- CNN-LSTM_BP_30
- CNN_BP_30
- LSTM_BP_30
- MLP_BP_30

## Horizon 40
<div class="formatting">
The third dataset used is the S&P 500 (^GSPC). Prices are in USD and listed on SNP. The data is extracted via the Yahoo Finance API accessed via the pandas data reader function. The adjusting closing price was
used to train the following predictors. <a href="https://uk.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC">Link</a> to the data.

The data the model was trained on is the historical adjusted close stock price of the S&P500 index, ranging from 1st of January 2010 to the 1st of January 2018.

</div>
<embed src="/resources/ModelTraining3Horizon40.pdf" type="application/pdf" width="100%" height="620px">

The resulting pre-trained models are:

- BI-LSTM_SP500_40
- CNN-LSTM_SP500_40
- CNN_SP500_40
- LSTM_SP500_40
- MLP_SP500_40
