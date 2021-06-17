<style>
	div {
		text-align: justify;
	    }
</style>

# Pretrained models

In the following you can find the training processes of all pre-trained individual predictors.

## Horizon 5

The first dataset used is the stock price of Ford Motor Company (F). Prices are in USD and listed on NYSE - Nasdaq. The data is extracted via the Yahoo Finance API accessed via the pandas data reader function. The
adjusting closing price was used to train the following predictors. [Link](https://uk.finance.yahoo.com/quote/F/history?p=F) to the data.

The data the model was trained on is the historical adjusted close stock price of Ford Motor Company, ranging from 1st of January 2010 to the 1st of January 2018.

<embed src="/resources/ModelTrainingHorizon5.pdf" type="application/pdf" width="100%" height="620px">

## Horizon 30

The second dataset used is the stock price of BP p.l.c. (BP). Prices are in USD and listed on NYSE - Nasdaq. The data is extracted via the Yahoo Finance API accessed via the pandas data reader function. The
adjusting closing price was used to train the following predictors. [Link](https://uk.finance.yahoo.com/quote/BP/history?p=BP) to the data.


The data the model was trained on is the historical adjusted close stock price of Ford BP, ranging from 1st of January 2010 to the 1st of January 2018.

<embed src="/resources/ModelTraining2Horizon30.pdf" type="application/pdf" width="100%" height="620px">

## Horizon 40

The third dataset used is the S&P 500 (^GSPC). Prices are in USD and listed on SNP. The data is extracted via the Yahoo Finance API accessed via the pandas data reader function. The adjusting closing price was
used to train the following predictors. [Link](https://uk.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) to the data.

The data the model was trained on is the historical adjusted close stock price of the S&P500 index, ranging from 1st of January 2010 to the 1st of January 2018.

<embed src="/resources/ModelTraining3Horizon40.pdf" type="application/pdf" width="100%" height="620px">
