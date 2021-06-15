<style>
	div {
		text-align: justify;
	    }
</style>

# Modifications

The system was designed to be adjusted and modified in multiple ways. First, the "tools" directory contains the individual predictors and supports example predictors based on the Keras python library. This overal framework can be used to add further individual predictor configurations but also offers the liberty to add predictors independent of this format. In the case of diverting from this structure, it is important that the predictor returns a DataFrame containing the future predictions. However, the pre-existing predictors can also be used to pre-train individual predictors.

Please find in the following examples of the pre-training process:

<embed src="/resources/ModelTraining2Horizon30.pdf" type="application/pdf" width="100%" height="500px">

The "tools" directory furthermore contains the dataloader which serves to import stock data. Other data can be added by creating a similar data import solution. Second, the "consensus" directory contains the algorithm that build the systems final consensus/prediction value. Further, consensus algorithm solutions can be added here as well. In general, the algorithm will need to take in a DataFrame containing the different predictors forecasts and another list or DataFrame containing actual real values. 

Third, the directory "system" contains the activate.py file which brings all of the individual parts together and enables the system to run. This file furthermore contains evaluation and plotting capabilities. Multiple adjustments can be made here to tailor the systems output.

## activate.py

```python3

```




