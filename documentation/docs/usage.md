<style>
	.formatting {
		text-align: justify;
	 }
</style>

# Starting the program
<div class="formatting">
## Jupyter Notebook
Open up an anaconda terminal/command line and activate your environment. Now open a Jupyter notebook by executing: 
```shell
jupyter notebook
```
Finally, locate the downloaded repository and open up the System.ipynb contained in the notebook folder. 

The following is an example of a successful System.ipynb run:


<embed src="/resources/ExampleSystem.pdf" type="application/pdf" width="100%" height="1200px">


## main.py script
Alternatively, to run the program via the main.py file (make sure to be in the directory where the main.py file is located):
```shell
python main.py
```
Depending on what individual predictor configuration is used, training of the model begins and after each completed model training cycle test and validation metrics are graphically shown. After all models in the configuration have been trained, the following GUI will appear:

![GUI menu main.py](resources/gui.png)

The following shows a full test run of the main.py:


![Test run main.py](resources/ExampleMain.gif)

This GUI enables the user to explore the systems consensus predictions and performances in detail.
</div>
