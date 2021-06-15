# Starting the program

Open up an anaconda terminal/command line and activate your enviorment. Now open a Jupyter notebook by executing: 
```shell
jupyter notebook
```
Finally, locate the downloaded repository and open up the System.ipynb contained in the notebook folder. 

Alternatively, to run the program via the main.py file (make sure to be in the directory where the main.py file is located):
```shell
python main.py
```
Depending on what individual predictor configuration is used, training of the model begins and after each completed model training cycle test and validation metrices are graphically shown. After all models in the configuration have been trained, the follwing GUI will appear:

![GUI menu main.py](resources/gui.png)

This GUI enables the user to explore the systems consensus predictions and performances in detail.

