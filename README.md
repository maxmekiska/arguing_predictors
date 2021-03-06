# Arguing Predictors

The aim of this proof-of-concept program is to let multiple individual predictors, forecasting chaotic time series data, communicate with each other and output a consensus prediction. This project will use financial data to test the system performance.

The program is build to be easily extended. New consensus algorithms or individual predictors can be added. 

For more detailed information visit the mkdocs build documentation of this program [here](https://maxmekiska.github.io/arguing_predictors/index.html): 


[![Website](documentation/docs/resources/WebsiteExample.png)](https://maxmekiska.github.io/arguing_predictors/index.html)

This documentation was build with the mkdocs python library which can be found [here](https://www.mkdocs.org/).


## System structure

```shell
+---arguing_predictors
    +---c 
    |   +---average.c
    |   +---c_wrapper.py
    |   +---correcting.c
    |   +---disagreement.c
    |   +---lib.so
    |   +---main.c
    |   +---prototype.h
    |   +---test_integrate_c.py
    +---consensus
    |   +---algorithms.py
    +---documentation
    |   +---docs
    |       +---...
    |   +---mkdocs.yml	
    +---experimental
    |   +---predictorsX.py
    +---notebooks
    |   +---System.ipynb
    |   +---TestingEnviornment.ipynb
    +---pretrained
    |	+---BI-LSTM_BP_30
    |	+---BI-LSTM_Ford_5
    |   +---BI-LSTM_SP500_40
    |	+---CNN-LSTM_BP_30
    |	+---CNN-LSTM_Ford_5
    |	+---CNN-LSTM_SP500_40
    |	+---CNN_BP_30
    |	+---CNN_Ford_5
    |	+---CNN_SP500_40
    |	+---LSTM_BP_30
    |	+---LSTM_Ford_5
    |	+---LSTM_SP500_40
    |	+---MLP_BP_30
    |	+---MLP_Ford_5
    |	+---MLP_SP500_40
    +---system
    |   +---activate.py
    +---test
    |   +---test_activate.py
    |   +---test_evaluation.py
    |   +---test_algorithms.py
    |   +---test_dataloader.py
    |   +---test_predictorsI.py
    |   +---test_predictorsII.py
    |   +---test_predictorsIII.py
    +---tools
    |   +---dataloader.py
    |   +---evaluation.py
    |   +---predictorsI.py
    |   +---predictorsII.py
    |   +---predictorsIII.py 
    +---main.py
```

## Installation

**IMPORTANT: The installation instruction are tested on a Windows 10 operating system. There might be alterations necessary to run the program on another operating system**

First download the program repository.

The proof of concept program uses a Jupyter notebook as UI. The program can also be used without a Jupyter notebook by executing the main.py file. In case of executing the main.py file, a small GUI will appear after the model set-up in the file has been successfully trained. The GUI is a window containing buttons to display different statistics about the systems overall performance. However, it is recommended to use the system with a Jupyter notebook and the main.py file to demo the system.

I recommend to use Anaconda to use Jupyter notebook and manage the necessary python libraries for this program. Anaconda can be downloaded [here](https://www.anaconda.com/products/individual#Downloads).

After Anaconda has been downloaded and successfully installed, open the anaconda terminal/command line and create a virtual environment with the following command:

```shell
conda create -n yourenvname python=3.8. 
```

After the environment is installed, activate it by typing: 

```shell
conda activate yourenvname
```

Please proceed by installing Jupyter notebook by executing:

```shell
conda install jupyter notebook
```

Now, all dependencies for the program need to be installed. First, pip install torch v.1.6 from [here](https://pytorch.org/get-started/previous-versions/). I recommend to use the CPU only version.

As indicated on the official PyTorch website, the CPU version for Windows (and Linux) can be installed by executing:

```shell
pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
```

Or alternatively for OSX:

```shell
pip install torch==1.6.0 torchvision==0.7.0
```

Finally, cd into the program directory and execute:

```shell
pip install -r requirements.txt
```

This will install all other necessary dependencies.

## Starting the program

Open up an anaconda terminal/command line and activate your environment. Now open a Jupyter notebook by executing: 
```shell
jupyter notebook
```
Finally, locate the downloaded repository and open up the System.ipynb contained in the notebook folder. The system can also be imported in any new jupyter notebook with the following line within the notebook (the path might differ depending on where the new jupyter notebook is located): 

```python3
%run ../system/activate.py
```

Alternatively, to run the program via the main.py file (make sure to be in the directory where the main.py file is located):
```shell
python main.py
```
Depending on what individual predictor configuration is used, training of the model begins and after each completed model training cycle test and validation metrics are graphically shown. Each appearing window containing plots/graphics need to be closed manually before the program can proceed running (close pop-up windows by clicking on the upper right red corner "x" button). After all models in the configuration have been trained, the following GUI will appear:

![GUI menu main.py](documentation/docs/resources/gui.png)

This GUI enables the user to explore the systems consensus predictions and performances in detail.

## Example main.py execution

![Gif example execution](documentation/docs/resources/ExampleMain.gif)

## Run example Jupyter Notebook

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/maxmekiska/arguing_predictors/master?labpath=notebooks%2FSystem.ipynb)

**IMPORTANT: May take a while until build and not all functionalities are available. For full functionality, please follow the installation guide.**
