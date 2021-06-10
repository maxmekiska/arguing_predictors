# Arguing Predictors

The aim of this proof-of-concept program is to let multiple individual predictors, forecasting chaotic time series data, communicate with each other and output a consensus prediction. This project will use financial data to test the system performance.

The program is build to be easily extended. New consensus algorithms or individual predictors can be added. 

## Installation

First download the program repository.

The proof of concept program uses a Jupyter notebook as UI. The program can also be used without a Jupyter notebook by executing the main.py file. However, it is recommended to use it together with a Jupyter notebook.

I recommend to use Anaconda to use Jupyter notebook and manage the necessary python libraries for this program. Anaconda can be downloaded [here](https://www.anaconda.com/products/individual#Downloads).

After Anaconda has been downloaded and successfully installed, open the anaconda terminal/command line and create a virtual enviornment with the following command:

```shell
conda create -n yourenvname python=3.8. 
```

After the enviornment is installed, activate it by typing: 

```shell
conda activate yourenvname
```

Please proceed by installing Jupyter notebook by executing:

```shell
conda install jupyter notebook
```

Now, all dependiencies for the program need to be installed. First, pip install torch v.1.6 from [here](https://pytorch.org/get-started/previous-versions/). I recommend to use the CPU only version.

Finally, cd into the program directory and execute:

```shell
pip install -r requierements.txt
```

This will install all other necessary dependencies. For full reference, the full_env_requirements.txt contains all dependencies installed in the anaconda enviornment that was used to build this prototype system.

## Starting the program

Open up an anaconda terminal/command line and activate your enviorment. Now open a Jupyter notebook by executing: 
```shell
jupyter notebook
```
Finally, locate the downloaded repository and open up the System.ipynb contained in the notebook folder. 
