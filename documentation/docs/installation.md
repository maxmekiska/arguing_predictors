<style>
	.formatting {
		text-align: justify;
	 }
</style>

# Installation

**IMPORTANT: The installation instruction are tested on a Windows 10 operating system. There might be alterations necessary to run the program on another operating system**


<div class="formatting">

First download the program repository.

The proof of concept program uses a Jupyter notebook as UI. The program can also be used without a Jupyter notebook by executing the main.py file. In case of executing the main.py file, a small GUI will appear after the model set-up in the file has been successfully trained. The GUI is a window containing buttons to display different statistics about the systems overall performance. However, it is recommended to use the system with a Jupyter notebook and the main.py file to demo the system.

In addition, it is recommend to use Anaconda to run the Jupyter notebook and manage the necessary python libraries for this program. Anaconda can be downloaded <a href="https://www.anaconda.com/products/individual#Downloads">here</a>.

After Anaconda has been downloaded and successfully installed, open the anaconda terminal/command line and create a virtual environment with the following command:

```shell
conda create -n yourenvname python=3.8
```

After the environment is installed, activate it by typing: 

```shell
conda activate yourenvname
```

Please proceed by installing Jupyter notebook by executing:

```shell
conda install jupyter notebook
```

Now, all dependencies for the program need to be installed. First, pip install torch v.1.6 from <a href="https://pytorch.org/get-started/previous-versions/">here</a>. I recommend to use the CPU only version.

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

This will install all other necessary dependencies. For full reference, the full_env_requirements.txt contains all dependencies installed in the anaconda environment that was used to build this prototype system.
</div>
