<style>
	.formatting {
		text-align: justify;
	 }
</style>

# Individual predictors
<div class="formatting">
There are three different categories of example individual predictors included in the proof-of-concept system. The first category consists of univariate multistep Kears based predictors which are implemented in the predictorsI.py and predictorsIII.py files. The second category includes Facebook's Prophet model and the Neural Prophet model which are contained within the predictorsII.py file. The third and last category translates the univariate multistep predictors in the predictorsI.py file into multivariate multistep Keras predictors. The last category is implemented in the predictorsX.py file contained within the experimental directory.

The example predictors contained within the first and third category can be used as basis for new architectures. For example, the following code is contained within the predictorsIII.py file and creates a CNN-LSTM structure.

```python3
def create_cnnlstm(self):
	'''Creates CNN-LSTM hybrid model by defining all layers with activation functions, optimizer, loss function and evaluation metrics. 
	'''
	self.set_model_id('CNN-LSTM')

	self.model = Sequential()
	self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=2, activation='relu'), input_shape=(None,self.modified_back, 1)))
	self.model.add(TimeDistributed(Conv1D(filters=32, kernel_size=2, activation='relu')))
	self.model.add(TimeDistributed(MaxPooling1D(pool_size=2)))
	self.model.add(TimeDistributed(Flatten()))
	self.model.add(LSTM(50, activation='relu', return_sequences=True))
	self.model.add(LSTM(25, activation='relu'))
	self.model.add(Dense(self.input_y.shape[1]))
	self.model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

```
</div>
