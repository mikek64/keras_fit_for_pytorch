# Keras fit for PyTorch #

Basic implementation of some of the keras functionality for PyTorch models.
Implements fit, evaluate and predict  plus save and load of the model and parameters
Accepts data inputs as numpy arrays or torch tensors

### Set up ###

Clone the repo and copy the file keraspytorch to your project
Requires PyTorch and Numpy

### Demo ###

Step through demo.py to see it working on the fashion mnist data
The demo uses Keras to obtain the data as numpy arrays and matplotlib to chart progress
The code was written with Spyder

### Usage ### 

(x(data), y(labels) can be all numpy arrays or all torch tensors, or, for x, lists thereof for multiple inputs.
The main class is called CompiledModel to align terminology with Keras though with PyTorch models are not compiled.

	from keraspytorch import CompiledModel
	compiled_model = CompiledModel(model,      # a PyTorch model
			optimizer,  # PyTorch optimizer instance
			lossfn,     # PyTorch loss function
			metrics,    # list of metrics. ['accuracy'] is the only metric currently coded 
						# but will accept a function metric(x,y) where x is the output of the model.
			predictfn)  # optional function applied to model output with predict, default None, 
						#E.g. nn.Softmax(dim = 1) to get probabilities if Softmax is not in the model

	training_history = compiled_model.fit(x_train, y_train, batch_size = 32, epochs = 10, validation_split = 0.2)
	test_loss, test_accuracy = compiled_model.evaluate(x_test, y_test, batch_size = 32)
	predictions = compiled_model.predict(x)

	compiled_model.save(file) # file path as string
	compiled_model.load(file)


### Potential issues ###

The current code loads the entire training set from numpy to PyTorch Tensor.  If that is very large it can cause memory issues on a GPU.
In such cases the train set needs splitting into large batches, or the code would need altering to enable loading mini batches to the GPU
