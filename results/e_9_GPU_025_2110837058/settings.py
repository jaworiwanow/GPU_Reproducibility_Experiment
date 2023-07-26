### Experiment parameters
SEED = 2110837058
repetitions = 100           # number of repetitions of the experiment
GPU = True                 # determines, if GPU is used [boolean]
CPU_multithreading = True  # determines, if CPU multithreading is allowed
optimiser = 'adam'          # determines the optimiser, refer to tensorflow documentation for furhter options
loss_function = 'mse'       # determines the loss function, refer to tensorflow documentation for furhter options

### Data parameters
test_size = 0.25             # percentage of data used for testing
scaling = True              # determines, if data will be scaled (MinMax Scaling) [boolean]


### Model parameters
epochs = 5                      # number of epochs
batch_size = 125                # number of samples propagated, should be a factor of the number of training samples 
layersizes = [64, 32, 16, 8]    # hidden layer sizes for the autoencoder

