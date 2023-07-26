from setup import *
from os import environ
# fixing hash randomisation seed
environ['PYTHONHASHSEED'] = str(SEED)
from tensorflow import config
from tensorflow import device
from pandas import DataFrame, Series
from pathlib import Path
from autoencoder import *
from extreme_autoencoder import ExtremeAutoEncoder
from auxiliary_functions import *
from data_preparation import prepare_data, partition_data, save_dataframe


if not GPU:
    # Hiding available GPUs (as TensorFlow has been reported to still utilise 
    # them, even when CPU processing has been specified)
    environ["CUDA_VISIBLE_DEVICES"] = ""

    if not CPU_multithreading:
        # disabling multi-threading, if requried
        config.threading.set_inter_op_parallelism_threads(1)
        # setting avariable for folder-name
        thread = 'single'
    else:
        thread = 'multi'
else:
    thread = 'multi'

def run_experiment(encoder_depth):
    """Run the reproducibility experiment.

    Returns a DataFrame of loss histories and saves loss histories, 
    validation loss histories, models and decoder outputs. 

    Parameters:
    ------------
    encodernumber (int): number of the encoder, integer in [0,2]

    Returns:
    -----------
    loss_histories (DataFrame): pandas DataFrame of loss histories

    """
    # loading the selected AutoEncoder class
    AutoEncoder, layersizes = get_autoencoder(encoder_depth)
    # setting the active device
    active_device = determine_device()
    # defining the foldername for the saved data
    foldername = get_foldername(encoder_depth, active_device, thread)
    # re-setting the random seeds
    reset_random_seeds()

    # creating empty DataFrames to be filled iteratively
    loss_histories = DataFrame()
    val_loss_hists = DataFrame()

    # data preparation

    training_set, testing_set = prepare_data(SEED, scaling, test_size)
    normal_train_data, normal_test_data, anomaly_train_data, anomaly_test_data = partition_data(
        SEED, test_size, training_set, testing_set)

    sample_number = normal_train_data.shape[1]

    with device(active_device):

        for repetition in list(range(repetitions+1)):
            # re-setting random seeds with every iteration
            reset_random_seeds()
            # initiating AutoEncoder object and compiling the model
            model= AutoEncoder(sample_number, layersizes, iteration_number= repetition)
            model.compile(optimizer=optimiser, loss=loss_function)

            # fitting the model
            history = model.fit(normal_train_data, normal_train_data, 
                                epochs=epochs, 
                                batch_size=batch_size,
                                validation_data=(training_set.drop('class', axis=1), 
                                                 training_set.drop('class', axis=1)),
                                shuffle=False)
            
            # extracting loss and validation loss histories
            loss_histories['{}'.format(repetition)] = Series(history.history['loss'])
            val_loss_hists['{}'.format(repetition)] = Series(history.history['val_loss'])

            # dropping duplicate performances
            loss_histories = loss_histories.T.drop_duplicates().T
            val_loss_hists = val_loss_hists.T.drop_duplicates().T
            # saving the training and validation metrics data
            save_dataframe(loss_histories, foldername, 'loss_histories.csv')
            save_dataframe(val_loss_hists, foldername, 'val_loss_hists.csv')
            # saving the model
            save_model(encoder_depth, model, epochs, batch_size, repetition, active_device, thread)

            # encoding and decoding the normal test data
            encoder_out = model.encoder(normal_test_data.to_numpy())
            decoder_out = model.decoder(encoder_out)
            decoded = DataFrame(decoder_out.numpy())
            # saving the decoded data
            save_dataframe(decoded, foldername, 'decoder_output_{}'.format(repetition))

        # copying the settings file into the results folder
        save_settings(foldername)

    return loss_histories


def save_model(encoder_depth, model, epochs, batch_size, repetition, active_device, thread):
    """Auxiliary function to save trained models.

    Parameters:
    -----------
        encodernumber (int): determines the encoder, integer in [0,3]
        model (Model): TensorFlow AutoEncoder model
        epochs (int): number of training epochs
        batch_size (int): size of training batches
        repetition (int): number of the experiment's execution
        active_device (str): str reflecting the active device, GPU or CPU
        thread (str): multi-threading setting. single or multi

    Returns:
    --------
        None

    """
    # defining file and flodername
    foldername = get_foldername(encoder_depth, active_device, thread)
    filename = 'model_{}_{}_{}'.format(epochs, batch_size, repetition)
    directory = Path(foldername)
    # Folder creation
    directory.mkdir(parents=True, exist_ok=False)
    # saving the model
    model.save(foldername + filename)


def get_foldername(encoder_depth, active_device, thread):
    """Auxiliary function for folder-name generation.

    Parameters:
    -------------- 
        encodernumber (int): determines the encoder, integer in [0,3]
        active_device (str): str reflecting the active device, either GPU or CPU
        thread (str): str reflective of multi-threading setting. either single or multi 

    Returns:
    --------
        foldername (str): automatially generated folder-name based on settings
    """
    if GPU:
        foldername = './results/{}_{}_{}_{}/'.format('e_' + str(encoder_depth),
                                                     active_device[1:4],
                                                     str(test_size).replace('.', ''),
                                                     SEED)
    else:
        foldername = './results/{}_{}_{}_{}_{}/'.format('e_' + str(encoder_depth),
                                                        active_device[1:4],
                                                        thread,
                                                        str(test_size).replace('.', ''),
                                                        SEED)

    return foldername


def get_autoencoder(encoder_depth):
    """Auxiliary function for the AutoEncoder and layersizes imports.

    Parameters:
    -----------
        encodernumber(int): determines the encoder, integer between [0,3]

    Returns:
    --------
        AutoEncoder (class[Model]): AutoEncoder model class
        layersizes (list[int]): list of specified layer-sizes for the AutoEncoder
    """
    if encoder_depth == 1:
        AutoEncoder = AutoEncoder9
        layersizes = layersizes_9
    elif encoder_depth == 2:
        AutoEncoder = AutoEncoder13
        layersizes = layersizes_13
    elif encoder_depth == 3:
        AutoEncoder = AutoEncoder23
        layersizes = layersizes_23
    elif encoder_depth == 0:
        AutoEncoder = ExtremeAutoEncoder
        layersizes = list(range(140, 3, -1))
    elif encoder_depth == 4:
        AutoEncoder = AutoEncoder11
        layersizes = layersizes_11

    return AutoEncoder, layersizes
