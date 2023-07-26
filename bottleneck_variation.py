from setup import *
from os import environ
# fixing hash randomisation seed
environ['PYTHONHASHSEED'] = str(SEED)
from pathlib import Path
from autoencoder import AutoEncoder3
from pandas import Series, DataFrame
from auxiliary_functions import reset_random_seeds, save_settings, determine_device
from data_preparation import prepare_data, partition_data, save_dataframe
from setup import *
from tensorflow import device

AutoEncoder = AutoEncoder3

training_set, testing_set = prepare_data(SEED, scaling, test_size)
normal_train_data, normal_test_data, anomaly_train_data, anomaly_test_data = partition_data(
        SEED, test_size, training_set, testing_set)

sample_number = normal_train_data.shape[1]


foldername = './results/bottleneck_variation_3_layers/'
directory = Path(foldername)
directory.mkdir(parents=True, exist_ok=True)
save_settings(foldername)

mean_scores = DataFrame()
score_stds = DataFrame()
distinct_counts = DataFrame()

active_device = determine_device()

with device(active_device):
    for bottleneck_size in list(range(1, 140)):
        print(bottleneck_size)
        reset_random_seeds()
        final_losses = []
        final_val_losses = []
        for repetition in list(range(repetitions+1)):
            reset_random_seeds()
            # initiating AutoEncoder object and compiling the model
            model = AutoEncoder(sample_number, bottleneck_size, iteration_number=repetition)
            model.compile(optimizer=optimiser, loss=loss_function)

            # fitting the model
            history = model.fit(normal_train_data, normal_train_data, 
                                epochs=epochs, 
                                batch_size=batch_size,
                                validation_data=(training_set.drop('class', axis=1), 
                                                    training_set.drop('class', axis=1)),
                                shuffle=False)

            # extracting final loss and validation loss histories
            final_losses.append(history.history['loss'][-1])
            final_val_losses.append(history.history['val_loss'][-1])
            print(bottleneck_size, repetition)

        final_loss_series = Series(final_losses)
        final_val_loss_series = Series(final_val_losses)
        mean_scores[str(bottleneck_size)] = Series([final_loss_series.mean(), final_val_loss_series.mean()])
        score_stds[str(bottleneck_size)] = Series([final_loss_series.std(), final_val_loss_series.std()])
        distinct_counts[str(bottleneck_size)] = Series([len(set(final_losses)), len(set(final_val_losses))])
    save_dataframe(mean_scores, foldername, 'mean_scores.csv')
    save_dataframe(score_stds, foldername, 'score_stds.csv')
    save_dataframe(distinct_counts, foldername, 'distinct_counts.csv')
