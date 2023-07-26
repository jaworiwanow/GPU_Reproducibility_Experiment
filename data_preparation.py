from pandas import read_csv, concat, DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path


def prepare_data(seed, scaling=False, test_size=0.25):
    """Prepare the data for the model training.

    Performs a train-test split and performs scaling if required.
    Also saves the data as csv files.

    Parameters:
    -----------
        seed (int): random seed for the train test split
        scaling (bool): determines if scaling of the data is performed
        test_size(float): determines the percentage of data used for testing 

    Returns: 
    --------
        training_set (DataFrame): pandas DataFrame containing the training data 
        testing_set (DataFrame): pandas DataFrame containing the testing data

    """
    # Loading and preparing the data
    test_set = read_csv('./data/ECG5000_test.txt', sep='  ', header=None, engine='python')
    train_set = read_csv('./data/ECG5000_train.txt', sep='  ', header=None, engine='python')

    # Combining the dataset and renaming the target column
    # index needs to be ignored and reset for the merge to work
    dataset = concat([test_set, train_set], ignore_index=True)
    dataset.rename(columns={0: 'class'}, inplace=True)

    # Pre-processing
    # optional scaling and splitting anomalous from normal data
    target = dataset['class']
    values = dataset.drop('class', axis=1)

    if scaling:
        scaler = MinMaxScaler()

        scaler.fit(values)
        scaled_values = scaler.transform(values)
        scaled_values_df = DataFrame(scaled_values)

        train_data, test_data, train_classes, test_classes = train_test_split(scaled_values_df, target, test_size=test_size, random_state=seed, stratify=target)
    else:
        train_data, test_data, train_classes, test_classes = train_test_split(values, target, test_size=test_size, random_state=seed)

    # Data and labels are being "re-merged", so that 
    # only normal data is used in training
    training_set = concat([train_classes, train_data], axis=1)
    testing_set = concat([test_classes, test_data], axis=1)

    foldername = './results/{}_{}/'.format(str(test_size).replace('.', ''), seed)
    filename = 'training_set.csv'

    save_dataframe(training_set, foldername, filename)

    return training_set, testing_set


def partition_data(seed, test_size, training_set, testing_set):
    """Partition the training and test sets based on classification.

    Also saves the DataFrames relevant for encoder training as csv files.   

    Parameters:
    -----------
        seed (int): random seed for the train test split
        test_size(float): determines the percentage of data used for testing 
        training_set (DataFrame): pandas DataFrame containing the training data 
        testing_set (DataFrame): pandas DataFrame containing the testing data

    Returns:
    ---------
        normal_train_data (DataFrame): normal part of the training data
        normal_test_data (DataFrame): normal part of the test data
        anomaly_train_data (DataFrame): anomalous part of the train data
        anomaly_test_data (DataFrame): anomalous part of the test data

    """
    normal_train_data = training_set.loc[training_set['class'] == 1].drop('class', axis=1)
    anomaly_train_data = training_set.loc[training_set['class'] != 1].drop('class', axis=1)

    normal_test_data = testing_set.loc[testing_set['class'] == 1].drop('class', axis=1)
    anomaly_test_data = testing_set.loc[testing_set['class'] != 1].drop('class', axis=1)

    foldername = './results/{}_{}/'.format(str(test_size).replace('.', ''), seed)

    save_dataframe(normal_train_data, foldername, 'normal_train_data.csv')
    save_dataframe(anomaly_test_data, foldername, 'anomaly_test_data.csv')
    save_dataframe(normal_test_data, foldername, 'normal_test_data.csv')

    return normal_train_data, normal_test_data, anomaly_train_data, anomaly_test_data


def save_dataframe(dataframe, foldername, filename):
    """Auxiliary function saving dataframes to csv files.

    Parameters:
    -------------
        dataframe (DataFrame): pandas DataFrame containing the training data 
        foldername (str): name of the folder, in which the csv file is stored
        filename (str): name the csv file is assigned

    Returns:
    --------
        None
    """

    directory = Path(foldername)
    # Creation of the folder
    directory.mkdir(parents=True, exist_ok=True) 

    dataframe.to_csv(foldername + filename)