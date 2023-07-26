from setup import SEED, GPU
from os import environ
# fixing hash randomisation seed
environ['PYTHONHASHSEED']=str(SEED)
if not GPU:
    # Hiding available GPUs (as TensorFlow has been reported to still
    # utilise them, even when CPU processing has been specified)
    environ["CUDA_VISIBLE_DEVICES"] = ""
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


class AutoEncoder9(Model):
    """AutoEncoder Class for the 9 layer autoencoder setup."""

    def __init__(self, samplenumber, layersizes, iteration_number=0):
        """Construct the class.

        Attributes:
        -----------
        samplenumber (int): Number of samples within the time-series
        layersizes (list[int]): descending list of layer-sizes for each layer
        iteration_number (int): counter variable; number of created models

        Methods:
        --------
        init: class constructor
        """
        super().__init__()
    # the encoder and decoder are separate Sequential models
        self.encoder = Sequential([
          Dense(samplenumber, activation="relu"),
          Dense(layersizes[0], activation="relu"),
          Dense(layersizes[1], activation="relu"),
          Dense(layersizes[2], activation="relu"),
          Dense(layersizes[3], activation="relu"), ],
          name="encoder_{}".format(iteration_number))

        self.decoder = Sequential([
          Dense(layersizes[2], activation="relu"),
          Dense(layersizes[1], activation="relu"),
          Dense(layersizes[0], activation="relu"),
          Dense(samplenumber, activation="sigmoid")],
          name="decoder_{}".format(iteration_number))

    def call(self, data):
        """Apply the model. Encodes and decodes.

        Parameters:
        -----------
        data (numpy array): data to be encoded or decoded

        Returns:
        ---------
        decoded (numpy array): decoded data
        """
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded



class AutoEncoder13(Model):
    """AutoEncoder Class for the 13 layer autoencoder setup."""

    def __init__(self, samplenumber, layersizes, iteration_number=0):
        """Construct the class.

        Attributes:
        -----------
        samplenumber (int): Number of samples within the time-series
        layersizes (list[int]): descending list of layer-sizes for each layer
        iteration_number (int): counter variable; number of created models
        
        Methods:
        --------
        init: class constructor
        """
        super().__init__()
    
        # the encoder and decoder are separate Sequential models
        self.encoder = Sequential([
          Dense(samplenumber, activation="relu"),
          Dense(layersizes[0], activation="relu"),
          Dense(layersizes[1], activation="relu"),
          Dense(layersizes[2], activation="relu"),
          Dense(layersizes[3], activation="relu"),
          Dense(layersizes[4], activation="relu"),
          Dense(layersizes[5], activation="relu")],
          name="encoder_{}".format(iteration_number))
        
        self.decoder = Sequential([
          Dense(layersizes[4], activation="relu"),
          Dense(layersizes[3], activation="relu"),
          Dense(layersizes[2], activation="relu"),
          Dense(layersizes[1], activation="relu"),
          Dense(layersizes[0], activation="relu"),
          Dense(samplenumber, activation="sigmoid")],
          name="decoder_{}".format(iteration_number))

    def call(self, data):
        """Apply the model. Encodes and decodes.

        Parameters:
        -----------
        data (numpy array): data to be encoded or decoded

        Returns:
        ---------
        decoded (numpy array): decoded data
        """
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded



class AutoEncoder23(Model):
    """AutoEncoder Class for the 23 layer autoencoder setup."""

    def __init__(self, samplenumber, layersizes, iteration_number=0):
        """Construct the class.

        Attributes:
        -----------
        samplenumber (int): Number of samples within the time-series
        layersizes (list[int]): descending list of layer-sizes for each layer
        iteration_number (int): counter variable; number of created models

        Methods:
        --------
        init: class constructor
        """
        super().__init__()

        # the encoder and decoder are separate Sequential models
        self.encoder = Sequential([
          Dense(samplenumber, activation="relu"),
          Dense(layersizes[0], activation="relu"),
          Dense(layersizes[1], activation="relu"),
          Dense(layersizes[2], activation="relu"),
          Dense(layersizes[3], activation="relu"),
          Dense(layersizes[4], activation="relu"),
          Dense(layersizes[5], activation="relu"),
          Dense(layersizes[6], activation="relu"),
          Dense(layersizes[7], activation="relu"),
          Dense(layersizes[8], activation="relu"),
          Dense(layersizes[9], activation="relu"),
          Dense(layersizes[10], activation="relu")],
          name="encoder_{}".format(iteration_number))

        self.decoder = Sequential([
          Dense(layersizes[9], activation="relu"),
          Dense(layersizes[8], activation="relu"),
          Dense(layersizes[7], activation="relu"),
          Dense(layersizes[6], activation="relu"),
          Dense(layersizes[5], activation="relu"),
          Dense(layersizes[4], activation="relu"),
          Dense(layersizes[3], activation="relu"),
          Dense(layersizes[2], activation="relu"),
          Dense(layersizes[1], activation="relu"),
          Dense(layersizes[0], activation="relu"),
          Dense(samplenumber, activation="sigmoid")],
          name="decoder_{}".format(iteration_number))

    def call(self, data):
        """Apply the model. Encodes and decodes.

        Parameters:
        -----------
        data (numpy array): data to be encoded or decoded

        Returns:
        ---------
        decoded (numpy array): decoded data
        """
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder11(Model):
    """AutoEncoder Class for the 11 layer autoencoder setup."""

    def __init__(self, samplenumber, layersizes, iteration_number=0):
        """Construct the class.

        Attributes:
        -----------
        samplenumber (int): Number of samples within the time-series
        layersizes (list[int]): descending list of layer-sizes for each layer
        iteration_number (int): counter variable; number of created models

        Methods:
        --------
        init: class constructor
        """
        super().__init__()
    # the encoder and decoder are separate Sequential models
        self.encoder = Sequential([
          Dense(samplenumber, activation="relu"),
          Dense(layersizes[0], activation="relu"),
          Dense(layersizes[1], activation="relu"),
          Dense(layersizes[2], activation="relu"),
          Dense(layersizes[3], activation="relu"),
          Dense(layersizes[4], activation="relu")],
          name="encoder_{}".format(iteration_number))

        self.decoder = Sequential([
          Dense(layersizes[3], activation="relu"),  
          Dense(layersizes[2], activation="relu"),
          Dense(layersizes[1], activation="relu"),
          Dense(layersizes[0], activation="relu"),
          Dense(samplenumber, activation="sigmoid")],
          name="decoder_{}".format(iteration_number))

    def call(self, data):
        """Apply the model. Encodes and decodes.

        Parameters:
        -----------
        data (numpy array): data to be encoded or decoded

        Returns:
        ---------
        decoded (numpy array): decoded data
        """
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded


class AutoEncoder3(Model):
    """AutoEncoder Class for the 3 layer autoencoder setup."""

    def __init__(self, samplenumber, layersize, iteration_number=0):
        """Construct the class.

        Attributes:
        -----------
        samplenumber (int): Number of samples within the time-series
        layersize (int): size of the bottleneck layer
        iteration_number (int): counter variable; number of created models

        Methods:
        --------
        init: class constructor
        """
        super().__init__()
    # the encoder and decoder are separate Sequential models
        self.encoder = Sequential([
          Dense(samplenumber, activation="relu"),
          Dense(layersize, activation="relu")],
          name="encoder_{}".format(iteration_number))

        self.decoder = Sequential([
          Dense(samplenumber, activation="sigmoid")],
          name="decoder_{}".format(iteration_number))

    def call(self, data):
        """Apply the model. Encodes and decodes.

        Parameters:
        -----------
        data (numpy array): data to be encoded or decoded

        Returns:
        ---------
        decoded (numpy array): decoded data
        """
        encoded = self.encoder(data)
        decoded = self.decoder(encoded)
        return decoded
