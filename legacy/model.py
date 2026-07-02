import tensorflow as tf
from tensorflow.keras import layers, models, backend as K, regularizers

# Define Sampling layer
class Sampling(layers.Layer):
    """
    Custom Sampling Layer for Variational Autoencoder (VAE).
    
    This layer samples a latent vector z from a distribution defined by 
    (z_mean, z_log_var) using the reparameterization trick.
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

# Define the VAE class
class VAE(models.Model):
    """
    Variational Autoencoder (VAE) class.
    
    This class combines an encoder and a decoder to form a VAE model.
    It also computes the KL divergence loss during the forward pass.
    """

    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        """
        Forward pass of the VAE model.
        
        Parameters:
        - inputs: Input data to be passed to the encoder.
        
        Returns:
        - reconstructed: Reconstructed output from the decoder.
        """

        z_mean, z_log_var, z, skip1, skip2, skip3, skip4 = self.encoder(inputs)

        # Pass z along with the skip connection outputs to the decoder
        reconstructed = self.decoder([z, skip1, skip2, skip3, skip4])

        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        self.add_loss(K.mean(kl_loss) / (64 * 64 * 3))
        
        return reconstructed

    def get_config(self):
        return {"encoder": self.encoder, "decoder": self.decoder}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def create_encoder(latent_dim):
    """
    Creates the encoder model for the Variational Autoencoder.
    
    Parameters:
    - latent_dim: The dimensionality of the latent space (z).
    
    Returns:
    - encoder: The encoder model.
    """

    encoder_inputs = layers.Input(shape=(64, 64, 3))
    
    x1 = layers.Conv2D(32, 3, activation=None, strides=2, padding="same", kernel_regularizer=regularizers.l2(0.01))(encoder_inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)
    
    x2 = layers.Conv2D(64, 3, activation=None, strides=2, padding="same", kernel_regularizer=regularizers.l2(0.01))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)
    
    x3 = layers.Conv2D(128, 3, activation=None, strides=2, padding="same", kernel_regularizer=regularizers.l2(0.01))(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)
    
    x4 = layers.Conv2D(256, 3, strides=2, padding="same", kernel_regularizer=regularizers.l2(0.01))(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)
    x4 = layers.Dropout(0.5)(x4)
    
    flat = layers.Flatten()(x4)
    z_mean = layers.Dense(latent_dim, name="z_mean")(flat)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(flat)
    
    z = Sampling()([z_mean, z_log_var])
    
    encoder = models.Model(encoder_inputs, [z_mean, z_log_var, z, x1, x2, x3, x4], name="encoder")

    return encoder

def create_decoder(latent_dim, skip_shapes):
    """
    Creates the decoder model for the Variational Autoencoder.
    
    Parameters:
    - latent_dim: The dimensionality of the latent space (z).
    - skip_shapes: Shapes of the skip connections to be used in the decoder.
    
    Returns:
    - decoder: The decoder model.
    """

    latent_inputs = layers.Input(shape=(latent_dim,))
    
    skip_inputs = [layers.Input(shape=shape) for shape in skip_shapes]
    
    x = layers.Dense(8 * 8 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((8, 8, 256))(x)
    
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same")(x)
    upsampled_skip3 = layers.UpSampling2D(size=(4, 4))(skip_inputs[3])
    x = layers.Concatenate()([x, upsampled_skip3])
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)

    x = layers.Conv2DTranspose(128, 3, activation=None, strides=2, padding="same")(x)
    upsampled_skip2 = layers.UpSampling2D(size=(4, 4))(skip_inputs[2])
    x = layers.Concatenate()([x, upsampled_skip2])
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, activation=None, strides=2, padding="same")(x)
    upsampled_skip1 = layers.UpSampling2D(size=(4, 4))(skip_inputs[1])
    x = layers.Concatenate()([x, upsampled_skip1])
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 3, activation=None, strides=2, padding="same")(x)
    upsampled_skip0 = layers.UpSampling2D(size=(4, 4))(skip_inputs[0])
    x = layers.Concatenate()([x, upsampled_skip0])
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    decoder_outputs = layers.Conv2D(3, 3, strides=2, activation="sigmoid", padding="same")(x)
    decoder = models.Model([latent_inputs, *skip_inputs], decoder_outputs, name="decoder")
    
    return decoder
