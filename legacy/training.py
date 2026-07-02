import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from model import VAE, create_encoder, create_decoder
import math

class CustomLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Custom Learning Rate Scheduler callback that reduces the learning rate when the monitored metric 
    (typically validation loss) does not improve for a specified number of epochs.
    
    Parameters:
    - factor: Factor by which to reduce the learning rate when a plateau is detected.
    - patience: Number of epochs to wait without improvement before reducing the learning rate.
    - min_lr: Minimum learning rate after reduction.
    - monitor: Metric to monitor for improvement.
    """

    def __init__(self, factor=0.1, patience=2, min_lr=1e-6, monitor='val_loss'):
        super(CustomLearningRateScheduler, self).__init__()
        self.factor = factor  # Factor by which the learning rate will be reduced. new_lr = lr * factor
        self.patience = patience  # Number of epochs with no improvement after which learning rate will be reduced.
        self.min_lr = min_lr  # Lower bound on the learning rate.
        self.monitor = monitor  # Metric to be monitored.
        self.wait = 0  # Number of epochs we have waited without improvement.
        self.best = float('inf')  # Initialize the best as infinity.
        self.lr = 0  # Current learning rate.

    def on_epoch_end(self, epoch, logs=None):
        """
        Called at the end of each epoch to check the monitored metric and adjust the learning rate if necessary.
        
        Parameters:
        - epoch: Current epoch number.
        - logs: Dictionary of the metrics from the current epoch.
        """

        current = logs.get(self.monitor)
        if current is None:
            raise ValueError(f"The specified metric '{self.monitor}' is not being recorded.")
        if current < self.best:
            self.best = current
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.wait = 0
                self.lr = float(tf.keras.backend.get_value(self.model.optimizer.lr))
                # Debugging: Hard code the factor value
                debug_factor = 0.1  # Ensure this is a float
                new_lr = max(self.lr * debug_factor, self.min_lr)
                if new_lr < self.lr:
                    tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                    print(f"\nEpoch {epoch+1}: Learning rate decreased to {new_lr:.4f} due to no improvement in {self.monitor}.")

def train_vae(Xtrain, Xval, latent_dim=64*64, batch_size=128, epochs=200):
    """
    Train the VAE model on the given dataset without data augmentation.
    
    Parameters:
    - Xtrain: Training data (NumPy array).
    - Xval: Validation data (NumPy array).
    - latent_dim: Dimensionality of the latent space.
    - batch_size: Number of samples per gradient update.
    - epochs: Number of training epochs.
    
    Returns:
    - encoder: The trained encoder model.
    - decoder: The trained decoder model.
    - vae: The complete VAE model.
    - history: Training history containing loss values and other metrics.
    """

    skip_shapes = [(32, 32, 32), (16, 16, 64), (8, 8, 128), (4, 4, 256)]
    
    encoder = create_encoder(latent_dim)
    decoder = create_decoder(latent_dim, skip_shapes)
    
    vae = VAE(encoder, decoder)
    
    initial_learning_rate = 0.0001  # Example custom learning rate
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    vae.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint('best_vae_model', monitor='val_loss', save_best_only=True, mode='min', save_format='tf')
    custom_lr_scheduler = CustomLearningRateScheduler(factor=0.1, patience=10, min_lr=1e-10)

    history = vae.fit(Xtrain, Xtrain, epochs=epochs, steps_per_epoch=math.ceil(len(Xtrain) / 32)
                      , batch_size=64, validation_data=(Xval, Xval), callbacks=[custom_lr_scheduler, early_stopping, model_checkpoint])
    
    return encoder, decoder, vae, history

def train_vae_augment(train_ds, val_ds, latent_dim=64*64, batch_size=128, epochs=200):
    """
    Train the VAE model on the given dataset with data augmentation.
    
    Parameters:
    - train_ds: Augmented training dataset (TensorFlow Dataset).
    - val_ds: Validation dataset (TensorFlow Dataset).
    - latent_dim: Dimensionality of the latent space.
    - batch_size: Number of samples per gradient update.
    - epochs: Number of training epochs.
    
    Returns:
    - vae: The complete VAE model.
    - history: Training history containing loss values and other metrics.
    """

    skip_shapes = [(32, 32, 32), (16, 16, 64), (8, 8, 128), (4, 4, 256)]
    
    encoder = create_encoder(latent_dim)
    decoder = create_decoder(latent_dim, skip_shapes)
    
    vae = VAE(encoder, decoder)
    
    initial_learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    vae.compile(optimizer=optimizer, loss=tf.keras.losses.MeanSquaredError())
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min')
    model_checkpoint = ModelCheckpoint('best_vae_model', monitor='val_loss', save_best_only=True, mode='min', save_format='tf')
    custom_lr_scheduler = CustomLearningRateScheduler(factor=0.1, patience=10, min_lr=1e-10)
    
    steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
    validation_steps = tf.data.experimental.cardinality(val_ds).numpy()
    
    history = vae.fit(train_ds, epochs=epochs, steps_per_epoch=steps_per_epoch,
                      validation_data=val_ds, validation_steps=validation_steps,
                      callbacks=[early_stopping, model_checkpoint, custom_lr_scheduler])
    
    return vae, history