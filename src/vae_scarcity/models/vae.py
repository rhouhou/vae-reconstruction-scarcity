"""VAE models for image reconstruction."""

from __future__ import annotations

from typing import Any

try:
    import tensorflow as tf
except ImportError:
    tf = None


def _require_tensorflow() -> None:
    """Raise a clear error if TensorFlow is not installed."""
    if tf is None:
        raise ImportError(
            "TensorFlow is required for VAE models. "
            'Install it with: python -m pip install -e ".[deep-learning]"'
        )

if tf is not None:
    BaseLayer = tf.keras.layers.Layer
    BaseModel = tf.keras.Model
else:
    BaseLayer = object
    BaseModel = object

class Sampling(BaseLayer):
    """Sampling layer using the VAE reparameterization trick."""

    def call(self, inputs: tuple[Any, Any]) -> Any:
        """Sample latent vector z from z_mean and z_log_var."""
        _require_tensorflow()

        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))

        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class SkipVAE(BaseModel):
    """Skip-connected convolutional VAE for 64x64 image reconstruction."""

    def __init__(
        self,
        encoder: Any,
        decoder: Any,
        image_shape: tuple[int, int, int] = (64, 64, 3),
        kl_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the VAE."""
        _require_tensorflow()
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.image_shape = image_shape
        self.kl_weight = kl_weight

    def call(self, inputs: Any) -> Any:
        """Run forward pass and add KL loss."""
        z_mean, z_log_var, z, skips = self.encoder(inputs)
        reconstructed = self.decoder([z, *skips])

        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1,
        )

        normalization = float(
            self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        )

        self.add_loss(self.kl_weight * tf.reduce_mean(kl_loss) / normalization)

        return reconstructed


def build_skip_encoder(
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
) -> Any:
    """Build a skip-connected convolutional VAE encoder."""
    _require_tensorflow()

    layers = tf.keras.layers
    models = tf.keras.models

    encoder_inputs = layers.Input(shape=image_shape)

    x1 = layers.Conv2D(32, 3, strides=2, padding="same")(encoder_inputs)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.ReLU()(x1)

    x2 = layers.Conv2D(64, 3, strides=2, padding="same")(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.ReLU()(x2)

    x3 = layers.Conv2D(128, 3, strides=2, padding="same")(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.ReLU()(x3)

    x4 = layers.Conv2D(256, 3, strides=2, padding="same")(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.ReLU()(x4)

    flat = layers.Flatten()(x4)

    z_mean = layers.Dense(latent_dim, name="z_mean")(flat)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(flat)
    z = Sampling()((z_mean, z_log_var))

    return models.Model(
        encoder_inputs,
        [z_mean, z_log_var, z, [x1, x2, x3, x4]],
        name="skip_encoder",
    )


def build_skip_decoder(
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
) -> Any:
    """Build a skip-connected convolutional VAE decoder."""
    _require_tensorflow()

    layers = tf.keras.layers
    models = tf.keras.models

    channels = image_shape[-1]

    latent_inputs = layers.Input(shape=(latent_dim,))
    skip1 = layers.Input(shape=(32, 32, 32))
    skip2 = layers.Input(shape=(16, 16, 64))
    skip3 = layers.Input(shape=(8, 8, 128))
    skip4 = layers.Input(shape=(4, 4, 256))

    x = layers.Dense(4 * 4 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((4, 4, 256))(x)

    x = layers.Concatenate()([x, skip4])

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip3])

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip2])

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Concatenate()([x, skip1])

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    decoder_outputs = layers.Conv2D(
        channels,
        3,
        activation="sigmoid",
        padding="same",
        name="reconstruction",
    )(x)

    return models.Model(
        [latent_inputs, skip1, skip2, skip3, skip4],
        decoder_outputs,
        name="skip_decoder",
    )


def build_skip_vae(
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
    kl_weight: float = 1.0,
) -> SkipVAE:
    """Build a skip-connected convolutional VAE."""
    _require_tensorflow()

    encoder = build_skip_encoder(
        image_shape=image_shape,
        latent_dim=latent_dim,
    )
    decoder = build_skip_decoder(
        image_shape=image_shape,
        latent_dim=latent_dim,
    )

    return SkipVAE(
        encoder=encoder,
        decoder=decoder,
        image_shape=image_shape,
        kl_weight=kl_weight,
        name="skip_vae",
    )

class PlainVAE(BaseModel):
    """Plain convolutional VAE without skip connections."""

    def __init__(
        self,
        encoder: Any,
        decoder: Any,
        image_shape: tuple[int, int, int] = (64, 64, 3),
        kl_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the plain VAE."""
        _require_tensorflow()
        super().__init__(**kwargs)

        self.encoder = encoder
        self.decoder = decoder
        self.image_shape = image_shape
        self.kl_weight = kl_weight

    def call(self, inputs: Any) -> Any:
        """Run forward pass and add KL loss."""
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)

        kl_loss = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=-1,
        )

        normalization = float(
            self.image_shape[0] * self.image_shape[1] * self.image_shape[2]
        )

        self.add_loss(self.kl_weight * tf.reduce_mean(kl_loss) / normalization)

        return reconstructed


def build_plain_encoder(
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
) -> Any:
    """Build a convolutional VAE encoder without skip connections."""
    _require_tensorflow()

    layers = tf.keras.layers
    models = tf.keras.models

    encoder_inputs = layers.Input(shape=image_shape)

    x = layers.Conv2D(32, 3, strides=2, padding="same")(encoder_inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Flatten()(x)

    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()((z_mean, z_log_var))

    return models.Model(
        encoder_inputs,
        [z_mean, z_log_var, z],
        name="plain_encoder",
    )


def build_plain_decoder(
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
) -> Any:
    """Build a convolutional VAE decoder without skip connections."""
    _require_tensorflow()

    layers = tf.keras.layers
    models = tf.keras.models

    channels = image_shape[-1]

    latent_inputs = layers.Input(shape=(latent_dim,))

    x = layers.Dense(4 * 4 * 256, activation="relu")(latent_inputs)
    x = layers.Reshape((4, 4, 256))(x)

    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    decoder_outputs = layers.Conv2D(
        channels,
        3,
        activation="sigmoid",
        padding="same",
        name="reconstruction",
    )(x)

    return models.Model(
        latent_inputs,
        decoder_outputs,
        name="plain_decoder",
    )


def build_plain_vae(
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
    kl_weight: float = 1.0,
) -> PlainVAE:
    """Build a plain convolutional VAE without skip connections."""
    _require_tensorflow()

    encoder = build_plain_encoder(
        image_shape=image_shape,
        latent_dim=latent_dim,
    )

    decoder = build_plain_decoder(
        image_shape=image_shape,
        latent_dim=latent_dim,
    )

    return PlainVAE(
        encoder=encoder,
        decoder=decoder,
        image_shape=image_shape,
        kl_weight=kl_weight,
        name="plain_vae",
    )

def build_vae_model(
    model_type: str = "skip_vae",
    image_shape: tuple[int, int, int] = (64, 64, 3),
    latent_dim: int = 256,
    kl_weight: float = 1.0,
) -> Any:
    """Build a VAE model by type.

    Supported model types:
    - skip_vae
    - plain_vae
    """
    normalized_model_type = model_type.lower()

    if normalized_model_type in {"skip_vae", "skip"}:
        return build_skip_vae(
            image_shape=image_shape,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
        )

    if normalized_model_type in {"plain_vae", "plain"}:
        return build_plain_vae(
            image_shape=image_shape,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
        )

    raise ValueError(
        f"Unsupported VAE model_type: {model_type}. "
        "Supported values are: skip_vae, plain_vae."
    )