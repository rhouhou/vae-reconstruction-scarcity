from vae_scarcity.training.reconstruction_training import (
    compile_reconstruction_model,
    reconstruct_images,
    save_reconstruction_model,
    train_reconstruction_model,
)


def test_reconstruction_training_helpers_are_importable():
    assert callable(compile_reconstruction_model)
    assert callable(train_reconstruction_model)
    assert callable(reconstruct_images)
    assert callable(save_reconstruction_model)