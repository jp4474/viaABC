import os
import numpy as np
import yaml

def load_pretrained_model(
    folder_name: str,
    model_class,
    lightning_class,
    checkpoint_substr: str = "SUM",
    in_chans: int = 3
):
    """
    Load a pretrained model from a folder containing a YAML config and checkpoint.

    Args:
        folder_name (str): Path to the folder containing `config.yaml` and checkpoint.
        model_class (class): Model class to instantiate (e.g., MaskedAutoencoderViT).
        lightning_class (class): PyTorch Lightning wrapper class (e.g., PreTrainLightningSpatial2D).
        checkpoint_substr (str): Substring to match the desired checkpoint filename.
        in_chans (int): Number of input channels for the model.

    Returns:
        An instance of the loaded PyTorch Lightning model.
    """

    config_path = os.path.join(folder_name, "config.yaml")
    config = yaml.safe_load(open(config_path))

    model = model_class(**config["model"]["params"], in_chans=in_chans)

    checkpoint_files = [f for f in os.listdir(folder_name) if f.endswith(".ckpt") and checkpoint_substr in f]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint file containing '{checkpoint_substr}' found in {folder_name}")
    
    checkpoint_path = os.path.join(folder_name, checkpoint_files[0])
    pl_model = lightning_class.load_from_checkpoint(checkpoint_path, model=model)
    pl_model.eval()  # Set the model to evaluation mode

    print("Successfully loaded model")
    return pl_model

def labels2map(y):
    """
    Convert 28 x 28 class labels into 3 x 28 x 28 binary masks.
    Assumes:
        - label_map is a torch.Tensor of shape (28, 28), where each pixel is 0, 1, or 2
    Returns:
        - imgs: torch.Tensor of shape (3, 28, 28), where each channel is a binary mask for class 0, 1, or 2
    """

    susceptible = (y == 0)
    infected = (y == 1)
    resistant = (y == 2)

    y_onehot = np.stack([susceptible, infected, resistant], axis=1)  # Shape: (3, H, W)

    return y_onehot