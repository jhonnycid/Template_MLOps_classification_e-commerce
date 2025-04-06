import torch
import yaml

def get_device(force_cpu=False):
    """
    Sélectionne le device en fonction de la configuration et des arguments.
    
    :param force_cpu: Force l'utilisation du CPU
    :return: Device PyTorch (cuda, mps, ou cpu)
    """
    # Charger la configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    use_mps = params.get('use_mps', False)
    
    # Force CPU si demandé
    if force_cpu:
        return torch.device('cpu')
    
    # Vérifier la disponibilité de MPS
    if use_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device('mps')
    
    # Vérifier la disponibilité de CUDA
    if torch.cuda.is_available():
        return torch.device('cuda')
    
    # Retourner CPU par défaut
    return torch.device('cpu')

def ensure_tensor_on_device(tensor, device):
    """
    S'assure qu'un tensor est sur le bon device.
    
    :param tensor: Tensor PyTorch
    :param device: Device cible
    :return: Tensor sur le device spécifié
    """
    return tensor.to(device)

def ensure_model_on_device(model, device):
    """
    S'assure qu'un modèle est sur le bon device.
    
    :param model: Modèle PyTorch
    :param device: Device cible
    :return: Modèle sur le device spécifié
    """
    return model.to(device)
