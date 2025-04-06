import torch
import yaml
import platform

def check_device_compatibility():
    """
    Diagnostics complets sur la compatibilité des devices
    """
    print("=== Diagnostics de compatibilité des devices ===")
    print(f"Système d'exploitation: {platform.platform()}")
    
    # Vérification de CUDA
    print("\n=== CUDA ===")
    print(f"CUDA disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Nombre de GPUs CUDA: {torch.cuda.device_count()}")
        print(f"GPU actuel: {torch.cuda.current_device()}")
        print(f"Nom du GPU: {torch.cuda.get_device_name(0)}")
    
    # Vérification de MPS
    print("\n=== Metal Performance Shaders (MPS) ===")
    print(f"MPS disponible: {torch.backends.mps.is_available()}")
    print(f"MPS construit: {torch.backends.mps.is_built()}")
    
    # Lecture de la configuration
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)
    
    use_mps = params.get('use_mps', False)
    print(f"\nConfiguration use_mps dans params.yaml: {use_mps}")
    
    # Recommandation de device
    print("\n=== Recommandation ===")
    if torch.cuda.is_available():
        print("✅ CUDA est disponible. Recommandation : utilisez CUDA.")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("✅ MPS est disponible. Vous pouvez l'activer dans params.yaml.")
    else:
        print("⚠️ Utilisation du CPU recommandée. Aucun GPU détecté.")

def test_device_performance():
    """
    Test rapide des performances du device
    """
    print("\n=== Test de performance du device ===")
    
    # Créer un tenseur de test
    x = torch.rand(10000, 10000)
    
    # Test de performance sur différents devices
    devices = [
        torch.device('cpu'),
        torch.device('cuda') if torch.cuda.is_available() else None,
        torch.device('mps') if torch.backends.mps.is_available() else None
    ]
    
    for device in devices:
        if device is None:
            continue
        
        print(f"\nTest sur {device}")
        x_device = x.to(device)
        
        # Multiplication matricielle comme benchmark
        import time
        start_time = time.time()
        _ = torch.matmul(x_device, x_device)
        end_time = time.time()
        
        print(f"Temps d'exécution : {(end_time - start_time)*1000:.2f} ms")

if __name__ == "__main__":
    check_device_compatibility()
    test_device_performance()
