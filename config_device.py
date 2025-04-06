import yaml
import sys

def update_mps_config(use_mps):
    """
    Met à jour la configuration MPS dans params.yaml
    
    :param use_mps: Booléen indiquant si MPS doit être activé
    """
    # Lire la configuration actuelle
    with open('params.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Mettre à jour la configuration
    config['use_mps'] = use_mps
    
    # Écrire la nouvelle configuration
    with open('params.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Configuration MPS mise à jour : {'Activé' if use_mps else 'Désactivé'}")

def main():
    # Gérer les arguments en ligne de commande
    if len(sys.argv) != 2:
        print("Usage: python config_device.py [on|off]")
        sys.exit(1)
    
    action = sys.argv[1].lower()
    
    if action == 'on':
        update_mps_config(True)
    elif action == 'off':
        update_mps_config(False)
    else:
        print("Argument invalide. Utilisez 'on' ou 'off'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
