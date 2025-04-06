# Configuration MPS (Metal Performance Shaders)

## Introduction

Ce projet supporte la configuration dynamique de l'accélération matérielle via le flag `use_mps` dans `params.yaml`.

## Prérequis

- macOS avec Apple Silicon (M1, M2, etc.)
- Python 3.8+
- PyTorch compatible MPS

## Configuration

### Activation de MPS

1. Ouvrez `params.yaml`
2. Modifiez la valeur de `use_mps`
   - `false` : Utilise CUDA si disponible, sinon CPU
   - `true` : Tente d'utiliser MPS si disponible

### Diagnostic

Utilisez le script de diagnostics :

```bash
python diagnostics.py
```

Ce script vérifiera :
- La disponibilité de CUDA
- La disponibilité de MPS
- Les recommandations de configuration

## Comportement du Device

Le device est sélectionné dans l'ordre de priorité :
1. MPS (si `use_mps: true` et disponible)
2. CUDA (si disponible)
3. CPU (par défaut)

## Problèmes Courants

- Assurez-vous d'avoir une version récente de PyTorch
- Vérifiez la compatibilité de votre matériel
- Consultez la documentation de PyTorch pour les détails MPS

## Performance

Les performances peuvent varier. Utilisez `diagnostics.py` pour comparer les performances entre différents devices.
