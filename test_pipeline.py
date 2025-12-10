#!/usr/bin/env python
"""
Script de test pour exécuter le pipeline complet avec données réelles
Simule les entrées utilisateur pour automatiser le processus
"""
import sys
import os

# Rediriger les inputs pour simuler les choix utilisateur
# Pour le pipeline complet (option 5)
# Répondre "non" à toutes les questions pour forcer la recréation

test_inputs = [
    "5\n",           # Option 5: Pipeline complet
    "n\n",           # Réutiliser raw? Non
    "n\n",           # Réutiliser clean? Non
    "n\n",           # Réutiliser features? Non
    "n\n",           # Réutiliser modèle? Non
    "7\n"            # Option 7: Quitter
]

# Créer un pipe avec les inputs
import io
sys.stdin = io.StringIO("".join(test_inputs))

# Maintenant importer et lancer main
from main import main_menu

if __name__ == "__main__":
    main_menu()
