"""
Test simple pour vérifier que l'installation est correcte.
"""
import os
import sys
import unittest

class TestInstallation(unittest.TestCase):
    def test_python_version(self):
        """Vérifier la version de Python"""
        major, minor = sys.version_info[:2]
        self.assertGreaterEqual(major, 3, "Python 3 ou supérieur est requis")
        self.assertGreaterEqual(minor, 6, "Python 3.6 ou supérieur est requis")
    
    def test_required_packages(self):
        """Vérifier que les packages requis sont installés"""
        required_packages = [
            "tensorflow", "keras", "numpy", "pandas", 
            "scikit-learn", "matplotlib", "pillow", "nltk",
            "mlflow", "flask", "dvc"
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                self.fail(f"Le package {package} n'est pas installé")
    
    def test_directory_structure(self):
        """Vérifier que la structure de répertoires est correcte"""
        # Monter au répertoire parent
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        required_dirs = [
            "src", "models", "tests", "notebooks", "logs", 
            "mlflow", ".github/workflows", ".dvc"
        ]
        
        for directory in required_dirs:
            self.assertTrue(
                os.path.exists(directory), 
                f"Le répertoire {directory} n'existe pas"
            )
    
    def test_mlops_files(self):
        """Vérifier que les fichiers MLOps sont présents"""
        # Monter au répertoire parent
        os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        required_files = [
            "Dockerfile", "docker-compose.yml", "dvc.yaml", 
            "params.yaml", ".gitignore", ".env", 
            "src/api.py", "src/main.py", "src/evaluate.py"
        ]
        
        for file in required_files:
            self.assertTrue(
                os.path.exists(file), 
                f"Le fichier {file} n'existe pas"
            )

if __name__ == "__main__":
    unittest.main()
