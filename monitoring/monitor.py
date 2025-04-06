import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from pathlib import Path
from bs4 import BeautifulSoup

# Fonction de nettoyage HTML
def clean_html(text):
    try:
        return BeautifulSoup(text, "html.parser").get_text()
    except:
        return text  # en cas de valeur non string

# Chargement des fichiers CSV avec renommage des colonnes dupliquées
reference = pd.read_csv("data/reference.csv")
current = pd.read_csv("data/current.csv")

# Sélection des colonnes pertinentes (exemple ici : designation + 1 description)
reference = reference[["designation", "description"]].copy()
current = current[["designation", "description"]].copy()

# Nettoyage de la colonne 'description'
reference["description"] = reference["description"].apply(clean_html)
current["description"] = current["description"].apply(clean_html)

# Création du rapport Evidently
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

# Sauvegarde des rapports
Path("monitoring/reports").mkdir(parents=True, exist_ok=True)
report.save_html("monitoring/reports/drift_report.html")
report.save_json("monitoring/reports/drift_report.json")

print("✅ Rapports générés dans 'monitoring/reports/' (plus léger grâce au nettoyage)")
