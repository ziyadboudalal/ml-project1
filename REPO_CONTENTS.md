# Contenu exact du dépôt final CouscousAI

La liste exhaustive est `DELIVERY_FILES.txt` (30 fichiers). Le dossier prêt à
remettre et l'archive de livraison contiennent exactement ces fichiers.

```text
.
├── .gitattributes
├── .gitignore
├── README.md
├── REPO_CONTENTS.md
├── DELIVERY_FILES.txt
├── requirements.txt
├── implementations.py
├── helpers.py
├── project_io.py
├── final_pipeline.py
├── run.py
├── submission.csv
├── submission.manifest.json
├── artifacts/
│   ├── final_model.npz
│   ├── delivery_verification.json
│   └── validation/
│       ├── protocol.json
│       ├── results.json
│       └── oof_predictions.npz
├── experiments/
│   └── validate_final.py
├── tests/
│   └── test_final_pipeline.py
└── documentation/
    ├── Results.md
    ├── final_figures.py
    ├── final_figures/
    │   ├── threshold.png
    │   ├── model_comparison.png
    │   └── provenance.json
    └── report/
        ├── IEEEtran.cls
        ├── report.tex
        ├── results_summary.tex
        ├── results_table.tex
        └── report.pdf
```

## Pourquoi garder ces fichiers

- Les scripts à la racine contiennent les six méthodes imposées, notre pipeline
  final, le chargement des données et la production du CSV.
- Les poids sauvegardés permettent de reproduire la soumission sans réentraîner.
- Les résultats et prédictions de validation permettent de régénérer nos
  figures immédiatement et de vérifier les chiffres du rapport.
- Le rapport PDF est le document à rendre ; le source LaTeX et la classe
  permettent sa recompilation.
- Le test commun vérifie notre pipeline final. Les manifestes enregistrent les
  empreintes des fichiers et les vérifications effectuées.

## À garder seulement en local ou dans vos archives

- `dataset/` : nécessaire pour exécuter les commandes, mais exclu de Git.
- `submissions/` : anciennes variantes et notebooks, déjà consolidés dans les
  scripts finaux. Garder une copie d'archive avant tout nettoyage.
- `documentation/figures.py`, `documentation/figures/` et
  `documentation/ziyad_documentation/` : anciennes analyses avec d'autres
  protocoles, hors du rendu final.
- `tests/test_ismail.py` : tests de l'ancienne variante, à archiver avec elle.
- `documentation/project1_description.pdf` : consignes, pas un livrable.
- `documentation/report/IEEEtran.bst` : inutile ici, car la bibliographie est
  directement incluse dans le source LaTeX.
- Les fichiers `fold_*.npz`, `__pycache__/`, `.pytest_cache/`, `.aux`, `.log`,
  `.out`, les outils temporaires et les dossiers de vérification.

Ne supprimez pas le dossier interne `.git` de votre dépôt de travail. L'archive
de livraison l'exclut simplement, comme elle exclut les données brutes.

## Dernière étape en ligne

Notre CSV livré a été évalué avec succès sur AIcrowd le 28 octobre 2026 :
soumission **338118**, **F1 = 0,455**, **accuracy = 0,886**. Le rapport et le
manifeste intègrent ce résultat, rattaché à l'empreinte exacte du CSV.
Il reste à relire ensemble le rendu final, puis à remettre le PDF et le lien
GitHub vers le commit exact sur la plateforme du cours.
