# Contenu exact du dépôt final CouscousAI

La branche à relire et à remettre est **`couscousAI-final`**. Garder tous les
fichiers de [DELIVERY_FILES.txt](DELIVERY_FILES.txt), soit **31 fichiers**.
Le dossier de livraison et son archive contiennent cette même liste.
Le code nécessaire à la correction reste à la racine du dépôt.

| Emplacement | Rôle |
|---|---|
| Racine | README, six méthodes imposées, pipeline, exécution et soumission |
| `artifacts/` | Poids entraînés, validation sauvegardée et preuves de reproduction |
| `experiments/` | Script pour relancer les comparaisons sur les mêmes folds |
| `tests/` | Tests de notre pipeline et de l'intégrité des prédictions |
| `documentation/` | Résultats, figures, consignes de remise et rapport LaTeX/PDF |

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
    ├── SUBMISSION.md
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

## Fichiers locaux exclus de la remise

Les données `dataset/`, l'environnement Python `.venv/`, les caches de folds,
les sorties de reproduction dans `artifacts/reproduction/` et les fichiers
temporaires LaTeX sont ignorés par Git. Ils ne font pas partie des 31 fichiers.

Les anciennes variantes individuelles et leurs notebooks restent consultables
dans l'historique Git. Leur code utile a été consolidé dans le pipeline commun ;
la branche finale contient une seule version de référence.

Ne supprimez pas le dossier interne `.git` de votre dépôt de travail. L'archive
de livraison l'exclut simplement, comme elle exclut les données brutes.

## Dernière étape en ligne

Notre CSV livré a été évalué avec succès sur AIcrowd le 28 octobre 2026 :
soumission **338118**, **F1 = 0,455**, **accuracy = 0,886**. Le rapport et le
manifeste intègrent ce résultat, rattaché à l'empreinte exacte du CSV.
Il reste à relire ensemble le rendu final, puis à remettre le PDF et le lien
GitHub vers le commit exact sur la plateforme du cours. Les étapes et la
correspondance avec l'énoncé sont dans
[documentation/SUBMISSION.md](documentation/SUBMISSION.md).
