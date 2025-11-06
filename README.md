# ANO2 – Pipeline de Catégorisation de Feedback

Pipeline schema-agnostic pour catégoriser des feedbacks en trois colonnes: `Category`, `Sub Category`, `Sentiment` (positif ou négatif), sans taxonomie prédéfinie. La consolidation des catégories sémantiquement proches est réalisée via embeddings + clustering (pas de saturation de contexte). Deux modes LLM: `api` (provider OpenAI‑compatible) et `local` (serveur vLLM OpenAI‑compatible).

## Prérequis
- Python 3.10+
- uv (gestionnaire de paquets Python)
- Variables d'environnement: voir `.env.exemple`

## Installation
```bash
uv sync
```

## Configuration
Modifiez `config/pipeline.example.yaml` selon votre source de données:
- `io.path`: chemin du CSV (ou JSONL si `io.type: jsonl`)
- `io.text_field`: colonne contenant le texte du feedback
- `io.output_path`: chemin du CSV de sortie
- `limit`: limite optionnelle du nombre de lignes à traiter (utile pour tester rapidement)
  - `workers`: nombre de workers pour paralléliser les appels LLM (>=1)
  - Horodatage sortie (facultatif):
    - `io.add_timestamp_column`: ajoute une colonne UTC ISO-8601 (ex: `processed_at`)
    - `io.timestamp_column_name`: nom de la colonne d'horodatage
    - `io.append_timestamp_to_output_path`: suffixe l'horodatage dans le nom du fichier
    - `io.timestamp_format`: format du suffixe (par défaut `"%Y%m%d-%H%M%S"`)
  - Récapitulatif (persisté):
    - `io.write_summary`: si vrai, écrit un JSON récapitulatif
    - `io.summary_path`: chemin explicite (sinon `<output>_summary.json`)

LLM via variables d'environnement (exemple dans `.env.exemple`):
- `LLM_MODE=api|local`
- `OPENAI_BASE_URL` (ex: `http://localhost:8000/v1` pour vLLM)
- `OPENAI_API_KEY` (requis pour `api`)
- `LLM_MODEL` (ex: `gpt-4.1` ou modèle local)

## Exécution en local (CLI)
```bash
uv run ano2 -c config/pipeline.example.yaml
```
Résultat: ajoute les colonnes `Category`, `Sub Category`, `Sentiment` et sauvegarde le CSV si `output_path` est défini. Une barre de progression (tqdm) s'affiche durant la classification et la consolidation des sous‑catégories.

Astuce test rapide:
- Dans le YAML, mettez `limit: 5` pour traiter uniquement 5 lignes.

En fin de run, un récapitulatif des comptes est affiché dans les logs:
- Nombre par `Category`
- Nombre par couple (`Category`, `Sub Category`)
Si `io.write_summary: true`, ce même récapitulatif est aussi écrit en JSON.

## API (FastAPI)
Lancement rapide:
```bash
export PIPELINE_CONFIG=config/pipeline.example.yaml
uv run uvicorn ano2.server:app --reload --host 0.0.0.0 --port 8080
```

Exemple d’app (intégration dans votre code):
```python
from ano2.config import PipelineConfig
from ano2.api import create_app

cfg = PipelineConfig.from_yaml("config/pipeline.example.yaml")
app = create_app(cfg)
```
Endpoints:
- `GET /healthz`
- `POST /categorize` avec body `{ "items": [{"id": "1", "text": "..."}] }`

## Notes de conception
- Pas de fallback implicite: en cas de sortie LLM invalide, l'erreur est explicite
- Consolidation par embeddings (multilingue) + agglomératif (seuils configurables)
- Traitement séquentiel par défaut (évite les races)
- Logs concis et utiles (niveau via `LOG_LEVEL`)

## Change Log
- 2025-04-27: Removed the `departement` and `resume` columns from `data/tickets_jira.csv` to align the dataset with current requirements.
