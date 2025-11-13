# ANO2 – Pipeline de Catégorisation de Feedback

Pipeline schema-agnostic pour catégoriser des feedbacks en trois colonnes: `Category`, `Sub Category`, `Sentiment` (positif / neutre / négatif), et extraire une liste de mots‑clés `Keywords` liés au problème. Pas de taxonomie prédéfinie. La consolidation des catégories sémantiquement proches est réalisée via embeddings + clustering (pas de saturation de contexte). Deux modes LLM: `api` (provider OpenAI‑compatible) et `local` (serveur vLLM OpenAI‑compatible).

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
- `limit`: limite optionnelle du nombre de lignes à traiter. Par défaut (`null` ou omis) tout le fichier est traité; mettez un entier pour échantillonner (utile pour tester rapidement).
  - `workers`: nombre de workers pour paralléliser les appels LLM (>=1)
  - Horodatage sortie (facultatif):
    - `io.add_timestamp_column`: ajoute une colonne UTC ISO-8601 (ex: `processed_at`)
    - `io.timestamp_column_name`: nom de la colonne d'horodatage
    - `io.append_timestamp_to_output_path`: suffixe l'horodatage dans le nom du fichier (Option A)
    - `io.timestamp_subdir`: crée un sous-dossier horodaté (Option B)
    - `io.timestamp_format`: format du suffixe (par défaut `"%Y%m%d-%H%M%S"`)
  - Récapitulatif (persisté):
    - `io.write_summary`: si vrai, écrit un JSON récapitulatif
    - `io.summary_path`: chemin explicite (sinon dérivé)
    - `io.summary_basename`: si `output_dir` est utilisé, nom du JSON (sinon `<stem>_summary.json`)

Schémas d’écriture recommandés:
- Option A (héritée): fichier direct avec suffixe horodaté
  - `io.output_path: data/output/categorized.csv`
  - `io.append_timestamp_to_output_path: true`
- Option B (recommandée): dossier avec sous-dossier horodaté contenant CSV + JSON
  - `io.output_dir: data/output`
  - `io.output_basename: categorized.csv`
  - `io.timestamp_subdir: true`
  - `io.write_summary: true`, `io.summary_basename: summary.json`

LLM via variables d'environnement (exemple dans `.env.exemple`):
- `LLM_MODE=api|local`
- `OPENAI_BASE_URL` (ex: `http://localhost:8000/v1` pour vLLM)
- `OPENAI_API_KEY` (requis pour `api`)
- `LLM_MODEL` (ex: `gpt-4.1` ou modèle local)

Variables d'environnement (mots‑clés):
- `KEYWORDS_ENFORCE_IN_TEXT` (default `1`): ne garder que des mots présents dans le texte.
- `KEYWORDS_DROP_GENERIC` (default `1`): retirer des termes génériques (ex: "issue", "problem", "application", "utilisateur").
- `KEYWORDS_MIN_LENGTH` (default `2`): longueur minimale (après normalisation) pour garder un mot‑clé.
- `KEYWORDS_SINGLE_WORDS_ONLY` (default `1`): n'autoriser que des mots simples (pas de phrases, ni noms composés avec espace/tiret).

JSON strict vs tolérant:
- `LLM_STRICT_JSON` (default `1`): en mode strict, la réponse du modèle DOIT être un JSON valide (objet) sans texte additionnel; sinon erreur. Recommandé en prod.
- `LLM_JSON_MODE` (default `0`): si supporté par le provider (OpenAI, etc.), active `response_format: {type: json_object}` pour forcer un JSON côté modèle.

## Exécution en local (CLI)
Usage simple avec script:
```bash
# Raccourci direct
./start.sh config/pipeline.example.yaml

# Ou forme explicite
./start.sh pipeline -c config/pipeline.example.yaml
```
Résultat: ajoute les colonnes `Category`, `Sub Category`, `Sentiment` (positif / neutre / négatif) et `Keywords` (liste de mots‑clés).
Par défaut, `Keywords` est sérialisée en chaîne JSON (ex: `["latence","erreur 500"]`) dans le CSV. Pour usage NL‑SQL/
textuel simple, vous pouvez aussi activer `field_names.keywords_text` (ex: `Keywords Text`) qui contient une version jointe `latence; erreur 500`.

Note sur les mots‑clés: ils sont ancrés dans le texte. Le modèle est instruit de n’extraire que des mots/expressions réellement présents dans le feedback et un filtre retire les termes trop génériques. La liste peut donc être plus courte si le texte ne contient que peu de termes pertinents.

Astuce test rapide:
- Par défaut, toutes les lignes du CSV sont traitées. Pour tester rapidement, mettez `limit: 5` (ou tout autre entier) pour ne traiter qu'un échantillon.

En fin de run, un récapitulatif des comptes est affiché dans les logs:
- Nombre par `Category`
- Nombre par couple (`Category`, `Sub Category`)
Si `io.write_summary: true`, ce même récapitulatif est aussi écrit en JSON.

## API (FastAPI)
Lancement rapide:
```bash
export PIPELINE_CONFIG=config/pipeline.example.yaml
./start.sh api -c "$PIPELINE_CONFIG" -H 0.0.0.0 -p 8080
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

## Script de démarrage
`start.sh` facilite le lancement:
- `./start.sh install` — installe les dépendances (uv sync)
- `./start.sh pipeline -c <config.yaml>` — exécute le pipeline CLI
- `./start.sh api -c <config.yaml> [-H host] [-p port] [--no-reload]` — lance l’API

## Notes de conception
- Pas de fallback implicite: en cas de sortie LLM invalide, l'erreur est explicite
- Consolidation par embeddings (multilingue) + agglomératif (seuils configurables)
- Traitement séquentiel par défaut (évite les races)
- Logs concis et utiles (niveau via `LOG_LEVEL`)

## Change Log
- 2025-04-27: Removed the `departement` and `resume` columns from `data/tickets_jira.csv` to align the dataset with current requirements.
