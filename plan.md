# Plan de Pipeline de Catégorisation de Feedback

Objectif: Construire une pipeline schema-agnostic qui catégorise des feedbacks en Catégorie, Sous-catégorie et Sentiment (positif/négatif), sans taxonomie prédéfinie. Deux modes LLM: `local` (vLLM) et `api` (provider OpenAI-compatible).

Étapes:
1) Scaffold du projet avec uv + structure `src/` minimale, logs, config YAML
2) Abstraction LLM (api/local vLLM) + gestion prompts/validation JSON
3) Embeddings et clustering pour fusion sémantique (catégorie + sous-catégorie)
4) Pipeline batch: chargement CSV, appels LLM, fusion, export
5) Interfaces: CLI (local) et API (FastAPI) avec endpoints batch
6) YAML de config (entrées, sorties, LLM, embeddings, batch, seuils)
7) Documentation (README) + exemple d’exécution
8) Barre de progression via tqdm pour le batch et la consolidation des sous-catégories
9) Paramètre `limit` dans le YAML pour limiter rapidement le nombre de lignes traitées lors des tests
10) Clarifier l'usage du CLI: `uv run ano2 -c ...` (commande par défaut)
11) Ajout horodatage output (colonne + suffixe de fichier) piloté par YAML
12) Récapitulatif des comptes par catégorie et sous-catégorie dans les logs de fin
13) Persister le récapitulatif en JSON optionnellement (piloté par YAML)
14) Optimiser le clustering via déduplication des libellés (cat/sub)
15) Ajouter la parallélisation contrôlée des appels LLM (`workers`)
16) Support d'un `output_dir` avec sous-dossier horodaté contenant CSV + JSON

Contraintes:
- Pas de fallback implicite; erreurs explicites et configurables
- Logs utiles uniquement; éviter artefacts
- Éviter les races; traitement séquentiel par défaut
- Modularité stricte; code minimal

Livrables:
- `pyproject.toml` compatible uv
- `src/ano2/` (config, llm, embed, pipeline, api, cli, log)
- `config/pipeline.example.yaml`
- Mise à jour `README.md`
