#!/usr/bin/env bash
set -euo pipefail

# Simple launcher for ANO2: pipeline or API
# - Loads .env if present
# - Uses uv to run the Python entrypoints

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_CONFIG="config/pipeline.example.yaml"

usage() {
  cat <<'USAGE'
Usage:
  ./start.sh <config.yaml>               # lance le pipeline (raccourci)
  ./start.sh pipeline [-c <config.yaml>]
  ./start.sh api [-c <config.yaml>] [-H <host>] [-p <port>] [--no-reload]
  ./start.sh install   # uv sync

Notes:
  - .env est chargé automatiquement si présent
  - Config par défaut: config/pipeline.example.yaml
USAGE
}

load_env() {
  if [[ -f .env ]]; then
    # Export variables from .env without printing them
    set -a
    # shellcheck disable=SC1091
    . ./.env
    set +a
  fi
  # Ensure unbuffered stdout/stderr so tqdm/logs flush correctly
  export PYTHONUNBUFFERED=1
}

check_uv() {
  if ! command -v uv >/dev/null 2>&1; then
    echo "[ERR] 'uv' non trouvé. Installez-le via:" >&2
    echo "      curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
}

cmd_install() {
  check_uv
  load_env
  echo "[INFO] Installation des dépendances (uv sync)..."
  uv sync
  echo "[OK] Dépendances installées."
}

cmd_pipeline() {
  local config="$DEFAULT_CONFIG"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -c|--config)
        config="$2"; shift 2;;
      -h|--help)
        usage; exit 0;;
      *)
        echo "[ERR] Option inconnue: $1" >&2; usage; exit 2;;
    esac
  done
  check_uv
  load_env
  echo "[INFO] Lancement pipeline avec config: $config"
  # Use console script with default callback enabled; PYTHONUNBUFFERED ensures real-time logs/tqdm
  uv run ano2 -c "$config"
}

cmd_api() {
  local config="$DEFAULT_CONFIG" host="0.0.0.0" port="8080" reload=1
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -c|--config) config="$2"; shift 2;;
      -H|--host) host="$2"; shift 2;;
      -p|--port) port="$2"; shift 2;;
      --no-reload) reload=0; shift;;
      -h|--help) usage; exit 0;;
      *) echo "[ERR] Option inconnue: $1" >&2; usage; exit 2;;
    esac
  done
  check_uv
  load_env
  export PIPELINE_CONFIG="$config"
  echo "[INFO] Lancement API (host=$host port=$port) avec config: $config"
  if [[ $reload -eq 1 ]]; then
    uv run uvicorn ano2.server:app --host "$host" --port "$port" --reload
  else
    uv run uvicorn ano2.server:app --host "$host" --port "$port"
  fi
}

main() {
  if [[ $# -lt 1 ]]; then
    usage; exit 1
  fi
  local sub="$1"; shift || true
  case "$sub" in
    pipeline) cmd_pipeline "$@" ;;
    api)      cmd_api "$@" ;;
    install)  cmd_install "$@" ;;
    -h|--help|help) usage; exit 0 ;;
    *)
      # Raccourci: ./start.sh config.yaml
      if [[ -f "$sub" || "$sub" == *.yml || "$sub" == *.yaml ]]; then
        cmd_pipeline -c "$sub"
      else
        echo "[ERR] Argument inconnu: $sub" >&2; usage; exit 1
      fi
      ;;
  esac
}

main "$@"
