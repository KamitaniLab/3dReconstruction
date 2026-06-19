#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
analysis_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${analysis_dir}/.." && pwd)"

image="${IMAGE:-3d-recon-analysis}"
container_home="${CONTAINER_HOME:-${HOME}/.cache/3d-recon-analysis/home}"

mkdir -p "${container_home}"

if ! docker image inspect "${image}" >/dev/null 2>&1; then
  docker build -t "${image}" "${analysis_dir}"
fi

docker_args=(
  --rm
  --user "$(id -u):$(id -g)"
  -e HOME=/home/user
  -e PATH=/app/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
  -e PYTHONPATH=/app/analysis/src
  -v "${container_home}:/home/user"
  -v "${analysis_dir}:/app/analysis"
  -v "${repo_root}/data:/app/data:ro"
  -w /app/analysis
)

if [[ -t 0 && -t 1 ]]; then
  docker_args=(-it "${docker_args[@]}")
fi

if (($#)); then
  docker run "${docker_args[@]}" "${image}" "$@"
else
  docker run "${docker_args[@]}" "${image}" bash
fi
