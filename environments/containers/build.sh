#!/usr/bin/env bash
# Explicit on-target build only. Existing evidence directories are preserved.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash environments/containers/build.sh --mode upstream|upstream-cu118|modern [--tag IMAGE] [--output PATH]

Builds one Linux x86_64 CUDA tooling container from a pinned official NVIDIA
image. Saves Docker build logs, immutable image ID, image inspection and recipe
copies. Does not install Python inference packages or execute GPU workloads.
EOF
}

mode=""
image_tag=""
output_dir=""
ncu_release=2024.1.1
ncu_package_version=2024.1.1.4-1
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) mode="${2:?missing mode}"; shift 2 ;;
    --tag) image_tag="${2:?missing image tag}"; shift 2 ;;
    --output) output_dir="${2:?missing output directory}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

case "$mode" in
  upstream)
    cuda_release=11.7
    image_tag="${image_tag:-drivebench/vista-upstream:cuda11.7.1-nsight2024}"
    cuda_image='nvidia/cuda:11.7.1-devel-ubuntu22.04@sha256:a3184e4dc6f968da5bba86df3081ff3013f8e3674a9bfce544af8be905d2f17a'
    ;;
  upstream-cu118)
    cuda_release=11.8
    ncu_release=2025.2.1
    ncu_package_version=2025.2.1.3-1
    image_tag="${image_tag:-drivebench/vista-upstream-cu118:cuda11.8.0-ncu2025.2.1}"
    cuda_image='nvidia/cuda:11.8.0-devel-ubuntu22.04@sha256:60eda04ab6790aa76d73bf0df245b361eabc6d8f7b6f6cf9846c70f399b9a1eb'
    ;;
  modern)
    cuda_release=12.4
    image_tag="${image_tag:-drivebench/vista-modern:cuda12.4.1-nsight2024}"
    cuda_image='nvidia/cuda:12.4.1-devel-ubuntu22.04@sha256:5645fec64549cc35930eee9d85aafd2b0006c0c3f22632be5a1d85e2604e9749'
    ;;
  *) usage >&2; exit 2 ;;
esac

[[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || {
  echo 'Run the container build on the Linux x86_64 target.' >&2; exit 1;
}
command -v docker >/dev/null || { echo 'Docker is required.' >&2; exit 1; }
recipe_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$recipe_dir/../.." && pwd)"
output_dir="${output_dir:-$repo_root/outputs/environments/containers/$mode}"
[[ ! -e "$output_dir" ]] || {
  echo "Refusing to overwrite container build evidence: $output_dir" >&2; exit 1;
}
mkdir -p -- "$output_dir"
output_dir="$(cd -- "$output_dir" && pwd)"
cp -- "$recipe_dir/Dockerfile" "$recipe_dir/build.sh" "$recipe_dir/README.md" "$output_dir/"
printf '%s\n' "$cuda_image" > "$output_dir/base-image.txt"
printf '%s\n' "$image_tag" > "$output_dir/image-tag.txt"

# Context is the recipe directory, never the repository/dataset/checkpoint.
docker build --platform linux/amd64 \
  --build-arg "MODE=$mode" --build-arg "CUDA_RELEASE=$cuda_release" \
  --build-arg "CUDA_IMAGE=$cuda_image" \
  --build-arg "NCU_RELEASE=$ncu_release" --build-arg "NCU_PACKAGE_VERSION=$ncu_package_version" \
  --iidfile "$output_dir/image-id.txt" --tag "$image_tag" \
  --file "$recipe_dir/Dockerfile" "$recipe_dir" 2>&1 | tee "$output_dir/build.log"
image_id="$(cat "$output_dir/image-id.txt")"
docker image inspect "$image_id" > "$output_dir/image-inspect.json"
echo "Built $image_id; evidence: $output_dir"
echo 'Run the project bootstrap and real GPU preflight before downloading the checkpoint.'
