#!/usr/bin/env bash
# Downloads CubiCasa5K (Kalervo et al., SCIA 2019; CC BY-NC 4.0) from Zenodo (~5.3 GB)
# and pre-rasterises masks for the chosen taxonomy.
set -euo pipefail
DEST=${1:-data}
URL=${CUBICASA_URL:-https://zenodo.org/record/2613548/files/cubicasa5k.zip}
mkdir -p "$DEST"
if [ ! -d "$DEST/cubicasa5k" ]; then
  echo "Downloading CubiCasa5K -> $DEST"
  wget -c -O "$DEST/cubicasa5k.zip" "$URL"
  unzip -q "$DEST/cubicasa5k.zip" -d "$DEST"
  rm -f "$DEST/cubicasa5k.zip"
fi
python scripts/prepare_cubicasa.py --root "$DEST/cubicasa5k" --taxonomy "${TAXONOMY:-coarse}"
