#!/usr/bin/env bash
set -euo pipefail

# ===== Konfiguracja =====
N_TRAIN=56
N_VAL=12
N_TEST=13

SEED=42

SRC_DIR="original"
OUT_DIR="dataset"
TRAIN_DIR="1_trening"
VAL_DIR="2_walidacja"
TEST_DIR="3_test"
# ========================

cd "$(dirname "$0")"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Brak $SRC_DIR — uruchom: uv run python src/prepare_dataset.py" >&2
    exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/$TRAIN_DIR" "$OUT_DIR/$VAL_DIR" "$OUT_DIR/$TEST_DIR"

shuffle() {
    if [[ -n "$SEED" ]]; then
        shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"$SEED$1" -nosalt </dev/zero 2>/dev/null)
    else
        shuf
    fi
}

mapfile -t classes < <(find "$SRC_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
needed=$((N_TRAIN + N_VAL + N_TEST))

for class in "${classes[@]}"; do
    mapfile -t files < <(
        find "$SRC_DIR/$class" -maxdepth 1 -type f -name '*.png' -printf '%f\n' | sort | shuffle "$class"
    )
    total=${#files[@]}
    if (( total < needed )); then
        echo "Klasa '$class' ma tylko $total plików (potrzeba $needed)" >&2
        exit 1
    fi

    mkdir -p "$OUT_DIR/$TRAIN_DIR/$class" "$OUT_DIR/$VAL_DIR/$class" "$OUT_DIR/$TEST_DIR/$class"
    for f in "${files[@]:0:N_TRAIN}"; do cp "$SRC_DIR/$class/$f" "$OUT_DIR/$TRAIN_DIR/$class/"; done
    for f in "${files[@]:N_TRAIN:N_VAL}"; do cp "$SRC_DIR/$class/$f" "$OUT_DIR/$VAL_DIR/$class/"; done
    for f in "${files[@]:N_TRAIN+N_VAL:N_TEST}"; do cp "$SRC_DIR/$class/$f" "$OUT_DIR/$TEST_DIR/$class/"; done
done

echo "Klasy: ${#classes[@]}"
for d in "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR"; do
    echo "$d: $(find "$OUT_DIR/$d" -type f | wc -l) plików"
done
