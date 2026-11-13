#!/usr/bin/env bash
set -euo pipefail

# Dzieli train na trening/walidację; test zostaje osobno (wymaganie zadania).
VAL_FRAC=20   # procent z train na walidację
SEED=42

SRC_DIR="data/apples_tomatoes"
OUT_DIR="dataset"
TRAIN_DIR="1_trening"
VAL_DIR="2_walidacja"
TEST_DIR="3_test"

cd "$(dirname "$0")"

if [[ ! -d "$SRC_DIR/train" ]]; then
    echo "Brak $SRC_DIR/train" >&2
    exit 1
fi

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR/$TRAIN_DIR" "$OUT_DIR/$VAL_DIR" "$OUT_DIR/$TEST_DIR"

shuffle() {
    shuf --random-source=<(openssl enc -aes-256-ctr -pass pass:"${SEED}${1}" -nosalt </dev/zero 2>/dev/null)
}

mapfile -t classes < <(find "$SRC_DIR/train" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

for class in "${classes[@]}"; do
    mapfile -t files < <(
        find "$SRC_DIR/train/$class" -maxdepth 1 -type f \( -name '*.jpeg' -o -name '*.jpg' \) -printf '%f\n' | sort | shuffle "$class"
    )
    total=${#files[@]}
    n_val=$(( (total * VAL_FRAC + 50) / 100 ))
    n_train=$(( total - n_val ))

    mkdir -p "$OUT_DIR/$TRAIN_DIR/$class" "$OUT_DIR/$VAL_DIR/$class"
    for f in "${files[@]:0:n_train}"; do
        cp "$SRC_DIR/train/$class/$f" "$OUT_DIR/$TRAIN_DIR/$class/"
    done
    for f in "${files[@]:n_train:n_val}"; do
        cp "$SRC_DIR/train/$class/$f" "$OUT_DIR/$VAL_DIR/$class/"
    done
done

for class in "${classes[@]}"; do
    mkdir -p "$OUT_DIR/$TEST_DIR/$class"
    find "$SRC_DIR/test/$class" -maxdepth 1 -type f \( -name '*.jpeg' -o -name '*.jpg' \) -exec cp {} "$OUT_DIR/$TEST_DIR/$class/" \;
done

echo "Podział (seed=$SEED, walidacja=${VAL_FRAC}% train):"
for d in "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR"; do
    echo "  $d: $(find "$OUT_DIR/$d" -type f | wc -l) plików"
done
