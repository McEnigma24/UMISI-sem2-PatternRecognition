#!/usr/bin/env bash
set -euo pipefail

# ===== Konfiguracja =====
N_TRAIN=14
N_VAL=3
N_TEST=3

SEED=42                          # pusty ("") = za każdym razem inny podział

SRC_DIR="original"
OUT_DIR="dataset"
TRAIN_DIR="1_trening"
VAL_DIR="2_walidacja"
TEST_DIR="3_test"

PER_CLASS_SUBDIRS=true           # true: trening/<klasa>/plik.gif, false: trening/plik.gif
# ========================

cd "$(dirname "$0")"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "Brak katalogu źródłowego: $SRC_DIR" >&2
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

copy_files() {
    local dest="$1" class="$2"
    shift 2
    if [[ "$PER_CLASS_SUBDIRS" == true ]]; then
        dest="$dest/$class"
    fi
    mkdir -p "$dest"
    for f in "$@"; do
        cp "$SRC_DIR/$f" "$dest/"
    done
}

mapfile -t classes < <(
    find "$SRC_DIR" -maxdepth 1 -type f -regextype posix-extended -regex '.*/.+-[0-9]+\.gif' -printf '%f\n' \
        | sed -E 's/-[0-9]+\.gif$//' | sort -u
)

for class in "${classes[@]}"; do
    mapfile -t files < <(
        find "$SRC_DIR" -maxdepth 1 -type f -regextype posix-extended -regex ".*/${class}-[0-9]+\.gif" -printf '%f\n' \
            | sort | shuffle "$class"
    )

    total=${#files[@]}
    needed=$((N_TRAIN + N_VAL + N_TEST))
    if (( total < needed )); then
        echo "Klasa '$class' ma tylko $total plików (potrzeba $needed)" >&2
        exit 1
    fi

    copy_files "$OUT_DIR/$TRAIN_DIR" "$class" "${files[@]:0:N_TRAIN}"
    copy_files "$OUT_DIR/$VAL_DIR"   "$class" "${files[@]:N_TRAIN:N_VAL}"
    copy_files "$OUT_DIR/$TEST_DIR"  "$class" "${files[@]:N_TRAIN+N_VAL:N_TEST}"
done

echo "Klasy: ${#classes[@]}"
for d in "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR"; do
    echo "$d: $(find "$OUT_DIR/$d" -type f | wc -l) plików"
done
