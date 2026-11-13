# Ćwiczenie 2 — klasyfikacja według tekstur (GLCM)

Klasyfikacja **5 tekstur** z **KTH-TIPS-2** (Kaggle) na podstawie cech Haralicka z macierzy GLCM
(`scikit-image`: `graycomatrix`, `graycoprops`) i klasycznych klasyfikatorów sklearn.

Klasy (zgodne z przykładem Roboflow z treści zajęć): `styrofoam`, `cotton`, `aluminium_foil`,
`brown_bread`, `corduroy`.

Segmentacja (pkt 10 PDF), GLRLM i CNN — poza tym repozytorium.

## Uruchomienie

```bash
cd lab02
./start.sh
```

Ręcznie:

```bash
uv sync
uv run python src/prepare_dataset.py   # pobiera Kaggle → original/
./split_dataset.sh                     # 56 / 12 / 13 na klasę
uv run python src/main.py              # cechy GLCM → features/
uv run python src/visualize.py
uv run python src/classify.py
uv run python src/plot_csv.py
```

## Dane

- Źródło: [ag3ntsp1d3rx/kth-tips-2](https://www.kaggle.com/datasets/ag3ntsp1d3rx/kth-tips-2) przez `kagglehub`
- Obrazy kolorowe 200×200 → konwersja do **grayscale** (luminancja) przed GLCM
- Podział: `SEED=42`, stratified per klasa w `split_dataset.sh`

## Cechy GLCM

| Parametr | Wartość |
|----------|---------|
| Poziomy szarości | 32 (kwantyzacja) |
| Odległość | 1 |
| Kąty | 0°, 45°, 90°, 135° (średnia po kątach) |
| Właściwości | contrast, dissimilarity, homogeneity, energy, correlation, ASM + **entropy** (z macierzy) |

Zestawy do klasyfikacji: **minimalne** (`energy`, `entropy`) i **pełne** (wszystkie powyższe).

## Struktura

| Ścieżka | Opis |
|---------|------|
| `src/prepare_dataset.py` | pobranie i kopiowanie 5 klas |
| `src/glcm_features.py` | ekstrakcja GLCM |
| `src/main.py` | CSV cech |
| `src/classify.py` | kNN, regresja logistyczna, drzewo, SVM, XGBoost; tuning na walidacji |
| `plots/`, `results/` | wykresy i raporty |

## Teoria (skrót do raportu)

**GLCM** liczy histogram par pikseli w odległości \(d\) wzdłuż kierunku \(\theta\). **Haralick features**
agregują macierz do skalarów (kontrast, energia, homogeniczność…). W **mahotas** odpowiednik to
`haralick()` (13 cech × 4 kierunki). **Entropia** mierzy „losowość” współwystępowań i nie jest w
`graycoprops` — liczymy ją ręcznie z znormalizowanej macierzy.

Roboflow / `inference-sdk` nie jest wymagane — te same klasy są w KTH-TIPS-2 lokalnie, bez API.
