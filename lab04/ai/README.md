# Lab04 AI — segmentacja BDD100K

Workflow jak w Computational-Linguistics/lab03: `./shell.sh`, `tb`, `jnb`, `./start.sh`.

## Pierwszy raz

1. Edytuj [`_config_/config.env`](_config_/config.env) (porty Jupyter/TensorBoard).
2. `docker compose --env-file _config_/config.env build`
3. `./scripts/drun sync`
4. `./scripts/drun verify` — CUDA w kontenerze (powinny być widoczne 2 GPU)

## Dane

- Obrazy/etykiety: `../data` (mount RO w kontenerze jako `/workspace/data/raw`).
- Maski: `data_derived/masks/{train,val,test}/*.png`, etykiety YOLO: `data_derived/yolo/<split>/labels/`.
- `main.py` sam generuje brakujące maski w liczbie wynikającej z aktywnych presetów.
  Ręcznie: `python src/data/prepare.py [--split val] [--limit N]` (`--limit` = obrazów na split).

## Trening

Jeden kod dla smoke i FULL — różnią się tylko aktywnymi wpisami presetów:

- `src/finetune_presets.py` → `FINETUNE_ALL` → `saved_models/finetuned/<arch>/<run>/`
- `src/training_presets.py` → `SCRATCH_ALL` → `saved_models/pretrained/from_scratch/<arch>/<run>/`
- Ewaluacja i porównanie: `model_evaluations/<run>___main___<data>/{finetuned,from_scratch}/`

```bash
./normal_process_start.sh       # smoke / dry run (aktywne wpisy SMOKE, <= 15 min)
./background_process_start.sh   # FULL w tle (odkomentuj wpisy FULL, zakomentuj SMOKE)
```
