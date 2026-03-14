# 🔄 How to Swap the VisionSnare Model

This guide explains how to replace the current model checkpoint with a new one — whether it's a retrained version or an entirely new architecture iteration.

---

## Quick Swap (Same Architecture, New Weights)

If you retrained VisionSnare with the same architecture but better data or more epochs:

### 1. Drop the new checkpoint

Copy your new `.pth` file into the `checkpoints/` folder:

```
checkpoints/
├── model_best.pth          ← current (keep as backup)
└── model_v2_best.pth       ← your new checkpoint
```

### 2. Edit `model_config.json`

Change **only** the `checkpoint` path:

```json
{
    "model_name": "VisionSnare",
    "checkpoint": "checkpoints/model_v2_best.pth",
    "lstm_hidden_dim": 512,
    "sequence_length": 12,
    "image_size": [256, 256],
    "motion_threshold": 0.7
}
```

### 3. Restart the backend

```powershell
# Kill old server (Ctrl+C), then:
cd d:\Fyp_Project\GAN_DeepfakeVideos_Detection_Model
python -m uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The startup log will confirm the new checkpoint path:
```
[VisionSnare API] Checkpoint  : ...\checkpoints\model_v2_best.pth
[VisionSnare API] Model loaded successfully ✓
```

**That's it — no code changes needed!**

---

## Full Swap (Different Hyperparameters)

If you changed the LSTM hidden size, sequence length, or image resolution during retraining:

### 1. Match `model_config.json` to your training config

| Config Key        | Must match                                  | Where to check          |
|-------------------|---------------------------------------------|-------------------------|
| `lstm_hidden_dim` | `LSTM_HIDDEN_DIM` in `training/train.py`    | Your training script    |
| `sequence_length` | `SEQUENCE_LENGTH` in `training/train.py`    | Your training script    |
| `image_size`      | Resolution used during preprocessing        | Your dataset pipeline   |

Example — if you retrained with 16-frame sequences and LSTM dim 256:

```json
{
    "model_name": "VisionSnare v2",
    "checkpoint": "checkpoints/model_v2_best.pth",
    "lstm_hidden_dim": 256,
    "sequence_length": 16,
    "image_size": [256, 256],
    "motion_threshold": 0.7
}
```

### 2. Update frontend text (optional but recommended)

If `sequence_length` changed, update these files to keep the UI accurate:

- `frontend/visionsnare/src/pages/HowItWorks.jsx` — Step 4 tech pills
- `frontend/visionsnare/src/pages/About.jsx` — Pipeline array and architecture cards

Search for the old number (e.g., `12-frame`) and replace with the new one.

---

## Architecture Swap (New Model Class)

If you built a completely different model (e.g., replaced LSTM with Transformer):

### 1. Add new model file

```
models/
├── visionsnare.py       ← current
└── visionsnare_v2.py    ← new architecture
```

### 2. Update `api/app.py` import

Change line ~34:

```python
# Before
from models.visionsnare import VisionSnare

# After
from models.visionsnare_v2 import VisionSnareV2 as VisionSnare
```

### 3. Update `model_config.json` with new hyperparameters

### 4. Restart the backend

---

## Checkpoint File Format

The API expects the checkpoint `.pth` file to contain a dict with at least:

```python
{
    "model_state_dict": model.state_dict(),
    # optional:
    "optimizer_state_dict": ...,
    "epoch": ...,
    "best_val_acc": ...,
}
```

This matches the format used by `training/train.py`. If your new checkpoint uses a different key (e.g., just `state_dict`), update line ~74 in `api/app.py`:

```python
# Change this:
_model.load_state_dict(_ckpt["model_state_dict"])

# To this:
_model.load_state_dict(_ckpt["state_dict"])
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `KeyError: 'model_state_dict'` | Your checkpoint uses a different key — see section above |
| `RuntimeError: size mismatch` | `lstm_hidden_dim` or architecture doesn't match the checkpoint |
| `FileNotFoundError` | Check the `checkpoint` path in `model_config.json` is relative to project root |
| Model always says "Real" or "Fake" | The checkpoint may not be fully trained — check `best_val_acc` in the `.pth` file |
