# Training Notes

## Training Time Estimates

Based on the test run:
- **Per epoch**: ~104 seconds (~1.7 minutes)
- **10 epochs**: ~17 minutes
- **100 epochs**: ~2.8 hours

## Current Status

**Training is working correctly!**
- Loss decreased from 258 → 6.5 in 2 epochs
- Model weights are being saved
- Validation loss is decreasing

## Tips for Training

### Option 1: Run in Background
```bash
nohup python3 -m src.train --epochs 10 > training.log 2>&1 &
tail -f training.log  # Monitor progress
```

### Option 2: Run Fewer Epochs First
```bash
# Test with 5 epochs first (~8 minutes)
python3 -m src.train --epochs 5
```

### Option 3: Monitor Progress
The training shows progress bars and loss values. Each epoch takes ~2 minutes.

## What to Expect

- **Early epochs**: Loss decreases rapidly (258 → 82 → 6.5)
- **Later epochs**: Loss decreases more slowly
- **Predictions**: May be empty initially, improve with more training
- **Weights**: Saved after each epoch to `models/weights_epoch_XX.h5`

## Stopping Training

If you need to stop:
- Press `Ctrl+C` in the terminal
- The latest weights will be saved

## Resuming Training

You can resume from saved weights by modifying `train.py` to load weights, or just continue training - the model will keep improving.

