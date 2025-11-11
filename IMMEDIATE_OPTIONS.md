# Immediate Options While Zenodo is Down

## Quick Summary

Zenodo is currently unavailable (503 error). Here are your **immediate options**:

## Option 1: Test with Minimal Data (Recommended for Now)

You already have the test structure set up! You can:

1. **Add 2-3 short video files** to `data/S1/`
   - Can be any video format (we can convert)
   - Even 5-10 second clips work for testing
   - Record yourself saying simple phrases

2. **The alignment files are already created** in `data/alignments/S1/`
   - video1.align, video2.align, video3.align

3. **Test the pipeline**:
   ```bash
   python3 -m src.train --video_pattern "data/S1/*.mpg" --epochs 5
   ```

This will verify your code works while waiting for GRID corpus.

## Option 2: Try GitHub Download Scripts

Some GitHub repositories have GRID download scripts:
- Search: `github.com GRID corpus download`
- Example: https://gist.github.com/Zikovich/6fd12835ac4d21d8afce25ec2fe3f45e

## Option 3: Check Kaggle

1. Go to: https://www.kaggle.com/datasets
2. Search: "GRID corpus" or "lip reading GRID"
3. Many researchers share processed versions

## Option 4: Wait and Retry

Zenodo may be temporarily down. Try again in a few hours:
- https://zenodo.org/records/3625687

## What You Can Do Right Now

Since you have the test structure ready:

1. **Add videos**: Place any video files in `data/S1/`
2. **Convert to MPG** (if needed): We can create a conversion script
3. **Test training**: Run with a small number of epochs

Would you like me to:
- Create a video conversion script?
- Help you set up a minimal test dataset?
- Search for other GRID corpus mirrors?

