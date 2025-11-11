# Alternative Ways to Get GRID Corpus

The Zenodo site is currently unavailable (503 error). Here are alternative options:

## Option 1: Wait and Retry Zenodo

The Zenodo site may be temporarily down. Try again later:
- https://zenodo.org/records/3625687
- Sometimes these sites have maintenance windows

## Option 2: Search Academic Repositories

### Kaggle
- Search for "GRID corpus" or "lip reading GRID"
- URL: https://www.kaggle.com/datasets
- Many researchers upload processed versions

### GitHub
- Search GitHub for GRID dataset repositories
- Many projects include download scripts
- Search: `site:github.com GRID corpus download`

### Academic Papers
- Papers with Code: https://paperswithcode.com/
- Search for lip-reading papers that mention GRID
- Authors often provide dataset links in their repositories

## Option 3: Contact Dataset Maintainers

### University of Sheffield
- The GRID corpus was originally from University of Sheffield
- Contact: Speech and Hearing Research Group
- They may have alternative download links

## Option 4: Use Alternative Datasets

If GRID is unavailable, consider these alternatives:

### LRW (Lip Reading in the Wild)
- URL: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/
- Large-scale dataset (500 words)
- May require conversion to GRID format

### LRS2/LRS3
- URL: https://www.robots.ox.ac.uk/~vgg/data/lrs/
- BBC dataset
- Requires research access request

## Option 5: Create Minimal Test Dataset

For immediate testing, you can create a small dataset:

1. Record short videos of yourself speaking
2. Create alignment files manually
3. Use the existing test structure we created

Run: `python3 create_test_data.py` (already done)

Then add your own video files to `data/S1/` and corresponding `.align` files.

## Option 6: Check Archive Sites

- Internet Archive: https://archive.org/
- Search for "GRID corpus" or "GRID dataset"
- Some datasets are archived there

## Quick Test Setup

If you just want to test the code works:

1. Use the test structure we already created (`data/S1/`, `data/alignments/S1/`)
2. Add 2-3 short video files (any format, we can convert)
3. Create simple alignment files
4. Test the training pipeline

This will verify everything works while you wait for GRID corpus access.

