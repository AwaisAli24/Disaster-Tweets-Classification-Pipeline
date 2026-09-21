# Data Directory

This directory stores the raw and processed dataset files for Disaster Tweets classification.

## Expected File Formats

1. **`train.csv`**: Standard training dataset with columns:
   - `text`: Raw tweet text content.
   - `target`: Binary integer label (`1` for real-world disaster, `0` for non-disaster / everyday event).

2. **`test.csv`**: Standard test dataset with columns:
   - `text`: Raw tweet text content.
   - `target`: Binary integer label (`1` or `0`).

3. **`DisasterTweets.csv`**: Pre-loaded raw dataset.

## Preparing Data
To automatically split raw data into standardized `train.csv` and `test.csv`:

```bash
python data/prepare_data.py
```
