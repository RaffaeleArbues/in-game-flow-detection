# EEG Data Analysis Pipeline

## Overview
This pipeline processes EEG data to analyze brainwave activity during a gaming experiment. It extracts EEG frequency bands, segments the data based on timestamps, normalizes it, and computes amplitude metrics.

## Steps
### Extract EEG Bands
   - Loads EEG power data from `power_by_band.json` files for each participant.
   - Stores data in a dictionary of DataFrames.
   
   ```python
   eeg_data = create_power_by_band_dataframes(json_file_paths)
   ```

### Segment Data by Events
   - Uses `split_dataframes()` to divide EEG data into three segments:
     - **Baseline Video 1** (baseline measurement)
     - **Game 1** (excluding pauses for in-game questionnaires)
     - **Game 2** (excluding pauses for in-game questionnaires)
   - Timestamps are extracted from log files using `extract_timestamps_from_log()`.
   
   ```python
   segmented_data = split_dataframes(eeg_data, log_directory)
   ```

### Normalize EEG Data
   - Applies **Z-score normalization** using the last 30 seconds of Video 1 as a baseline.
   - Extracts EEG values and computes the mean and standard deviation for normalization.
   - Each game segment is normalized using the formula:
     
     ```python
     normalized_value = (raw_value - baseline_mean) / baseline_std
     ```
   - If the standard deviation is zero, normalization is skipped to prevent division errors.
   
   ```python
   normalized_data = normalize_eeg(segmented_data)
   ```
   - The function ensures all EEG channels (`alpha`, `beta`, `delta`, `gamma`, `theta`) are normalized separately for each participant.

### Compute EEG Amplitudes
   - Calculates **Root Mean Square (RMS)** and **Peak-to-Peak (PtP) amplitudes** for each game segment:
     - **Root Mean Square (RMS)** computes the power of EEG signals over a specific interval:
       
       ```python
       rms_value = np.sqrt(np.mean(np.square(eeg_signal), axis=0))
       ```
     - **Peak-to-Peak (PtP)** measures the difference between the maximum and minimum EEG signal value:
       
       ```python
       ptp_value = np.max(eeg_signal, axis=0) - np.min(eeg_signal, axis=0)
       ```
   
   ```python
   rms_amplitudes = compute_aggregated_rms_amplitudes(normalized_data, log_directory)
   ptp_amplitudes = compute_aggregated_ptp_amplitudes(normalized_data, log_directory)
   ```
   - Each amplitude is computed over three game intervals:
     1. **First segment** (game start to first in-game questionnaire pause)
     2. **Second segment** (after first questionnaire to second questionnaire pause)
     3. **Full segment excluding pauses** (start to end, excluding questionnaire interruptions)

## Output
normalized_dataframes: A dictionary where each key is a participant ID and the value is a dictionary containing the normalized EEG DataFrames.

aggregated_dataframes from `compute_aggregated_ptp_amplitudes()` and `compute_aggregated_rms_amplitudes()`: Dictionaries where each key is a participant ID, and the value containing the computed RMS or PtP DataFrames for each game segment.
