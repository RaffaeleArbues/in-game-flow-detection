## Project Overview

This project is an exploratory work aimed at establishing an experimental protocol for measuring the state of **Flow** in video games.  
The concept of flow was introduced by **Mihály Csíkszentmihályi**, who defined it as a psychological state characterized by deep involvement in an activity, during which a person experiences a high level of concentration and intrinsic motivation.  
The context of video games is particularly suitable for measuring this condition, as it is standardized, repeatable, and allows for a rigorous analysis of the phenomenon.

---

## Experimental Protocol

Participants were asked to play two different games:

- One **familiar game**, chosen by each participant during recruitment.
- One **unfamiliar game**, developed by students and unknown to the participant.

Throughout the sessions, participants completed the **Game Experience Questionnaire (GEQ)** to self-report their experience of flow. The GEQ is a widely used tool in the gaming research field to assess subjective experience during gameplay.

---

## Study Pipeline

The experimental protocol followed this structure:

+-------------------------------------------+
|         Physiological baseline            |
|              (2 minutes)                  |
+-------------------+-----------------------+
                    |
                    v
+-------------------------------------------+
|           Gameplay - Game 1               |
|             (15 minutes)                  |
|                                           |
|   → In-game GEQ at minute 5               |
|   → In-game GEQ at minute 10              |
+-------------------+-----------------------+
                    |
                    v
+-------------------------------------------+
|             Post-game GEQ                 |
+-------------------+-----------------------+
                    |
                    v
+-------------------------------------------+
|       Second physiological baseline       |
|                (2 minutes)                |
+-------------------+-----------------------+
                    |
                    v
+-------------------------------------------+
|           Gameplay - Game 2               |
|             (15 minutes)                  |
|                                           |
|   → In-game GEQ at minute 5               |
|   → In-game GEQ at minute 10              |
+-------------------+-----------------------+
                    |
                    v
+-------------------------------------------+
|             Post-game GEQ                 |
+-------------------+-----------------------+
                    |
                    v
+-------------------------------------------+
|     Final baseline (relaxation phase)     |
+-------------------+-----------------------+
                    |
                    v
                 [ End ]

> Note: Game 1 and 2 is either the familiar or unfamiliar game depending on participant group. The order is counterbalanced.

## Data Collection and Analysis

During the experimental protocol, participants were equipped with:

- A **Neurosity Crown** headset for recording brain biosignals (EEG).
- An **Empatica EmbracePlus** wristband for acquiring peripheral biosignals such as Electrodermal Activity (EDA) and Blood Volume Pulse (BVP).
- A **webcam** for recording facial expressions.

All biosignals were preprocessed according to their specific characteristics, including:

- Adapting to the output file structure
- Noise reduction and signal cleaning
- Personalized normalization per participant
- Feature extraction on specific gameplay segments

After preprocessing, each type of signal was analyzed statistically using a **linear mixed model**, incorporating data extracted from the self-report questionnaires.  
This analysis aimed to identify potential correlations between the collected biosignals and the subjective experience reported by participants.

> Note: The **Game Experience Questionnaire (GEQ)** includes specific items categorized under the "Flow" dimension, which were used as reference points in the analysis.

## Project Structure (Visual Diagram)

src/
├── EEG/
│   ├── corrEEG.py
│   ├── dataFrameEEG.py
│   └── preProcessing.py
│       ─> EEG preprocessing, segmentation, feature extraction, and correlation analysis
│
├── facialAnalysis/
│   ├── corrFA.py
│   └── dataFrameFacial.py
│       ─> Processing and correlating facial features (action units, emotions, head pose)
│
├── physiological/
│   ├── corrPeriferic.py
│   └── dataFramePhysiological.py
│       ─> Processing and correlating peripheral signals (EDA, BVP)
│
├── questionnaire/
│   └── dataFrameQuest.py
│       ─> Parsing and organizing self-report questionnaire data (GEQ, iGEQ)

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
