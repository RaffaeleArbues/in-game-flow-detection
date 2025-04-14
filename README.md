# Project Overview

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

The experimental protocol was structured as follows:

1. **Physiological baseline** (2 minutes)

2. **Gameplay – Game 1** (15 minutes)  
   - In-game GEQ at minute 5  
   - In-game GEQ at minute 10

3. **Post-game GEQ** (Game 1)

4. **Second physiological baseline** (2 minutes)

5. **Gameplay – Game 2** (15 minutes)  
   - In-game GEQ at minute 5  
   - In-game GEQ at minute 10

6. **Post-game GEQ** (Game 2)

7. **Final baseline** (relaxation phase)

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

## Project Structure

The project is organized into the following main components:

### `src/EEG/`
- `corrEEG.py`  
  → Correlation analysis between EEG features and flow scores  
- `dataFrameEEG.py`  
  → EEG data segmentation, normalization, and feature extraction  
- `preProcessing.py`  
  → Preprocessing of raw EEG signals  

### `src/facialAnalysis/`
- `corrFA.py`  
  → Correlation analysis between facial features and flow scores  
- `dataFrameFacial.py`  
  → Processing of facial data (action units, emotions, head pose)  

### `src/physiological/`
- `corrPeriferic.py`  
  → Correlation of peripheral signals (EDA, BVP) with flow scores  
- `dataFramePhysiological.py`  
  → Processing and feature extraction for peripheral biosignals  

### `src/questionnaire/`
- `dataFrameQuest.py`  
  → Parsing and structuring of

# Questionnaire Data Processing Pipeline

## Overview
This pipeline processes self-report questionnaire data collected during a gaming experiment. It loads, organizes, and transforms participants' answers to compute flow-related scores across different game sessions (familiar vs unfamiliar game).

## Steps

### Questionnaire Structure
- Defines a generic `Questionario` class representing a questionnaire.
- Specialized subclasses for each questionnaire:
  - `InGameGEQ` → In-game Game Experience Questionnaire
  - `CoreGEQ` → Core Game Experience Questionnaire
  - `PostGameGEQ` → Post-session Game Experience Questionnaire

Each subclass specifies:
- Categories (e.g., Flow, Immersion, Competence, etc.)
- Associated items (question numbers) per category.

```python
questionari = {
    "iGEQ": InGameGEQ(),
    "GEQ": CoreGEQ(),
    "Post-game": PostGameGEQ()
}
```

---

### Load Questionnaire Data
- Loads answers from CSV files for each participant.
- Automatically detects participant group:
  - Group A → First game is familiar
  - Group B → First game is unfamiliar
- Labels each questionnaire response with:
  - Participant ID
  - Questionnaire Type (iGEQ, GEQ, Post-game)
  - Game Session (Known or Unknown)

```python
df = carica_questionari(cartella_principale)
```

---

### Filtering Flow-related Data
- Filters only:
  - iGEQ and GEQ data
  - Flow-related questions only
- Splits the data into:
  - `df_noto_filtrato` → Known game responses
  - `df_ignoto_filtrato` → Unknown game responses

---

### Extract Flow Scores
For each participant:
- `df_selfreport_1` → Flow score from iGEQ before the first in-game questionnaire pause.
- `df_selfreport_2` → Flow score from iGEQ before the second in-game questionnaire pause.
- `df_selfreport_final` → Flow score from the final GEQ.

Two dictionaries are returned:
- `df_noto_dict` → Flow scores for the Known Game.
- `df_ignoto_dict` → Flow scores for the Unknown Game.

---

## Output

| Variable         | Content                                | Purpose                                                       |
|-----------------|----------------------------------------|---------------------------------------------------------------|
| `df_noto_dict`  | Flow scores for Known Game            | For correlation with EEG/EDA/BVP data during Known Game       |
| `df_ignoto_dict`| Flow scores for Unknown Game          | For correlation with EEG/EDA/BVP data during Unknown Game     |

---

# EEG Data Analysis Pipeline

## Overview
This pipeline processes EEG data to analyze brainwave activity during a gaming experiment. It extracts EEG frequency bands, segments the data based on timestamps, normalizes it, and computes amplitude metrics.

## Steps

### 1. Extract EEG Bands
   - Loads EEG power data from `power_by_band.json` files for each participant.
   - Stores data in a dictionary of DataFrames.
   
   ```python
   eeg_data = create_power_by_band_dataframes(json_file_paths)
   ```

### 2. Segment Data by Events
   - Uses `split_dataframes()` to divide EEG data into three segments:
     - **Baseline Video 1** (baseline measurement)
     - **Game 1** (excluding pauses for in-game questionnaires)
     - **Game 2** (excluding pauses for in-game questionnaires)
   - Timestamps are extracted from log files using `extract_timestamps_from_log()`.
   
   ```python
   segmented_data = split_dataframes(eeg_data, log_directory)
   ```

### 3. Normalize EEG Data
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

### 4. Compute EEG Amplitudes
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

---

## Output
| Variable                      | Content                                                                 | Purpose                                      |
|------------------------------|-------------------------------------------------------------------------|----------------------------------------------|
| normalized_dataframes         | Dictionary: participant ID → dict of normalized EEG DataFrames         | EEG data normalized via baseline Z-scoring   |
| aggregated_dataframes_rms     | Dictionary from `compute_aggregated_rms_amplitudes()`                  | RMS amplitudes per EEG band and interval     |
| aggregated_dataframes_ptp     | Dictionary from `compute_aggregated_ptp_amplitudes()`                  | PtP amplitudes per EEG band and interval     |


---

# Periferic Data Processing Pipeline (EDA & BVP)

## Overview
This pipeline processes periferic signals (EDA and BVP) recorded from Empatica EmbracePlus. It segments data into experimental phases, applies signal filtering, extracts features, and normalizes signals based on baseline activity.

## Steps

### 1. Data Loading and Segmentation
Exact same procedure as EEG signals

```python
segmented_dataframes_eda, segmented_dataframes_bvp = split_dataframes(data_dir, log_dir)
```

### 2. Signal Filtering
- Applies Butterworth filters:
  - Low-pass filter on EDA (cutoff=1Hz, fs=4Hz)
  - Band-pass filter on BVP (1-8Hz, fs=64Hz)

```python
butter_lowpass_filter(data, cutoff=1, fs=4)
butter_bandpass_filter(data, lowcut=1, highcut=8, fs=64)
```

### 3. EDA Signal Decomposition
- Decomposes EDA into tonic and phasic components using `cvxEDA` from `neurokit2`.

```python
eda_signals = nk.eda_phasic(df["eda"], sampling_rate=4)
```

### 4. Z-Score Normalization
- Normalizes Game 1 and Game 2 signals using the last 30 seconds of Video 1 baseline.

```python
normalized_dataframes_eda, normalized_dataframes_bvp = normalize_physio_dataframes(segmented_dataframes_eda, segmented_dataframes_bvp)
```

### 5. Heart Rate Calculation
- Derives HR and IBI from BVP signals by detecting local minima (systolic peaks).
- Interpolates HR to create a continuous signal.

```python
hr_dataframes = calculate_heart_rate(normalized_dataframes_bvp)
```

### 6. Feature Extraction

#### EDA Features
- Extracts metrics from:
  - EDA
  - Tonic Component
  - Phasic Component

Metrics include: **Min**, **Max**, **Mean**, **Delta**, **Decrease Rate**, **Decrease Time**, **Number of Peaks**.

```python
eda_metrics = extract_eda_metrics(normalized_dataframes_eda, log_dir)
```

#### BVP & HR Features
- Extracts metrics from:
  - BVP Signal
  - HR
  - HRV (SDNN, RMSSD)

```python
bvp_metrics = extract_bvp_metrics(hr_dataframes, log_dir)
```

---

## Output

| Variable | Content | Purpose |
|-----------|--------------------------------------------|-----------------------------------------------|
| segmented_dataframes_eda | Filtered EDA DataFrames (Baseline, Game1, Game2) | Raw filtered data |
| segmented_dataframes_bvp | Filtered BVP DataFrames (Baseline, Game1, Game2) | Raw filtered data |
| normalized_dataframes_eda | Normalized EDA DataFrames for Game1 and Game2 | For correlation with questionnaires |
| normalized_dataframes_bvp | Normalized BVP DataFrames for Game1 and Game2 | For correlation with questionnaires |
| hr_dataframes | DataFrames with HR and IBI values | For HR and HRV analysis |
| eda_metrics | Extracted EDA metrics per interval | Statistical and peak analysis |
| bvp_metrics | Extracted BVP and HR metrics per interval | Time-domain analysis and HRV metrics |

---

# Facial Video Data Processing Pipeline

## Overview
This pipeline processes facial video recordings from participants during the gaming experiment. It cuts gameplay videos into meaningful segments, extracts facial features (Action Units, Emotions, and Head Pose) using Py-Feat, normalizes these features and computes descriptive statistics for subsequent analysis.

## Steps

### 1. Video Segmentation
- Splits original participant videos based on experimental events extracted from log files.
- Extracted segments include:
  - Baseline Video
  - Game 1 → 3 segments
  - Game 2 → 3 segments

```python
cut_ffmpeg_segments(video_dir, log_dir, output_dir)
```

### 2. Facial Feature Extraction
- Uses Py-Feat's Detector to extract:
  - Action Units (AUs)
  - Emotion Probabilities
  - Head Poses

```python
results = analyze_participant_videos(main_folder, output_folder)
```

### 3. Load and Merge Facial Data
- Loads AUs, Emotions, and Poses into a single DataFrame per segment.

```python
data_dict = load_facial_data(root_path)
```

### 4. Baseline Normalization
- Normalizes gameplay segments (Game 1 & Game 2) using the mean of the baseline segment.

```python
normalized_dict = normalize_with_baseline(data_dict)
```

### 5. Feature Extraction

#### Facial Features
- Extracts descriptive statistics from each segment:
  - Quartiles (Q1, Q2, Q3)
  - Standard Deviation
  - Emotion Dominance Metrics
  - Final Value Metrics

```python
feature_summary_dict = extract_feature_summary(normalized_dict)
```

---

## Output

| Variable | Content | Purpose |
|------------|---------------------------------------------|----------------------------------------------|
| cut_videos | Segmented videos per participant | To analyze specific experiment phases |
| results | Raw extracted features (AUs, Emotions, Pose) | Frame-by-frame data |
| data_dict | Loaded and merged DataFrames per segment | Unified facial data for each segment |
| normalized_dict | Baseline-normalized facial data | To remove individual variability |
| feature_summary_dict | Descriptive metrics per segment | For statistical correlation with other measures |

---

# Flow Analysis Pipeline (EEG, Facial, Periferic)

## Overview
This pipeline processes and integrates these three different data sources collected during the experiment to analyze the relationship between objective signals and subjective Flow experience:

- EEG data (brainwaves)
- Facial expression data (AUs, emotions, head poses)
- Physiological signals (EDA and BVP)

The final goal is to correlate these measurements with self-reported Flow scores from standardized questionnaires, using statistical modeling (Linear Mixed Models) to evaluate which features explain perceived Flow during gameplay.

---

## Pipeline Structure

### 1. Data Integration & Feature Extraction
Each pipeline processes its specific raw data to extract features per participant, game, and interval:

| Pipeline | Input Data | Extracted Features |
|----------|-------------|-------------------|
| EEG | EEG band amplitudes (alpha, beta, delta, gamma, theta) across 8 channels | RMS / PtP amplitudes per band and channel |
| Facial | Video analysis (AUs, emotions, poses) with py-feat | Statistical descriptors: quartiles, std, dominance, final value |
| Physiological | EDA & BVP signals | EDA metrics (min, max, mean, peaks, tonic/phasic), HR metrics, HRV (SDNN, RMSSD) |

Each segment is aligned with experimental events (baseline, gameplay intervals) and matched with corresponding Flow questionnaire scores.

---

### 2. Flow Score Handling
- Flow scores are extracted from iGEQ and GEQ questionnaires across 3 intervals.
- For each participant, Flow scores are normalized using Z-score normalization to control for individual differences.

---

### 3. Final Dataset Construction
Each pipeline produces a final dataframe where each row corresponds to:
- One participant
- One game type (Familiar / Unfamiliar)
- One gameplay interval
- The Flow score (raw and normalized)
- The extracted features from the respective data source

Final outputs:
| Pipeline | Output CSV |
|----------|------------|
| EEG | df_flow_eeg_data.csv |
| Facial | df_flow_face_data.csv |
| Physiological | df_flow_physiological_data.csv |

---

### 4. Statistical Analysis (R Mixed Models)
- **Linear Mixed Models (LMM)** are used to assess the contribution of extracted features in explaining normalized Flow scores.
- Analysis is performed separately for each data type.
- Deviance explained, coefficients, R² and power simulations (powerSim) are computed.

Results are automatically saved in dedicated text files for each analysis:
- EEG: full_model_results.txt + one result file per EEG band
- Facial: au_model_results.txt, emotion_model_results.txt, pose_model_results.txt
- Physiological: physio_model_results.txt

---