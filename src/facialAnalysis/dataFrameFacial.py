import src.EEG.dataFrameEEG as dfe
import os
import pandas as pd
import subprocess
from feat import Detector
from collections import defaultdict
import numpy as np

def unix_to_seconds(csv_df, target_ts):
    """
        Converts a Unix timestamp to seconds relative to the start of the video.
        Takes the first timestamp as the reference point (0).
    """
    csv_df_sorted = csv_df.sort_values("Unix_Timestamp")
    start_ts = csv_df_sorted["Unix_Timestamp"].iloc[0]
    return int((target_ts - start_ts) / 1000)

def ffmpeg_cut(input_path, output_path, start_sec, duration):
    """
        Cuts a segment from a video file using FFmpeg.

        Parameters:
        - input_path (str): Path to the input video file.
        - output_path (str): Path to save the cut video segment.
        - start_sec (float): Start time in seconds from which to begin the cut.
        - duration (float): Duration of the segment to cut (in seconds).

        The function uses FFmpeg with stream copy (`-c copy`) for fast cutting without re-encoding.
        The `-y` flag overwrites the output file if it already exists.
        Both stdout and stderr are suppressed.
    """
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",  
        output_path,
        "-y"  
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def cut_ffmpeg_segments(video_dir, log_dir, output_dir):
    """
        Cuts specific video segments for each participant based on timestamp logs and saves them to individual folders.

        Parameters:
        - video_dir (str): Directory containing .avi video files and their corresponding .csv files with frame timestamps.
        - log_dir (str): Directory containing .txt log files with labeled event timestamps for each participant.
        - output_dir (str): Directory where the output video segments will be saved.

        Function Workflow:
        1. Iterates through each .avi video in the `video_dir`.
        2. For each participant, attempts to find:
        - The video file (.avi)
        - A CSV file containing frame timestamps
        - A log file containing labeled events (start/end of video, game phases, questionnaire interruptions, etc.)
        3. Defines key segments to cut:
        - Baseline video viewing
        - Three parts for the first game (before, between, and after in-game questionnaires)
        - Three parts for the second game
        4. For each segment:
        - Converts the relevant Unix timestamps to seconds using the CSV as reference.
        - Uses `ffmpeg_cut()` to extract the desired segment without re-encoding.
        - If the segment has exclusions (time intervals to remove), it cuts valid subclips and concatenates them.
        5. Saves all segments as .mp4 files in a participant-specific folder inside `output_dir`.

        Notes:
        - If any required file is missing for a participant, that participant is skipped.
        - Any errors encountered during processing are printed but do not stop the execution.
        - Temporary video clips are deleted after being concatenated.
    """
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(video_dir):
        if not file.endswith(".avi"):
            continue

        participant_id = file.replace(".avi", "")
        video_path = os.path.join(video_dir, file)
        csv_path = os.path.join(video_dir, f"{participant_id}.csv")
        log_path = os.path.join(log_dir, f"{participant_id}.txt")

        if not os.path.exists(csv_path) or not os.path.exists(log_path):
            print(f"File mancanti per {participant_id}, salto.")
            continue

        try:
            timestamps = dfe.extract_timestamps_from_log(log_path, participant_id)
            csv_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Errore con {participant_id}: {e}")
            continue

        # Segments
        segments = [
            ("baseline", timestamps["Inizio riproduzione primo video"], timestamps["Fine riproduzione primo video"], []),
            ("game1_part1", timestamps["Avvio del primo gioco"], timestamps["primo iGEQ mostrato, gioco messo in pausa"], []),
            ("game1_part2", timestamps["primo iGEQ terminato, gioco ripreso"], timestamps["secondo iGEQ mostrato, gioco messo in pausa"], []),
            ("game1_part3", timestamps["secondo iGEQ terminato, gioco ripreso"], timestamps["Chiusura del primo gioco"], []),
            ("game2_part1", timestamps["Avvio secondo gioco"], timestamps["terzo iGEQ mostrato, gioco messo in pausa"], []),
            ("game2_part2", timestamps["terzo iGEQ terminato, gioco ripreso"], timestamps["quarto iGEQ mostrato, gioco messo in pausa"], []),
            ("game2_part3", timestamps["quarto iGEQ terminato, gioco ripreso"], timestamps["Chiusura secondo gioco"], [])
        ]

        # output folder
        participant_out = os.path.join(output_dir, participant_id)
        os.makedirs(participant_out, exist_ok=True)

        for name, start_ts, end_ts, exclusions in segments:
            try:
                if not exclusions:
                    start_sec = unix_to_seconds(csv_df, start_ts)
                    end_sec = unix_to_seconds(csv_df, end_ts)
                    duration = end_sec - start_sec
                    output_path = os.path.join(participant_out, f"{name}.mp4")
                    ffmpeg_cut(video_path, output_path, start_sec, duration)
                else:
                    # cuts the right parts and joins them
                    clips = []
                    current_ts = start_ts
                    for excl_start, excl_end in exclusions:
                        if excl_start > current_ts:
                            s = unix_to_seconds(csv_df, current_ts)
                            e = unix_to_seconds(csv_df, excl_start)
                            d = e - s
                            tmp_clip = os.path.join(participant_out, f"tmp_{s}_{e}.mp4")
                            ffmpeg_cut(video_path, tmp_clip, s, d)
                            clips.append(tmp_clip)
                        current_ts = excl_end

                    if current_ts < end_ts:
                        s = unix_to_seconds(csv_df, current_ts)
                        e = unix_to_seconds(csv_df, end_ts)
                        d = e - s
                        tmp_clip = os.path.join(participant_out, f"tmp_{s}_{e}.mp4")
                        ffmpeg_cut(video_path, tmp_clip, s, d)
                        clips.append(tmp_clip)

                    concat_list_path = os.path.join(participant_out, "concat_list.txt")
                    with open(concat_list_path, "w") as f:
                        for clip_path in clips:
                            f.write(f"file '{clip_path}'\n")
                    final_path = os.path.join(participant_out, f"{name}.mp4")
                    subprocess.run([
                        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path,
                        "-c", "copy", final_path, "-y"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    # clean the temp files
                    for clip in clips:
                        os.remove(clip)
                    os.remove(concat_list_path)

            except Exception as e:
                print(f"Errore nel taglio {name} per {participant_id}: {e}")

def analyze_participant_videos(main_folder, output_folder=None): 
    """
        Analyzes all participant videos using py-feat, extracting AUs, emotions, and head poses.

        Args:
            main_folder (str): Main folder containing subfolders for each participant.
            output_folder (str, optional): Folder to save the results. If None, results are not saved to disk.

        Returns:
            dict: Results organized by participant and video name.
    """
    # initializing the py-feat detector
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose"
    )

    results = {}

    # for loop in trought the participant folders
    for participant_folder in os.listdir(main_folder):
        participant_path = os.path.join(main_folder, participant_folder)
        if not os.path.isdir(participant_path) or "_" not in participant_folder:
            continue

        print(f"Partecipante: {participant_folder}")
        results[participant_folder] = {}

        # output folder for this participant
        if output_folder:
            participant_output = os.path.join(output_folder, participant_folder)
            os.makedirs(participant_output, exist_ok=True)

        # finds the clip
        video_files = [f for f in os.listdir(participant_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        if len(video_files) != 7:
            print(f"WARNING: found {len(video_files)} video instead of 7")

        for video_file in video_files:
            video_path = os.path.join(participant_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            print(f"Analyzing: {video_file}")

            try:
                # Analyze the video with skip-frames = 15 (30fps/15 = 2 fps) 
                fex = detector.detect_video(video_path, skip_frames=15)

                # dataframes extraction
                aus_df = fex.aus
                emotions_df = fex.emotions
                poses_df = fex.poses

                results[participant_folder][video_name] = {
                    "aus": aus_df,
                    "emotions": emotions_df,
                    "poses": poses_df
                }

                # save the csv
                if output_folder:
                    video_output = os.path.join(participant_output, video_name)
                    os.makedirs(video_output, exist_ok=True)

                    aus_df.to_csv(os.path.join(video_output, "aus.csv"), index=False)
                    emotions_df.to_csv(os.path.join(video_output, "emotions.csv"), index=False)
                    poses_df.to_csv(os.path.join(video_output, "poses.csv"), index=False)

            except Exception as e:
                print(f"Error occurred during the analysis of {video_file}: {str(e)}")

    return results


def load_facial_data(root_path):
    """
        Loads and organizes facial data (AUs, emotions, poses) for each participant.

        This function assumes a directory structure where each participant has their own folder containing:
        - A "baseline" subfolder with facial data before gameplay.
        - A "game1" and "game2" subfolder, each containing:
            * "firstInterval": first 5-minute segment
            * "secondInterval": second 5-minute segment
            * "fullNoInterruption": full uninterrupted gameplay video

        For each of these subfolders, the function loads and merges facial data using `load_and_merge_facial_components`.

        Args:
            root_path (str): Path to the root folder containing participant subfolders.

        Returns:
            dict: A dictionary where each key is a participant ID, and the value is a list of DataFrames in the following order:
                [baseline, game1_firstInterval, game1_secondInterval, game1_full, game2_firstInterval, game2_secondInterval, game2_full]
    """
    data_dict = defaultdict(list)
    
    for participant in os.listdir(root_path):
        participant_path = os.path.join(root_path, participant)
        if not os.path.isdir(participant_path):
            continue

        # 1. Baseline
        baseline_path = os.path.join(participant_path, "baseline")
        baseline_df = load_and_merge_facial_components(baseline_path)
        data_dict[participant].append(baseline_df)

        # 2. Game1
        game1_path = os.path.join(participant_path, "game1")
        for interval in ["firstInterval", "secondInterval", "fullNoInterruption"]:
            interval_path = os.path.join(game1_path, interval)
            df = load_and_merge_facial_components(interval_path)
            data_dict[participant].append(df)

        # 3. Game2
        game2_path = os.path.join(participant_path, "game2")
        for interval in ["firstInterval", "secondInterval", "fullNoInterruption"]:
            interval_path = os.path.join(game2_path, interval)
            df = load_and_merge_facial_components(interval_path)
            data_dict[participant].append(df)
    
    return data_dict

def load_and_merge_facial_components(folder_path):
    """
        Loads and merges facial analysis components (AUs, emotions, poses) from a given folder.

        This function attempts to read three CSV files from the specified folder:
        - "aus.csv": contains Action Units data
        - "emotions.csv": contains emotion probabilities
        - "poses.csv": contains head pose data (e.g., pitch, yaw, roll)

        It concatenates the three DataFrames horizontally (along columns) to produce a single DataFrame
        representing the full set of facial features for each frame.

        Args:
            folder_path (str): Path to the folder containing the facial analysis CSV files.

        Returns:
            pd.DataFrame: Merged DataFrame containing AUs, emotions, and poses.
                        If any file is missing or an error occurs, returns an empty DataFrame.
    """
    try:
        aus_path = os.path.join(folder_path, "aus.csv")
        emotions_path = os.path.join(folder_path, "emotions.csv")
        poses_path = os.path.join(folder_path, "poses.csv")

        aus_df = pd.read_csv(aus_path)
        emotions_df = pd.read_csv(emotions_path)
        poses_df = pd.read_csv(poses_path)

        merged_df = pd.concat([aus_df, emotions_df, poses_df], axis=1)
        return merged_df
    except Exception as e:
        print(f"Error occurred while loading file in {folder_path}: {e}")
        return pd.DataFrame()  # Empty dataframe if gives an error

def normalize_with_baseline(data_dict):
    """
        Normalizes facial data segments for each participant by subtracting the baseline mean.

        This function assumes that for each participant, `data_dict[participant]` is a list of 7 DataFrames:
        - Index 0: Baseline segment (used as the normalization reference)
        - Index 1–6: Facial data segments from two games (3 per game)

        The function computes the mean of the baseline DataFrame and subtracts it from each of the 6 gameplay segments,
        performing frame-wise normalization across all features (AUs, emotions, poses).

        If the baseline is missing or empty, normalization is skipped for that participant.
        The original baseline is always preserved in the output.

        Args:
            data_dict (dict): Dictionary mapping participant IDs to a list of 7 DataFrames.

        Returns:
            dict: Dictionary with the same structure, where gameplay segments (1–6) are baseline-normalized.
    """
    normalized_dict = defaultdict(list)

    for participant, df_list in data_dict.items():
        baseline_df = df_list[0]
        
        if baseline_df is None or baseline_df.empty:
            print(f"Baseline dataframe empty for {participant}, skip the normalization.")
            normalized_dict[participant] = df_list
            continue
        
        baseline_mean = baseline_df.mean()

        normalized_dict[participant].append(baseline_df)

        # Normalize segment 1-6 (first interval, second, third for game1 and game2 (6 clips))
        for i in range(1, 7):
            segment_df = df_list[i]
            if segment_df is not None and not segment_df.empty:
                normalized_df = segment_df - baseline_mean
                normalized_dict[participant].append(normalized_df)
            else:
                print(f"Segment {i} empty for {participant}")
                normalized_dict[participant].append(segment_df)

    return normalized_dict

def extract_facial_features(df, label):
    """
        Extracts statistical features from a facial data segment, including emotions, AUs (Action Units), and head poses.

        This function processes a DataFrame containing frame-by-frame values for:
        - Emotion probabilities (e.g., happiness, fear)
        - Action Units (e.g., AU06, AU12)
        - Head pose angles (Pitch, Roll, Yaw)

        For each emotion, it calculates:
        - Quartiles (Q1, Q2/median, Q3)
        - Standard deviation (SD)
        - Time Dominance (TD): proportion of frames where this emotion is the strongest
        - Longest Dominance (LD): longest continuous run where this emotion is dominant, normalized by total length
        - Final value and a binary flag indicating if it was the dominant emotion in the last frame

        For each AU and head pose variable, it calculates:
        - Quartiles (Q1, Q2/median, Q3)
        - Standard deviation (SD)
        - Final value

        Args:
            df (pd.DataFrame): DataFrame containing the facial features over time (frames).
            label (str): Label to assign to the resulting feature row (used as the Series name).

        Returns:
            pd.Series: A Series containing all extracted features for the given segment.
    """
    emotion_cols = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
    au_cols = [
        "AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11", "AU12",
        "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26", "AU28", "AU43"
    ]
    pose_cols = ["Pitch", "Roll", "Yaw"]
    data = {}
    
    # Emotion features
    for col in emotion_cols:
        if col in df.columns:
            values = df[col].values
            data[f"{col}_Q1"] = np.percentile(values, 25)
            data[f"{col}_Q2"] = np.percentile(values, 50)
            data[f"{col}_Q3"] = np.percentile(values, 75)
            data[f"{col}_SD"] = np.std(values)
            data[f"{col}_TD"] = np.mean(df[col] == df[emotion_cols].max(axis=1))
            max_run = (df[col] == df[emotion_cols].max(axis=1)).astype(int)
            data[f"{col}_LD"] = max_run.groupby((max_run != max_run.shift()).cumsum()).transform('count').max() / len(max_run)
            data[f"{col}_final_val"] = values[-1]
            data[f"{col}_final_bin"] = int(col == df[emotion_cols].iloc[-1].idxmax())

    # AU features
    for col in au_cols:
        if col in df.columns:
            values = df[col].values
            data[f"{col}_Q1"] = np.percentile(values, 25)
            data[f"{col}_Q2"] = np.percentile(values, 50)
            data[f"{col}_Q3"] = np.percentile(values, 75)
            data[f"{col}_SD"] = np.std(values)
            data[f"{col}_final"] = values[-1]

    # Pose features
    for col in pose_cols:
        if col in df.columns:
            values = df[col].values
            data[f"{col}_Q1"] = np.percentile(values, 25)
            data[f"{col}_Q2"] = np.percentile(values, 50)
            data[f"{col}_Q3"] = np.percentile(values, 75)
            data[f"{col}_SD"] = np.std(values)
            data[f"{col}_final"] = values[-1]

    return pd.Series(data, name=label)

def extract_feature_summary(normalized_dict):
    """
        Extracts summary features from normalized facial data for each participant and organizes them per game.

        This function assumes that `normalized_dict` contains baseline-normalized facial data
        for each participant as a list of 7 DataFrames:
        - Index 0: Baseline (not used here)
        - Index 1–3: Game 1 segments (firstInterval, secondInterval, fullNoInterruption)
        - Index 4–6: Game 2 segments (firstInterval, secondInterval, fullNoInterruption)

        For each interval in both games, it uses `extract_facial_features` to compute a rich set of statistical
        descriptors (quartiles, standard deviation, dominance, final values) across AUs, emotions, and pose metrics.

        Args:
            normalized_dict (dict): Dictionary mapping participant IDs to a list of 7 baseline-normalized DataFrames.

        Returns:
            dict: A dictionary mapping each participant ID to a list containing two DataFrames:
                - [0]: DataFrame with extracted features for Game 1 (3 rows: one per segment)
                - [1]: DataFrame with extracted features for Game 2 (3 rows: one per segment)
    """
    feature_summary_dict = {}

    for participant, df_list in normalized_dict.items():
        # indexes:
        # 1: game1_firstInterval
        # 2: game1_secondInterval
        # 3: game1_fullNoInterruption
        # 4: game2_firstInterval
        # 5: game2_secondInterval
        # 6: game2_fullNoInterruption

        game1 = pd.DataFrame([
            extract_facial_features(df_list[1], "firstInterval"),
            extract_facial_features(df_list[2], "secondInterval"),
            extract_facial_features(df_list[3], "fullNoInterruption")
        ])

        game2 = pd.DataFrame([
            extract_facial_features(df_list[4], "firstInterval"),
            extract_facial_features(df_list[5], "secondInterval"),
            extract_facial_features(df_list[6], "fullNoInterruption")
        ])

        feature_summary_dict[participant] = [game1, game2]

    return feature_summary_dict