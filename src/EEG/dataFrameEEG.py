import pandas as pd
import os
import json
import numpy as np
import ast 

def create_power_by_band_dataframes(json_file_paths):
    """
        Creates a separate DataFrame for each participant from the power_by_band.json files,
        keeping the timestamp in its original format.

        Parameters:
            json_file_paths (list): List of paths to the JSON files.

        Returns:
            dict: A dictionary with participant IDs as keys and the corresponding DataFrames as values.
    """
    dataframes = {}

    for file_path in json_file_paths:
        # Check if it's actually power_by_band.json
        if not file_path.endswith("power_by_band.json"):
            continue  # skip if it's not power_by_band.json

        # grab the participant ID from the path
        participant_id = os.path.basename(os.path.dirname(file_path))
        print(f"working on: {participant_id}")

        # load the .json
        with open(file_path, 'r') as f:
            power_data = json.load(f)

        # extract signal and timestamp
        rows = []
        for entry in power_data:
            data = entry["data"]
            timestamp = entry["timestamp"] 

            row = {"ID": participant_id, "Timestamp": timestamp}
            row.update(data)  # add alpha, beta, delta, gamma, theta as a column

            rows.append(row)

        # Create dataframe for each participant
        df = pd.DataFrame(rows)
        dataframes[participant_id] = df

    return dataframes


def extract_timestamps_from_log(log_file_path, participant_id):
    """
        Extracts the relevant timestamps from a log file (previously recorded during the tests).

        Parameters:
            log_file_path (str): Path to the log file.
            participant_id (str): Participant ID.

        Returns:
            dict: Dictionary with the extracted timestamps, or None if the file does not exist.
    """
    if not os.path.exists(log_file_path):
        print(f"log file not found for {participant_id}")
        return None

    timestamps = {}
    with open(log_file_path, 'r') as log_file:
        log_lines = log_file.readlines()

    for line in log_lines:
        if "Timestamp UNIX:" in line:
            parts = line.strip().split(" - ")
            if len(parts) < 3:
                print(f"Format not valid for {participant_id}: {line.strip()}")
                continue  # skip the row

            event = parts[1]
            timestamp_parts = parts[2].split(": ")
            if len(timestamp_parts) < 2:
                print(f"Missing Timestamp in the log file for {participant_id}: {line.strip()}")
                continue  # skip the row

            timestamp = int(timestamp_parts[1]) * 1000  # timestamp converted in milliseconds
            timestamps[event] = timestamp

    return timestamps

def split_dataframes(dataframes, log_dir):
    """
        Splits the participants' DataFrames into three segments based on timestamps from the log files.

        Parameters:
            dataframes (dict): Dictionary with participant IDs as keys and the corresponding DataFrames as values.
            log_dir (str): Path to the directory containing the log files.

        Returns:
            dict: Dictionary with participant IDs as keys and a list of three DataFrames corresponding to the segments.
    """
    segmented_dataframes = {}

    for participant_id, df in dataframes.items():
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")

        # This line extracts timestamps (for each participant) corresponding to relevant events for data segmentation from the file log.txt
        timestamps = extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue

        # Debug print
        print(f"Timestamp for {participant_id}: {timestamps}")

        # Retrive the useful timestamps
        try:
            ts_video1_start = timestamps["Inizio riproduzione primo video"]
            ts_video1_end = timestamps["Fine riproduzione primo video"]
            ts_game1_start = timestamps["Avvio del primo gioco"]
            ts_game1_end = timestamps["Chiusura del primo gioco"]
            ts_video2_end = timestamps["Avvio secondo gioco"]
            ts_game2_end = timestamps["Chiusura secondo gioco"]
         
        except KeyError as e:
            print(f"Missing Timestamp in the log for {participant_id}: {e}")
            continue
        
        # cutting the df (associated at this key) for this participant (key) in three intervals (baseline, 1st gameplay, 2nd gameplay)
        tolerance = 1000  # 1000 ms tolerance (1 second)
        df_video1 = df[(df["Timestamp"] >= ts_video1_start - tolerance) & (df["Timestamp"] <= ts_video1_end + tolerance)]
        df_game1 = df[(df["Timestamp"] >= ts_game1_start - tolerance) & (df["Timestamp"] <= ts_game1_end + tolerance)]
        df_game2 = df[(df["Timestamp"] >= ts_video2_end - tolerance) & (df["Timestamp"] <= ts_game2_end + tolerance)]

        # Creating the dictionary: participant_id is the key, for each participant_id owns a list of dataframes
        segmented_dataframes[participant_id] = [df_video1, df_game1, df_game2]
    
    return segmented_dataframes

def normalize_eeg(segmented_dataframes):
    """
        Normalizes df_game1 and df_game2 for each participant using Z-score normalization
        computed on the last 30 seconds of df_video1.

        Parameters:
        segmented_dataframes (dict): Dictionary containing each participant's DataFrames in the format: {participant_id: [df_video1, df_game1, df_game2]}

        Returns:
        dict: Dictionary with normalized df_game1 and df_game2 DataFrames for each participant.
    """
    normalized_dataframes = {}

    for participant_id, dfs in segmented_dataframes.items():
        df_video1, df_game1, df_game2 = dfs

        # Create a copy for in-place changes
        df_video1 = df_video1.copy()

        # check if the timestamp is numeric
        df_video1["Timestamp"] = pd.to_numeric(df_video1["Timestamp"], errors="coerce") / 1000

        # takes the max timestamp
        max_time = df_video1["Timestamp"].max()

        # Extract the last 30s, cutting the first minute and half
        baseline_df = df_video1[df_video1["Timestamp"] >= (max_time - 30)].copy()
    
        # Debugging
        if baseline_df.empty:
            print(f"Baseline is empty for {participant_id}. I'll skip the normalization.")
            continue
        else:
            print(f"Baseline found for {participant_id}, it contains {len(baseline_df)} rows.")

        # EEG columns
        eeg_columns = ['alpha', 'beta', 'delta', 'gamma', 'theta']

        for col in eeg_columns:
            baseline_df[col] = baseline_df[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))

        # Calculating the mean and standard dev from baseline for each EEG channel (8 values for band)
        baseline_means = {col: np.mean(np.vstack(baseline_df[col]), axis=0) for col in eeg_columns}
        baseline_stds = {col: np.std(np.vstack(baseline_df[col]), axis=0) for col in eeg_columns}

        # function for normalizing a df with z-score normalization
        def normalize_df(df):
            df_norm = df.copy()

            df_norm["Timestamp"] = pd.to_numeric(df_norm["Timestamp"], errors="coerce") / 1000.0

            for col in eeg_columns:
                df_norm[col] = df_norm[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))

            # apply Z-score normalization "element-wise" (for each 8 values per channel)
            for col in eeg_columns:
                df_norm[col] = df_norm[col].apply(lambda x: (x - baseline_means[col]) / baseline_stds[col] if np.any(baseline_stds[col] != 0) else x)

            return df_norm

        # Normalization (for the 2 segment concerning the gameplay data)
        df_game1_norm = normalize_df(df_game1)
        df_game2_norm = normalize_df(df_game2)
        
        # Creates the normalized dictionary 
        normalized_dataframes[participant_id] = [df_game1_norm, df_game2_norm]

    return normalized_dataframes

def compute_aggregated_rms_amplitudes(normalized_segmented_dataframes, log_dir):
    """
        Calculates the aggregated Root Mean Square (RMS) Amplitude for each EEG band and channel,
        returning three rows per game (six rows in total per participant).

        Parameters:
        normalized_segmented_dataframes (dict): Dictionary containing the normalized DataFrames for each participant.
        log_dir (str): Path to the log folder. Timestamps in the logs are in **milliseconds**.

        Returns:
        dict: Dictionary with the aggregated RMS DataFrames for each participant.
    """
    aggregated_dataframes = {}
    eeg_columns = ['alpha', 'beta', 'delta', 'gamma', 'theta']

    for participant_id, dfs in normalized_segmented_dataframes.items():
        df_game1_norm, df_game2_norm = dfs
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")

        # Extracting timestamp from log.txt tests file (for each participant) 
        timestamps = extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue

        timestamps = {key: value / 1000.0 for key, value in timestamps.items()}

        # Calculating the RMS amplitude mesure for each interval for each participant, skipping the "skip_start" seconds
        def compute_rms(df, start, end, exclude_intervals=None, skip_start=1.0):
            df_interval = df[(df["Timestamp"] >= start + skip_start) & (df["Timestamp"] <= end)]
            
            if exclude_intervals:
                for exc_start, exc_end in exclude_intervals:
                    df_interval = df_interval[(df_interval["Timestamp"] < exc_start) | (df_interval["Timestamp"] > exc_end)]
            
            if df_interval.empty:
                return {band: [np.nan] * 8 for band in eeg_columns}  # If empty, returns NaN

            rms_values = {}
            for band in eeg_columns:
                values = np.array(df_interval[band].to_list())  # Converts directly to NumPy array
                rms_values[band] = np.sqrt(np.mean(np.square(values), axis=0))  # RMS measure for each channel

            return rms_values

        # defining the exact intervals in which the rms will be applied to the data and a single amplitude value will be extracted (for both game1 and game2)
        game1_intervals = [
            (timestamps["Avvio del primo gioco"], timestamps["primo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["primo iGEQ terminato, gioco ripreso"], timestamps["secondo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio del primo gioco"], timestamps["Chiusura del primo gioco"], [
                (timestamps["primo iGEQ mostrato, gioco messo in pausa"], timestamps["primo iGEQ terminato, gioco ripreso"]),
                (timestamps["secondo iGEQ mostrato, gioco messo in pausa"], timestamps["secondo iGEQ terminato, gioco ripreso"])
            ])
        ]
        game2_intervals = [
            (timestamps["Avvio secondo gioco"], timestamps["terzo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["terzo iGEQ terminato, gioco ripreso"], timestamps["quarto iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio secondo gioco"], timestamps["Chiusura secondo gioco"], [
                (timestamps["terzo iGEQ mostrato, gioco messo in pausa"], timestamps["terzo iGEQ terminato, gioco ripreso"]),
                (timestamps["quarto iGEQ mostrato, gioco messo in pausa"], timestamps["quarto iGEQ terminato, gioco ripreso"])
            ])
        ]

        game1_rms = [compute_rms(df_game1_norm, *interval[:2], exclude_intervals=interval[2] if len(interval) > 2 else None) for interval in game1_intervals]
        game2_rms = [compute_rms(df_game2_norm, *interval[:2], exclude_intervals=interval[2] if len(interval) > 2 else None) for interval in game2_intervals]

        # Creating dataframes
        df_game1_rms = pd.DataFrame(game1_rms)
        df_game1_rms.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        df_game2_rms = pd.DataFrame(game2_rms)
        df_game2_rms.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        # Creating dictionary
        aggregated_dataframes[participant_id] = {
            "game1_rms": df_game1_rms,
            "game2_rms": df_game2_rms
        }

    return aggregated_dataframes

def compute_aggregated_ptp_amplitudes(normalized_segmented_dataframes, log_dir):
    """
        Calculates the aggregated Peak-to-Peak (PtP) Amplitude for each EEG band and channel,
        returning three rows per game (six rows in total per participant).

        Parameters:
        normalized_segmented_dataframes (dict): Dictionary containing the normalized DataFrames for each participant.
                                                Timestamps in the DataFrames are in **seconds**.
        log_dir (str): Path to the log folder. Timestamps in the logs are in **milliseconds**.

        Returns:
        dict: Dictionary with the aggregated PtP DataFrames for each participant.
    """
    aggregated_dataframes = {}
    eeg_columns = ['alpha', 'beta', 'delta', 'gamma', 'theta']

    for participant_id, dfs in normalized_segmented_dataframes.items():
        df_game1_norm, df_game2_norm = dfs
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")

        # Extracting timestamp from log.txt tests file (for each participant) 
        timestamps = extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue

        timestamps = {key: value / 1000.0 for key, value in timestamps.items()}

        # # Calculating the PTP amplitude measure for each interval for each participant, skipping the "skip_start" seconds
        def compute_ptp(df, start, end, exclude_intervals=None, skip_start=1.0):
            df_interval = df[(df["Timestamp"] >= start + skip_start) & (df["Timestamp"] <= end)]
        
            if exclude_intervals:
                for exc_start, exc_end in exclude_intervals:
                    df_interval = df_interval[(df_interval["Timestamp"] < exc_start) | (df_interval["Timestamp"] > exc_end)]
            
            if df_interval.empty:
                return {band: [np.nan] * 8 for band in eeg_columns}  # If empty, returns NaN

            ptp_values = {}
            for band in eeg_columns:
                values = np.array(df_interval[band].to_list())  # Converts directly to NumPy array
                ptp_values[band] = np.max(values, axis=0) - np.min(values, axis=0)  # Peak-to-Peak measure for each channel

            return ptp_values

        # defining the exact intervals in which the ptp will be applied to the data and a single amplitude value will be extracted (for both game1 and game2)
        game1_intervals = [
            (timestamps["Avvio del primo gioco"], timestamps["primo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["primo iGEQ terminato, gioco ripreso"], timestamps["secondo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio del primo gioco"], timestamps["Chiusura del primo gioco"], [
                (timestamps["primo iGEQ mostrato, gioco messo in pausa"], timestamps["primo iGEQ terminato, gioco ripreso"]),
                (timestamps["secondo iGEQ mostrato, gioco messo in pausa"], timestamps["secondo iGEQ terminato, gioco ripreso"])
            ])
        ]
        game2_intervals = [
            (timestamps["Avvio secondo gioco"], timestamps["terzo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["terzo iGEQ terminato, gioco ripreso"], timestamps["quarto iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio secondo gioco"], timestamps["Chiusura secondo gioco"], [
                (timestamps["terzo iGEQ mostrato, gioco messo in pausa"], timestamps["terzo iGEQ terminato, gioco ripreso"]),
                (timestamps["quarto iGEQ mostrato, gioco messo in pausa"], timestamps["quarto iGEQ terminato, gioco ripreso"])
            ])
        ]

        game1_ptp = [compute_ptp(df_game1_norm, *interval[:2], exclude_intervals=interval[2] if len(interval) > 2 else None) for interval in game1_intervals]
        game2_ptp = [compute_ptp(df_game2_norm, *interval[:2], exclude_intervals=interval[2] if len(interval) > 2 else None) for interval in game2_intervals]

        # Creating dataframes
        df_game1_ptp = pd.DataFrame(game1_ptp)
        df_game1_ptp.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        df_game2_ptp = pd.DataFrame(game2_ptp)
        df_game2_ptp.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        # Creating Dictionary
        aggregated_dataframes[participant_id] = {
            "game1_ptp": df_game1_ptp,
            "game2_ptp": df_game2_ptp
        }

    return aggregated_dataframes
