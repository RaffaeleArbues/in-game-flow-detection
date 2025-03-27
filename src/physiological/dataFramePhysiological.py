import src.EEG.dataFrameEEG as dfe
import os
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import neurokit2 as nk
import numpy as np

def split_dataframes(data_dir, log_dir):
    """
        Splits participant DataFrames into segments based on log file timestamps and applies Butterworth filters.
        (these datas are csv generated from .avro files (EDA, BVP), recorded by Empatica EmbracePlus device)

        Parameters:
            data_dir (str): Path to the folder containing participant subfolders with CSV files.
            log_dir (str): Path to the folder containing the log files.

        Returns:
            dict: Dictionary with participant IDs as keys and six filtered DataFrames as values.
    """
    segmented_dataframes_eda = {}
    segmented_dataframes_bvp = {}
    
    for participant_id in os.listdir(data_dir):
        participant_path = os.path.join(data_dir, participant_id)
        if not os.path.isdir(participant_path):
            continue  # skip if it's a directory
        
        eda_file_path = os.path.join(participant_path, f"{participant_id}_eda.csv")
        bvp_file_path = os.path.join(participant_path, f"{participant_id}_bvp.csv")
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")
        
        # Extracting timestamp from log.txt tests file (for each participant) 
        timestamps = dfe.extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue
        
        df_eda = pd.read_csv(eda_file_path, usecols=["unix_timestamp", "eda"])
        df_bvp = pd.read_csv(bvp_file_path, usecols=["unix_timestamp", "bvp"])
        
        df_eda.rename(columns={"unix_timestamp": "Timestamp"}, inplace=True)
        df_bvp.rename(columns={"unix_timestamp": "Timestamp"}, inplace=True)
        
        # Converting timestamp into  ms
        df_eda["Timestamp"] = df_eda["Timestamp"] // 1000
        df_bvp["Timestamp"] = df_bvp["Timestamp"] // 1000
        
        # Retrive the useful timestamps
        try:
            ts_video1_start = timestamps["Inizio riproduzione primo video"]
            ts_video1_end = timestamps["Fine riproduzione primo video"]
            ts_video2_start = timestamps["Inizio riproduzione secondo video"]
            ts_video2_end = timestamps["Fine riproduzione secondo video"]
            ts_game1_start = timestamps["Avvio del primo gioco"]
            ts_game1_end = timestamps["Chiusura del primo gioco"]
            ts_game2_start = timestamps["Avvio secondo gioco"]
            ts_game2_end = timestamps["Chiusura secondo gioco"]
        except KeyError as e:
            print(f"Timestamp mancante nel log di {participant_id}: {e}")
            continue
        
        # 
        tolerance = 1000  # 1000 ms (1 second)

        # (exception for two of the participants, i will use second baseline for them)
        if participant_id in ["Alessandro_Martina", "Leo_Colucci"]:
            df_video1_eda = df_eda[(df_eda["Timestamp"] >= ts_video2_start - tolerance) & 
                                   (df_eda["Timestamp"] <= ts_video2_end + tolerance)].copy()
            df_video1_bvp = df_bvp[(df_bvp["Timestamp"] >= ts_video2_start - tolerance) & 
                                   (df_bvp["Timestamp"] <= ts_video2_end + tolerance)].copy()
        else:
            df_video1_eda = df_eda[(df_eda["Timestamp"] >= ts_video1_start - tolerance) & 
                                   (df_eda["Timestamp"] <= ts_video1_end + tolerance)].copy()
            df_video1_bvp = df_bvp[(df_bvp["Timestamp"] >= ts_video1_start - tolerance) & 
                                   (df_bvp["Timestamp"] <= ts_video1_end + tolerance)].copy()

        df_game1_eda = df_eda[(df_eda["Timestamp"] >= ts_game1_start - tolerance) & 
                              (df_eda["Timestamp"] <= ts_game1_end + tolerance)].copy()
        df_game2_eda = df_eda[(df_eda["Timestamp"] >= ts_game2_start - tolerance) & 
                              (df_eda["Timestamp"] <= ts_game2_end + tolerance)].copy()

        df_game1_bvp = df_bvp[(df_bvp["Timestamp"] >= ts_game1_start - tolerance) & 
                              (df_bvp["Timestamp"] <= ts_game1_end + tolerance)].copy()
        df_game2_bvp = df_bvp[(df_bvp["Timestamp"] >= ts_game2_start - tolerance) & 
                              (df_bvp["Timestamp"] <= ts_game2_end + tolerance)].copy()

        # Applying filters
        df_video1_eda["eda"] = butter_lowpass_filter(df_video1_eda["eda"], cutoff=1, fs=4)
        df_game1_eda["eda"] = butter_lowpass_filter(df_game1_eda["eda"], cutoff=1, fs=4)
        df_game2_eda["eda"] = butter_lowpass_filter(df_game2_eda["eda"], cutoff=1, fs=4)
        
        df_video1_bvp["bvp"] = butter_bandpass_filter(df_video1_bvp["bvp"], lowcut=1, highcut=8, fs=64)
        df_game1_bvp["bvp"] = butter_bandpass_filter(df_game1_bvp["bvp"], lowcut=1, highcut=8, fs=64)
        df_game2_bvp["bvp"] = butter_bandpass_filter(df_game2_bvp["bvp"], lowcut=1, highcut=8, fs=64)
        
        # Creating dictionary
        segmented_dataframes_eda[participant_id] = [df_video1_eda, df_game1_eda, df_game2_eda]
        segmented_dataframes_bvp[participant_id] = [df_video1_bvp, df_game1_bvp, df_game2_bvp]
    
    return segmented_dataframes_eda, segmented_dataframes_bvp

# Eda filter 
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# BVP filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)


def separate_eda_components(segmented_dataframes_eda):
    """
        Separates the phasic and tonic components of the EDA signal for each participant and each game session.

        Parameters:
            norm_eda (dict): Dictionary with participant IDs as keys and a list of two DataFrames 
                            containing EDA data for game 1 and game 2.

        Returns:
            dict: Dictionary with the same DataFrames, but with two additional columns: 'eda_phasic' and 'eda_tonic'.
    """
    eda_dataframes = {}

    for participant_id, dfs in segmented_dataframes_eda.items():
        processed_dfs = []

        for df in dfs:
            if df.empty:
                processed_dfs.append(df)
                continue
            
            df_eda = df.copy()

            # Extract the phasic component with cvxEDA 
            eda_signals = nk.eda_phasic(df_eda["eda"].values, sampling_rate=4)

            # Adding components to df 
            df_eda["eda_tonic"] = eda_signals["EDA_Tonic"].values # SCL - Skin Conductance Level 
            df_eda["eda_phasic"] = eda_signals["EDA_Phasic"].values # SCR - Skin Conductance Response (peaks)

            processed_dfs.append(df_eda)

        eda_dataframes[participant_id] = processed_dfs

    return eda_dataframes


def normalize_physio_dataframes(segmented_dataframes_eda, segmented_dataframes_bvp):
    """
        Normalizes the data of df_game1 and df_game2 using Z-score normalization,
        based on the mean and standard deviation computed from the last 60 seconds of df_video1.

        Parameters:
            segmented_dataframes_eda (dict): Dictionary with participant IDs as keys and three filtered DataFrames for EDA data as values.
            segmented_dataframes_bvp (dict): Dictionary with participant IDs as keys and three filtered DataFrames for BVP data as values.

        Returns:
            dict: Dictionary with participant IDs as keys and two normalized DataFrames (df_game1_normalized_eda, df_game2_normalized_eda).
            dict: Dictionary with participant IDs as keys and two normalized DataFrames (df_game1_normalized_bvp, df_game2_normalized_bvp).
    """
    normalized_dataframes_eda = {}
    normalized_dataframes_bvp = {}
    
    for participant_id, dfs in segmented_dataframes_eda.items():
        df_video1_eda, df_game1_eda, df_game2_eda = dfs
        
        if df_video1_eda.empty or df_game1_eda.empty or df_game2_eda.empty:
            print(f"{participant_id}: one or more df empty, skipping the normalization.")
            continue
        
        # Selecting last 30 secs of df_video1
        last_30s_start = df_video1_eda["Timestamp"].max() - (30 * 1000)  
        df_video1_last_30s = df_video1_eda[df_video1_eda["Timestamp"] >= last_30s_start]
        
        if df_video1_last_30s.empty:
            print(f"{participant_id}: No data in the last 30s, skipping the normalization.")
            continue
        
        # calculating mean and standard dev. for last 30s of baseline
        mean_eda = df_video1_last_30s["eda"].mean()
        std_eda = df_video1_last_30s["eda"].std()
        mean_eda_tonic = df_video1_last_30s["eda_tonic"].mean()
        std_eda_tonic = df_video1_last_30s["eda_tonic"].std()
        mean_eda_phasic = df_video1_last_30s["eda_phasic"].mean()
        std_eda_phasic = df_video1_last_30s["eda_phasic"].std()
        
        # avoiding the division by 0
        std_eda = std_eda if std_eda != 0 else 1

        # normalizing df_game1 e df_game2 with Z-score normalization
        df_game1_normalized_eda = df_game1_eda.copy()
        df_game2_normalized_eda = df_game2_eda.copy()
        
        df_game1_normalized_eda["eda"] = (df_game1_eda["eda"] - mean_eda) / std_eda
        df_game1_normalized_eda["eda_tonic"] = (df_game1_eda["eda_tonic"] - mean_eda_tonic) / std_eda_tonic
        df_game1_normalized_eda["eda_phasic"] = (df_game1_eda["eda_phasic"] - mean_eda_phasic) / std_eda_phasic

        df_game2_normalized_eda["eda"] = (df_game2_eda["eda"] - mean_eda) / std_eda
        df_game2_normalized_eda["eda_tonic"] = (df_game2_eda["eda_tonic"] - mean_eda_tonic) / std_eda_tonic
        df_game2_normalized_eda["eda_phasic"] = (df_game2_eda["eda_phasic"] - mean_eda_phasic) / std_eda_phasic

        normalized_dataframes_eda[participant_id] = [df_game1_normalized_eda, df_game2_normalized_eda]

    for participant_id, dfs in segmented_dataframes_bvp.items():
        df_video1_bvp, df_game1_bvp, df_game2_bvp = dfs
        
        if df_video1_bvp.empty or df_game1_bvp.empty or df_game2_bvp.empty:
            print(f"{participant_id}: one or more df empty, skipping the normalization.")
            continue
    
        last_30s_start = df_video1_bvp["Timestamp"].max() - (30 * 1000)  
        df_video1_last_30s = df_video1_bvp[df_video1_bvp["Timestamp"] >= last_30s_start]
        
        if df_video1_last_30s.empty:
            print(f"{participant_id}: No data in the last 30s, skipping the normalization.")
            continue

        mean_bvp = df_video1_last_30s["bvp"].mean()
        std_bvp = df_video1_last_30s["bvp"].std()
        
        std_bvp = std_bvp if std_bvp != 0 else 1

        df_game1_normalized_bvp = df_game1_bvp.copy()
        df_game2_normalized_bvp = df_game2_bvp.copy()
        
        df_game1_normalized_bvp["bvp"] = (df_game1_bvp["bvp"] - mean_bvp) / std_bvp
        df_game2_normalized_bvp["bvp"] = (df_game2_bvp["bvp"] - mean_bvp) / std_bvp

        
        # Assegna ai risultati
        normalized_dataframes_bvp[participant_id] = [df_game1_normalized_bvp, df_game2_normalized_bvp]
    
    return normalized_dataframes_eda, normalized_dataframes_bvp


def calculate_heart_rate(norm_bvp, fs=64):
    """
        Calculates heart rate (HR) from BVP data by detecting local minima
        and computing inter-beat intervals (IBI).

        Parameters:
            norm_bvp (dict): Dictionary with participant IDs as keys and a list of two normalized BVP DataFrames.
            fs (int): Sampling frequency of the physiological data in Hz (default: 64 Hz).

        Returns:
            dict: Dictionary with the same DataFrames but with added HR and IBI columns.
    """
    hr_dataframes = {}
    
    for participant_id, dfs in norm_bvp.items():
        hr_dfs = []
        
        for df in dfs:
            if df.empty:
                hr_dfs.append(df)
                continue
            
            # Creates a copy of df
            df_hr = df.copy()
            
            # finds the local minimum in the BVP signal (lowest systolic wave)
            inverted_bvp = -df_hr["bvp"].values
            # minimum distance of  0.5s (120 BPM max)
            peaks, _ = find_peaks(inverted_bvp, distance=fs*0.5)  
            
            if len(peaks) < 2:
                df_hr["hr"] = np.nan
                df_hr["ibi"] = np.nan
                hr_dfs.append(df_hr)
                continue
                
            peak_timestamps = df_hr["Timestamp"].iloc[peaks].values
            # calculating ibis
            ibi_ms = np.diff(peak_timestamps)
            
            # convering IBI into HR 
            hr_values = 60000 / ibi_ms  # 60000 ms = 1 minute

            # (60-100 BPM range max)
            hr_values = np.where((hr_values >= 60) & (hr_values <= 100), hr_values, np.nan)
            
            df_hr["hr"] = np.nan
            df_hr["ibi"] = np.nan
            
            for i in range(len(peaks)-1):
                df_hr.loc[df_hr.index[peaks[i+1]], "hr"] = hr_values[i]
                df_hr.loc[df_hr.index[peaks[i+1]], "ibi"] = ibi_ms[i]
            
            # linear Interpolation to make the signal continuous
            df_hr["hr"] = df_hr["hr"].interpolate(method='linear')
            
            hr_dfs.append(df_hr)
        
        hr_dataframes[participant_id] = hr_dfs
    
    return hr_dataframes

def extract_eda_metrics(norm_eda, log_dir):
    """
        Extracts EDA, EDA_Tonic, and EDA_Phasic metrics in the three defined intervals for each game.

        Parameters:
            norm_eda (dict): Dictionary with participant IDs as keys and a list of two DataFrames 
                            (one for game 1 and one for game 2), with columns: 
                            Timestamp, eda, eda_tonic, eda_phasic.
            log_dir (str): Path to the folder containing the log files used to extract the timestamps.

        Returns:
            dict: Dictionary with participant IDs as keys and two DataFrames per game containing the requested metrics.
    """
    segmented_metrics = {}

    for participant_id, dfs in norm_eda.items():
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")
        timestamps = dfe.extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue
    
        # interval cut defined in the experimental protocol
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


        def compute_metrics(df):
            """
                Computes various metrics for the EDA, EDA_tonic, and EDA_phasic signals.

                Parameters:
                    df (DataFrame): DataFrame containing at least the columns 'eda', 'eda_tonic', and 'eda_phasic'.
                
                Returns:
                    dict: Dictionary with the extracted metrics.
            """

            # Peaks for eda
            peaks, _ = find_peaks(df["eda_phasic"])

            metrics = {
                "min_eda": df["eda"].min(), 
                "max_eda": df["eda"].max(), 
                "avg_eda": df["eda"].mean(),

                "min_eda_tonic": df["eda_tonic"].min(), 
                "max_eda_tonic": df["eda_tonic"].max(), 
                "avg_eda_tonic": df["eda_tonic"].mean(),

                "min_eda_phasic": df["eda_phasic"].min(), 
                "max_eda_phasic": df["eda_phasic"].max(), 
                "avg_eda_phasic": df["eda_phasic"].mean(),

                "delta_eda": df["eda"].diff().mean(),

                # Average decrease rate: mean of the negative derivative of the EDA
                "f_DecRate_eda": df["eda"].diff()[df["eda"].diff() < 0].mean(),

                # Percentage of decreasing values in the EDA signal
                "f_DecTime_eda": (df["eda"].diff() < 0).sum() / len(df),

                # number of eda peaks
                "f_NbPeaks_eda": len(peaks)
            }
            
            return metrics

        # Extract intervals
        def extract_intervals(df, intervals):
            interval_metrics = []
            for idx, interval in enumerate(intervals):
                if len(interval) == 2: # if interval has 2 pauses (first two cases) 
                    start, end = interval
                    df_segment = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
                    metrics = compute_metrics(df_segment)
                    interval_metrics.append(metrics)
                elif len(interval) == 3: # if interval has 2 pauses + a list of a pauses (third case)
                    start, end, pauses = interval
                    df_segment = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
                    for pause in pauses:
                        df_segment = df_segment[~((df_segment["Timestamp"] >= pause[0]) & (df_segment["Timestamp"] <= pause[1]))]
                    metrics = compute_metrics(df_segment)
                    interval_metrics.append(metrics)
            return interval_metrics

        game1_df = dfs[0]
        game2_df = dfs[1]

        game1_metrics = extract_intervals(game1_df, game1_intervals)
        game2_metrics = extract_intervals(game2_df, game2_intervals)

        df_game1 = pd.DataFrame(game1_metrics)
        df_game1.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        df_game2 = pd.DataFrame(game2_metrics)
        df_game2.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        segmented_metrics[participant_id] = [df_game1, df_game2]

    return segmented_metrics


def extract_bvp_metrics(norm_bvp, log_dir):
    """
        Extracts BVP and HR metrics in the three defined intervals for each game.

        Parameters:
            norm_bvp (dict): Dictionary with participant IDs as keys and a list of two DataFrames 
                            (one for game 1 and one for game 2), with columns: 
                            Timestamp, bvp, hr.
            log_dir (str): Path to the folder containing the log files used to extract the timestamps.

        Returns:
            dict: Dictionary with participant IDs as keys and two DataFrames per game containing the requested metrics.
    """
    segmented_metrics = {}

    for participant_id, dfs in norm_bvp.items():
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")
        timestamps = dfe.extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue

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

        def compute_metrics(df, fs=64):
            """
                Computes various metrics for the BVP and HR signals.

                Parameters:
                    df (DataFrame): DataFrame containing at least the columns 'bvp', 'hr', and potentially 'ibi'.
                    fs (int): Sampling frequency of the BVP signal (default: 64 Hz).

                Returns:
                    dict: Dictionary with the extracted metrics in both the time and frequency domains.
            """
            if df.empty:
                return {metric: np.nan for metric in ["mu_bvp", "sigma_bvp", "mu_hr", "delta_hr", "sigma_hr", "SDNN", "RMSSD"]}

            # base metrics for BVP
            mu_bvp = df["bvp"].mean()
            sigma_bvp = df["bvp"].std()

            # base metrics for HR
            mu_hr = df["hr"].mean()
            delta_hr = df["hr"].diff().mean()
            sigma_hr = df["hr"].std()
            
            # init. of the hrv metrics (time analysis)
            SDNN = np.nan
            RMSSD = np.nan
            
            if "ibi" in df.columns:
                ibi_values = df["ibi"].dropna().values
                if len(ibi_values) > 1:
                    # SDNN: standard deviation of IBIs
                    SDNN = np.std(ibi_values)
                    # RMSSD: Square root of the mean of squared differences
                    ibi_diff = np.diff(ibi_values)
                    RMSSD = np.sqrt(np.mean(ibi_diff**2))

            return {
                "mu_bvp": mu_bvp,
                "sigma_bvp": sigma_bvp,
                "mu_hr": mu_hr,
                "delta_hr": delta_hr,
                "sigma_hr": sigma_hr,
                "SDNN": SDNN,
                "RMSSD": RMSSD
            }

        def extract_intervals(df, intervals):
            interval_metrics = []
            for idx, interval in enumerate(intervals):
                if len(interval) == 2:
                    start, end = interval
                    df_segment = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
                    metrics = compute_metrics(df_segment)
                    interval_metrics.append(metrics)
                elif len(interval) == 3:
                    start, end, pauses = interval
                    df_segment = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
                    for pause in pauses:
                        df_segment = df_segment[~((df_segment["Timestamp"] >= pause[0]) & (df_segment["Timestamp"] <= pause[1]))]
                    metrics = compute_metrics(df_segment)
                    interval_metrics.append(metrics)
            return interval_metrics

        game1_df = dfs[0]
        game2_df = dfs[1]

        game1_metrics = extract_intervals(game1_df, game1_intervals)
        game2_metrics = extract_intervals(game2_df, game2_intervals)

        df_game1 = pd.DataFrame(game1_metrics)
        df_game1.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        df_game2 = pd.DataFrame(game2_metrics)
        df_game2.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        segmented_metrics[participant_id] = [df_game1, df_game2]

    return segmented_metrics