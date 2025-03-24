import src.EEG.dataFrameEEG as dfe
import os
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
import neurokit2 as nk
import numpy as np

def split_dataframes(data_dir, log_dir):
    """
    Divide i DataFrame dei partecipanti in segmenti basati sui timestamp dei file di log e applica i filtri Butterworth.
    
    Parametri:
        data_dir (str): Percorso alla cartella contenente le cartelle dei partecipanti con i file CSV.
        log_dir (str): Percorso alla cartella contenente i file di log.
        
    Ritorna:
        dict: Dizionario con ID partecipante come chiave e sei DataFrame filtrati.
    """
    segmented_dataframes_eda = {}
    segmented_dataframes_bvp = {}
    
    for participant_id in os.listdir(data_dir):
        participant_path = os.path.join(data_dir, participant_id)
        if not os.path.isdir(participant_path):
            continue  # Salta se non è una cartella
        
        # Costruisci i percorsi ai file eda e bvp
        eda_file_path = os.path.join(participant_path, f"{participant_id}_eda.csv")
        bvp_file_path = os.path.join(participant_path, f"{participant_id}_bvp.csv")
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")
        
        # Estrarre i timestamp dal log
        timestamps = dfe.extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue
        
        # Caricamento dei CSV
        df_eda = pd.read_csv(eda_file_path, usecols=["unix_timestamp", "eda"])
        df_bvp = pd.read_csv(bvp_file_path, usecols=["unix_timestamp", "bvp"])
        
        # Rinominiamo le colonne timestamp
        df_eda.rename(columns={"unix_timestamp": "Timestamp"}, inplace=True)
        df_bvp.rename(columns={"unix_timestamp": "Timestamp"}, inplace=True)
        
        # Convertiamo i timestamp da microsecondi a millisecondi
        df_eda["Timestamp"] = df_eda["Timestamp"] // 1000
        df_bvp["Timestamp"] = df_bvp["Timestamp"] // 1000
        
        # Recupera i timestamp rilevanti
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
        
        # Suddivisione con una tolleranza di 1 secondo
        tolerance = 1000  # 1000 millisecondi (1 secondo)

        # Per Alessandro_Martina e Leo_Colucci, usa i timestamp del secondo video per df_video1
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

        # Applicazione dei filtri
        df_video1_eda["eda"] = butter_lowpass_filter(df_video1_eda["eda"], cutoff=1, fs=4)
        df_game1_eda["eda"] = butter_lowpass_filter(df_game1_eda["eda"], cutoff=1, fs=4)
        df_game2_eda["eda"] = butter_lowpass_filter(df_game2_eda["eda"], cutoff=1, fs=4)
        
        df_video1_bvp["bvp"] = butter_bandpass_filter(df_video1_bvp["bvp"], lowcut=1, highcut=8, fs=64)
        df_game1_bvp["bvp"] = butter_bandpass_filter(df_game1_bvp["bvp"], lowcut=1, highcut=8, fs=64)
        df_game2_bvp["bvp"] = butter_bandpass_filter(df_game2_bvp["bvp"], lowcut=1, highcut=8, fs=64)
        
        # Assegnazione al dizionario
        segmented_dataframes_eda[participant_id] = [df_video1_eda, df_game1_eda, df_game2_eda]
        segmented_dataframes_bvp[participant_id] = [df_video1_bvp, df_game1_bvp, df_game2_bvp]
    
    return segmented_dataframes_eda, segmented_dataframes_bvp

# Filtro per l'EDA 
def butter_lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

#Filtro per il BVP
def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band', analog=False)
    return filtfilt(b, a, data)


def separate_eda_components(segmented_dataframes_eda):
    """
    Separa la componente fasica e tonica del segnale EDA per ciascun partecipante e per ciascuna sessione di gioco.
    
    Parametri:
        norm_eda (dict): Dizionario con ID partecipante come chiave e una lista di due DataFrame 
                         contenenti i dati EDA di gioco1 e gioco2.
    
    Ritorna:
        dict: Dizionario con gli stessi DataFrame, ma con due colonne aggiuntive: 'eda_phasic' e 'eda_tonic'.
    """
    eda_dataframes = {}

    for participant_id, dfs in segmented_dataframes_eda.items():
        processed_dfs = []

        for df in dfs:
            if df.empty:
                processed_dfs.append(df)
                continue
            
            df_eda = df.copy()

            # Estrai la componente fasica con cvxEDA senza bisogno del pre-processing
            eda_signals = nk.eda_phasic(df_eda["eda"].values, sampling_rate=4)

            # Aggiungi le componenti al DataFrame
            df_eda["eda_tonic"] = eda_signals["EDA_Tonic"].values # Variazione lenta, SCL - Skin Conductance Level (Di solito comprende la maggior parte dell'ampiezza del segnale EDA originale)
            df_eda["eda_phasic"] = eda_signals["EDA_Phasic"].values # SCR - Skin Conductance Response, Consiste in piccole fluttuazioni (picchi) che si sovrappongono alla componente tonica

            processed_dfs.append(df_eda)

        eda_dataframes[participant_id] = processed_dfs

    return eda_dataframes


def normalize_physio_dataframes(segmented_dataframes_eda, segmented_dataframes_bvp):
    """
    Normalizza i dati di df_game1 e df_game2 con la Z-score normalization,
    usando media e deviazione standard calcolate sugli ultimi 60 secondi di df_video1.
    
    Parametri:
        segmented_dataframes_eda (dict): Dizionario con ID partecipante come chiave e tre DataFrame filtrati per i dati EDA.
        segmented_dataframes_bvp (dict): Dizionario con ID partecipante come chiave e tre DataFrame filtrati per i dati BVP.
    
    Ritorna:
        dict: Dizionario con ID partecipante come chiave e due DataFrame normalizzati (df_game1_normalized_eda, df_game2_normalized_eda).
        dict: Dizionario con ID partecipante come chiave e due DataFrame normalizzati (df_game1_normalized_eda, df_game2_normalized_bvp).
    """
    normalized_dataframes_eda = {}
    normalized_dataframes_bvp = {}
    
    for participant_id, dfs in segmented_dataframes_eda.items():
        df_video1_eda, df_game1_eda, df_game2_eda = dfs
        
        if df_video1_eda.empty or df_game1_eda.empty or df_game2_eda.empty:
            print(f"{participant_id}: Uno o più DataFrame vuoti, salto la normalizzazione.")
            continue
        
        # Seleziona gli ultimi 30 secondi di df_video1
        last_60s_start = df_video1_eda["Timestamp"].max() - (30 * 1000)  # 60 secondi in millisecondi
        df_video1_last_60s = df_video1_eda[df_video1_eda["Timestamp"] >= last_60s_start]
        
        if df_video1_last_60s.empty:
            print(f"{participant_id}: Nessun dato negli ultimi 30s di video1, salto la normalizzazione.")
            continue
        
        # Calcola media e deviazione standard sui dati di riferimento (ultimi 30s di df_video1)
        mean_eda = df_video1_last_60s["eda"].mean()
        std_eda = df_video1_last_60s["eda"].std()
        mean_eda_tonic = df_video1_last_60s["eda_tonic"].mean()
        std_eda_tonic = df_video1_last_60s["eda_tonic"].std()
        mean_eda_phasic = df_video1_last_60s["eda_phasic"].mean()
        std_eda_phasic = df_video1_last_60s["eda_phasic"].std()
        
        # Evita la divisione per zero
        std_eda = std_eda if std_eda != 0 else 1

        # Normalizza df_game1 e df_game2 con la Z-score normalization
        df_game1_normalized_eda = df_game1_eda.copy()
        df_game2_normalized_eda = df_game2_eda.copy()
        
        df_game1_normalized_eda["eda"] = (df_game1_eda["eda"] - mean_eda) / std_eda
        df_game1_normalized_eda["eda_tonic"] = (df_game1_eda["eda_tonic"] - mean_eda_tonic) / std_eda_tonic
        df_game1_normalized_eda["eda_phasic"] = (df_game1_eda["eda_phasic"] - mean_eda_phasic) / std_eda_phasic

        df_game2_normalized_eda["eda"] = (df_game2_eda["eda"] - mean_eda) / std_eda
        df_game2_normalized_eda["eda_tonic"] = (df_game2_eda["eda_tonic"] - mean_eda_tonic) / std_eda_tonic
        df_game2_normalized_eda["eda_phasic"] = (df_game2_eda["eda_phasic"] - mean_eda_phasic) / std_eda_phasic

        
        # Assegna ai risultati
        normalized_dataframes_eda[participant_id] = [df_game1_normalized_eda, df_game2_normalized_eda]

    for participant_id, dfs in segmented_dataframes_bvp.items():
        df_video1_bvp, df_game1_bvp, df_game2_bvp = dfs
        
        if df_video1_bvp.empty or df_game1_bvp.empty or df_game2_bvp.empty:
            print(f"{participant_id}: Uno o più DataFrame vuoti, salto la normalizzazione.")
            continue
        
        # Seleziona gli ultimi 30 secondi di df_video1
        last_60s_start = df_video1_bvp["Timestamp"].max() - (60 * 1000)  # 60 secondi in millisecondi
        df_video1_last_60s = df_video1_bvp[df_video1_bvp["Timestamp"] >= last_60s_start]
        
        if df_video1_last_60s.empty:
            print(f"{participant_id}: Nessun dato negli ultimi 30s di video1, salto la normalizzazione.")
            continue
        
        # Calcola media e deviazione standard sui dati di riferimento (ultimi 30s di df_video1)
        mean_bvp = df_video1_last_60s["bvp"].mean()
        std_bvp = df_video1_last_60s["bvp"].std()
        
        # Evita la divisione per zero
        std_bvp = std_bvp if std_bvp != 0 else 1

        # Normalizza df_game1 e df_game2 con la Z-score normalization
        df_game1_normalized_bvp = df_game1_bvp.copy()
        df_game2_normalized_bvp = df_game2_bvp.copy()
        
        df_game1_normalized_bvp["bvp"] = (df_game1_bvp["bvp"] - mean_bvp) / std_bvp
        df_game2_normalized_bvp["bvp"] = (df_game2_bvp["bvp"] - mean_bvp) / std_bvp

        
        # Assegna ai risultati
        normalized_dataframes_bvp[participant_id] = [df_game1_normalized_bvp, df_game2_normalized_bvp]
    
    return normalized_dataframes_eda, normalized_dataframes_bvp

def calculate_heart_rate(norm_bvp, fs=64):
    """
    Calcola la frequenza cardiaca (HR) dai dati BVP trovando i minimi locali
    e calcolando i periodi inter-battito.
    
    Parametri:
        norm_bvp (dict): Dizionario con ID partecipante come chiave e una lista di due DataFrame bvp normalizzati.
        fs (int): Frequenza di campionamento dei dati fisiologici in Hz (default: 64Hz).
        
    Ritorna:
        dict: Dizionario con gli stessi DataFrame ma con la colonna HR e IBI aggiunte.
    """
    
    hr_dataframes = {}
    
    for participant_id, dfs in norm_bvp.items():
        hr_dfs = []
        
        for df in dfs:
            if df.empty:
                hr_dfs.append(df)
                continue
            
            # Crea una copia del DataFrame
            df_hr = df.copy()
            
            # Trova i minimi locali nel segnale BVP (equivalenti ai punti più bassi dell'onda sistolica)
            inverted_bvp = -df_hr["bvp"].values
            # Distanza minima di 0.5 secondi (corrisponde a 120 BPM max)
            peaks, _ = find_peaks(inverted_bvp, distance=fs*0.5)  
            
            # Se non ci sono abbastanza picchi, non possiamo calcolare l'HR
            if len(peaks) < 2:
                df_hr["hr"] = np.nan
                df_hr["ibi"] = np.nan
                hr_dfs.append(df_hr)
                continue
                
            # Calcola gli intervalli inter-battito in millisecondi
            peak_timestamps = df_hr["Timestamp"].iloc[peaks].values
            ibi_ms = np.diff(peak_timestamps)
            
            # Converte gli IBI in HR (battiti al minuto)
            hr_values = 60000 / ibi_ms  # 60000 ms = 1 minuto
            
            # Applica un filtro per il range per adulti a riposo (60-100 BPM)
            hr_values = np.where((hr_values >= 60) & (hr_values <= 100), hr_values, np.nan)
            
            # Inizializza le colonne HR e IBI con NaN
            df_hr["hr"] = np.nan
            df_hr["ibi"] = np.nan
            
            # Assegna i valori HR e IBI ai campioni corrispondenti ai picchi
            for i in range(len(peaks)-1):
                df_hr.loc[df_hr.index[peaks[i+1]], "hr"] = hr_values[i]
                df_hr.loc[df_hr.index[peaks[i+1]], "ibi"] = ibi_ms[i]
            
            # Interpolazione lineare per riempire i valori NaN tra i picchi (solo per HR)
            df_hr["hr"] = df_hr["hr"].interpolate(method='linear')
            
            hr_dfs.append(df_hr)
        
        hr_dataframes[participant_id] = hr_dfs
    
    return hr_dataframes

def extract_eda_metrics(norm_eda, log_dir):
    """
    Estrae le metriche EDA, EDA_Tonic ed EDA_Phasic nei tre intervalli definiti per ogni gioco.

    Parametri:
        norm_eda (dict): Dizionario con ID partecipante come chiave e una lista di due DataFrame 
                         (uno per il gioco 1 e uno per il gioco 2), con colonne: 
                         Timestamp, eda, eda_tonic, eda_phasic.
        log_dir (str): Percorso alla cartella contenente i file di log per estrarre i timestamp.

    Ritorna:
        dict: Dizionario con ID partecipante come chiave e due DataFrame per ciascun gioco contenenti le metriche richieste.
    """
    segmented_metrics = {}

    for participant_id, dfs in norm_eda.items():
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")
        timestamps = dfe.extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue
        
        # Definizione degli intervalli per il primo gioco
        game1_intervals = [
            (timestamps["Avvio del primo gioco"], timestamps["primo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["primo iGEQ terminato, gioco ripreso"], timestamps["secondo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio del primo gioco"], timestamps["Chiusura del primo gioco"], [
                (timestamps["primo iGEQ mostrato, gioco messo in pausa"], timestamps["primo iGEQ terminato, gioco ripreso"]),
                (timestamps["secondo iGEQ mostrato, gioco messo in pausa"], timestamps["secondo iGEQ terminato, gioco ripreso"])
            ])
        ]

        # Definizione degli intervalli per il secondo gioco
        game2_intervals = [
            (timestamps["Avvio secondo gioco"], timestamps["terzo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["terzo iGEQ terminato, gioco ripreso"], timestamps["quarto iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio secondo gioco"], timestamps["Chiusura secondo gioco"], [
                (timestamps["terzo iGEQ mostrato, gioco messo in pausa"], timestamps["terzo iGEQ terminato, gioco ripreso"]),
                (timestamps["quarto iGEQ mostrato, gioco messo in pausa"], timestamps["quarto iGEQ terminato, gioco ripreso"])
            ])
        ]

        # Funzione per calcolare le metriche richieste
        def compute_metrics(df):
            """
            Calcola diverse metriche per il segnale EDA, EDA_tonic e EDA_phasic.

            Parametri:
                df (DataFrame): DataFrame contenente almeno le colonne 'eda', 'eda_tonic' e 'eda_phasic'.
            
            Ritorna:
                dict: Dizionario con le metriche estratte.
            """

            # Rilevazione dei picchi corretta usando scipy.signal.find_peaks
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

                # Tasso medio di decremento: media della derivata negativa dell'EDA
                "f_DecRate_eda": df["eda"].diff()[df["eda"].diff() < 0].mean(),

                # Percentuale di valori decrescenti nel segnale EDA
                "f_DecTime_eda": (df["eda"].diff() < 0).sum() / len(df),

                # Numero di picchi nel segnale EDA (corretto)
                "f_NbPeaks_eda": len(peaks)
            }
            
            return metrics

        # Funzione per estrarre i dati nei tre intervalli
        def extract_intervals(df, intervals):
            interval_metrics = []
            for idx, interval in enumerate(intervals):
                if len(interval) == 2: # Se l'intervallo ha due pause (primi due casi) entra qui
                    start, end = interval
                    df_segment = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
                    metrics = compute_metrics(df_segment)
                    interval_metrics.append(metrics)
                elif len(interval) == 3: # Se l'intervallo ha due pause + una lista di pause (ultimo caso) entra qui
                    start, end, pauses = interval
                    df_segment = df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
                    for pause in pauses:
                        df_segment = df_segment[~((df_segment["Timestamp"] >= pause[0]) & (df_segment["Timestamp"] <= pause[1]))]
                    metrics = compute_metrics(df_segment)
                    interval_metrics.append(metrics)
            return interval_metrics

        # Estrazione delle metriche per i due giochi
        game1_df = dfs[0]
        game2_df = dfs[1]

        game1_metrics = extract_intervals(game1_df, game1_intervals)
        game2_metrics = extract_intervals(game2_df, game2_intervals)

        # Creazione dei DataFrame
        df_game1 = pd.DataFrame(game1_metrics)
        df_game1.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        df_game2 = pd.DataFrame(game2_metrics)
        df_game2.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        segmented_metrics[participant_id] = [df_game1, df_game2]

    return segmented_metrics


def extract_bvp_metrics(norm_bvp, log_dir):
    """
    Estrae le metriche BVP e HR nei tre intervalli definiti per ogni gioco.

    Parametri:
        norm_bvp (dict): Dizionario con ID partecipante come chiave e una lista di due DataFrame 
                         (uno per il gioco 1 e uno per il gioco 2), con colonne: 
                         Timestamp, bvp, hr.
        log_dir (str): Percorso alla cartella contenente i file di log per estrarre i timestamp.

    Ritorna:
        dict: Dizionario con ID partecipante come chiave e due DataFrame per ciascun gioco contenenti le metriche richieste.
    """
    segmented_metrics = {}

    for participant_id, dfs in norm_bvp.items():
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")
        timestamps = dfe.extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue

        # Definizione degli intervalli per il primo e secondo gioco
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

        # Funzione per calcolare le metriche richieste
        def compute_metrics(df, fs=64):
            """
            Calcola diverse metriche per il segnale BVP e HR.

            Parametri:
                df (DataFrame): DataFrame contenente almeno le colonne 'bvp', 'hr' e potenzialmente 'ibi'.
                fs (int): Frequenza di campionamento del segnale BVP (default 64 Hz).

            Ritorna:
                dict: Dizionario con le metriche estratte sia nel dominio della frequenza che del tempo.
            """

            if df.empty:
                return {metric: np.nan for metric in ["mu_bvp", "sigma_bvp", "mu_hr", "delta_hr", "sigma_hr", "SDNN", "RMSSD"]}

            # Metriche base BVP
            mu_bvp = df["bvp"].mean()
            sigma_bvp = df["bvp"].std()

            # Metriche base HR
            mu_hr = df["hr"].mean()
            delta_hr = df["hr"].diff().mean()
            sigma_hr = df["hr"].std()
            
            # Inizializziamo le metriche HRV nel dominio del tempo
            SDNN = np.nan
            RMSSD = np.nan
            pNN50 = np.nan
            
            # Calcolo metriche HRV nel dominio del tempo se abbiamo la colonna IBI
            if "ibi" in df.columns:
                ibi_values = df["ibi"].dropna().values
                if len(ibi_values) > 1:
                    # SDNN: deviazione standard di tutti gli intervalli NN (IBI)
                    SDNN = np.std(ibi_values)
                    
                    # RMSSD: radice quadrata della media delle differenze al quadrato
                    # tra intervalli NN successivi
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

        # Funzione per estrarre i dati nei tre intervalli
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

        # Estrazione delle metriche per i due giochi
        game1_df = dfs[0]
        game2_df = dfs[1]

        game1_metrics = extract_intervals(game1_df, game1_intervals)
        game2_metrics = extract_intervals(game2_df, game2_intervals)

        # Creazione dei DataFrame
        df_game1 = pd.DataFrame(game1_metrics)
        df_game1.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        df_game2 = pd.DataFrame(game2_metrics)
        df_game2.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        segmented_metrics[participant_id] = [df_game1, df_game2]

    return segmented_metrics