import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast  # Per convertire le stringhe che contengono liste in liste reali


def create_power_by_band_dataframes(json_file_paths):
    """
    Crea un DataFrame separato per ciascun partecipante dai file power_by_band.json,
    mantenendo il timestamp nel formato originale.

    Parametri:
        json_file_paths (list): Lista dei percorsi dei file JSON.

    Ritorna:
        dict: Un dizionario con ID partecipante come chiave e DataFrame corrispondente come valore.
    """
    dataframes = {}

    for file_path in json_file_paths:
        # Verifica che sia il file power_by_band.json
        if not file_path.endswith("power_by_band.json"):
            continue  # Salta il file se non è power_by_band.json

        # Ottieni l'ID del partecipante dal percorso
        participant_id = os.path.basename(os.path.dirname(file_path))
        print(f"Elaborazione per partecipante: {participant_id}")

        # Carica il file JSON
        with open(file_path, 'r') as f:
            power_data = json.load(f)

        # Estrai i dati e il timestamp
        rows = []
        for entry in power_data:
            data = entry["data"]
            timestamp = entry["timestamp"]  # Mantiene il timestamp originale

            # Crea una riga con i dati delle bande di frequenza
            row = {"ID": participant_id, "Timestamp": timestamp}
            row.update(data)  # Aggiunge alpha, beta, delta, gamma, theta come colonne

            rows.append(row)

        # Crea il DataFrame per il partecipante
        df = pd.DataFrame(rows)
        dataframes[participant_id] = df

    return dataframes


def extract_timestamps_from_log(log_file_path, participant_id):
    """
    Estrae i timestamp rilevanti da un file di log.

    Parametri:
        log_file_path (str): Percorso del file di log.
        participant_id (str): ID del partecipante.

    Ritorna:
        dict: Dizionario con i timestamp estratti o None se il file non esiste.
    """
    if not os.path.exists(log_file_path):
        print(f"File di log non trovato per {participant_id}")
        return None

    timestamps = {}
    with open(log_file_path, 'r') as log_file:
        log_lines = log_file.readlines()

    for line in log_lines:
        if "Timestamp UNIX:" in line:
            parts = line.strip().split(" - ")
            if len(parts) < 3:
                print(f"Formato non valido nel log di {participant_id}: {line.strip()}")
                continue  # Salta questa riga

            event = parts[1]
            timestamp_parts = parts[2].split(": ")
            if len(timestamp_parts) < 2:
                print(f"Timestamp mancante nel log di {participant_id}: {line.strip()}")
                continue  # Salta questa riga

            timestamp = int(timestamp_parts[1]) * 1000  # Converte il timestamp in millisecondi
            timestamps[event] = timestamp

    return timestamps

def split_dataframes(dataframes, log_dir):
    """
    Divide i DataFrame dei partecipanti in tre segmenti basati sui timestamp dei file di log.
    
    Parametri:
        dataframes (dict): Dizionario con ID partecipante come chiave e DataFrame corrispondente come valore.
        log_dir (str): Percorso alla cartella contenente i file di log.
        
    Ritorna:
        dict: Dizionario con ID partecipante come chiave e una lista di tre DataFrame corrispondenti ai segmenti.
    """
    segmented_dataframes = {}
    channels = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    waves = ['alpha', 'beta', 'delta', 'gamma', 'theta']

    for participant_id, df in dataframes.items():
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")

        # Estrarre i timestamp dal log usando la funzione esterna
        timestamps = extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue

        # Stampa i timestamp estratti per debug
        print(f"Timestamp per {participant_id}: {timestamps}")

        # Recupera i timestamp rilevanti
        try:
            ts_video1_start = timestamps["Inizio riproduzione primo video"]
            ts_video1_end = timestamps["Fine riproduzione primo video"]
            ts_game1_start = timestamps["Avvio del primo gioco"]
            ts_game1_end = timestamps["Chiusura del primo gioco"]
            ts_video2_end = timestamps["Avvio secondo gioco"]
            ts_game2_end = timestamps["Chiusura secondo gioco"]
            ts_first_igeq_start = timestamps["primo iGEQ mostrato, gioco messo in pausa"]
            ts_first_igeq_end = timestamps["primo iGEQ terminato, gioco ripreso"]
            ts_second_igeq_start = timestamps["secondo iGEQ mostrato, gioco messo in pausa"]
            ts_second_igeq_end = timestamps["secondo iGEQ terminato, gioco ripreso"]
            ts_third_igeq_start = timestamps["terzo iGEQ mostrato, gioco messo in pausa"]
            ts_third_igeq_end = timestamps["terzo iGEQ terminato, gioco ripreso"]
            ts_fourth_igeq_start = timestamps["quarto iGEQ mostrato, gioco messo in pausa"]
            ts_fourth_igeq_end = timestamps["quarto iGEQ terminato, gioco ripreso"]
         
        except KeyError as e:
            print(f"Timestamp mancante nel log di {participant_id}: {e}")
            continue
        
        # Stampa i valori prima della suddivisione
        print(f"{participant_id}: df Timestamp min = {df['Timestamp'].min()}, max = {df['Timestamp'].max()}")
        print(f"{participant_id}: ts_video1_start={ts_video1_start}, ts_video1_end={ts_video1_end}")
        print(f"{participant_id}: ts_game1_start={ts_game1_start}, ts_game1_end={ts_game1_end}")
        print(f"{participant_id}: ts_video2_end={ts_video2_end}, ts_game2_end={ts_game2_end}")
        print(f"{participant_id}: ts_video2_end={ts_first_igeq_start}, ts_game2_end={ts_first_igeq_end}")
        print(f"{participant_id}: ts_video2_end={ts_second_igeq_start}, ts_game2_end={ts_second_igeq_end}")
        print(f"{participant_id}: ts_video2_end={ts_third_igeq_start}, ts_game2_end={ts_third_igeq_end}")
        print(f"{participant_id}: ts_video2_end={ts_fourth_igeq_start}, ts_game2_end={ts_fourth_igeq_end}")
        
        # Suddivisione del DataFrame in base ai timestamp con tolleranza di 1 secondo
        tolerance = 1000  # tolleranza di 1000 millisecondi (1 secondo)
        df_video1 = df[(df["Timestamp"] >= ts_video1_start - tolerance) & (df["Timestamp"] <= ts_video1_end + tolerance)]
        df_game1 = df[(df["Timestamp"] >= ts_game1_start - tolerance) & (df["Timestamp"] <= ts_game1_end + tolerance)]
        df_game2 = df[(df["Timestamp"] >= ts_video2_end - tolerance) & (df["Timestamp"] <= ts_game2_end + tolerance)]

        # Stampa il numero di righe nei segmenti per verificare
        #print(f"{participant_id}: video1={len(df_video1)}, game1={len(df_game1)}, game2={len(df_game2)}")

        def extract_interval(df, start, end):
            return df[(df["Timestamp"] >= start) & (df["Timestamp"] <= end)]
        
        intervals_game1 = [
            extract_interval(df_game1, ts_game1_start, ts_first_igeq_start),
            extract_interval(df_game1, ts_first_igeq_end, ts_second_igeq_start),
            extract_interval(df_game1, ts_second_igeq_end, ts_game1_end)
        ]

        intervals_game2 = [
            extract_interval(df_game2, ts_video2_end, ts_third_igeq_start),
            extract_interval(df_game2, ts_third_igeq_end, ts_fourth_igeq_start),
            extract_interval(df_game2, ts_fourth_igeq_end, ts_game2_end)
        ]

        segmented_dataframes[participant_id] = [df_video1, df_game1, df_game2]
        
        '''
        # Crea i plot per ogni canale (sia per game1 che per game2 nello stesso grafico)
        for i, channel in enumerate(channels):
            plt.figure(figsize=(10, 6))  # Un nuovo "foglio" per ogni canale
            plt.title(f'{participant_id} - {channel}')  # Titolo con il nome del canale

            # Plot per ogni onda
            for wave in waves:
                # Estraggo i dati della specifica onda per il canale attuale da entrambi i giochi
                data_game1 = df_game1[wave].apply(lambda x: x[i])  # Dati da game1
                data_game2 = df_game2[wave].apply(lambda x: x[i])  # Dati da game2
                
                # Linea originale per game1
                plt.plot(df_game1['Timestamp'], data_game1, label=f'{wave} - game1', alpha=0.6)
                
                # Linea originale per game2
                plt.plot(df_game2['Timestamp'], data_game2, label=f'{wave} - game2', alpha=0.6)
            
            # Aggiungi le linee verticali per i timestamp degli iGEQ
            # Interruzioni di game1
            plt.axvline(x=ts_first_igeq_start, color='r', linestyle='--', label="1st iGEQ Start - game1")
            plt.axvline(x=ts_first_igeq_end, color='r', linestyle='--', label="1st iGEQ End - game1")
            plt.axvline(x=ts_second_igeq_start, color='g', linestyle='--', label="2nd iGEQ Start - game1")
            plt.axvline(x=ts_second_igeq_end, color='g', linestyle='--', label="2nd iGEQ End - game1")
            
            # Interruzioni di game2
            plt.axvline(x=ts_third_igeq_start, color='b', linestyle='--', label="3rd iGEQ Start - game2")
            plt.axvline(x=ts_third_igeq_end, color='b', linestyle='--', label="3rd iGEQ End - game2")
            plt.axvline(x=ts_fourth_igeq_start, color='m', linestyle='--', label="4th iGEQ Start - game2")
            plt.axvline(x=ts_fourth_igeq_end, color='m', linestyle='--', label="4th iGEQ End - game2")

            # Aggiungi griglia, etichette e legenda
            plt.xlabel('Tempo')
            plt.ylabel('Ampiezza EEG')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.legend(loc='upper right')

            # Mostra il grafico
            plt.xticks(rotation=45)  # Ruota i tick dell'asse X per una visualizzazione migliore
            plt.tight_layout()  # Per evitare sovrapposizioni
            plt.show()

        
        def plot_heatmaps(intervals, title):
            fig, axes = plt.subplots(2, 3, figsize=(36, 16), gridspec_kw={'height_ratios': [3, 1]})
            fig.suptitle(title, fontsize=20)
            
            for i, df_interval in enumerate(intervals):
                if df_interval.empty:
                    continue
                
                heatmap_data = np.zeros((len(waves), len(channels)))
                for w_idx, wave in enumerate(waves):
                    if wave in df_interval.columns:
                        wave_values = np.array(df_interval[wave].tolist())
                        if wave_values.shape[1] == len(channels):
                            heatmap_data[w_idx, :] = wave_values.mean(axis=0)
                
                sns.heatmap(heatmap_data, xticklabels=channels, yticklabels=waves, ax=axes[0, i], cmap='Reds', annot=True, square=True, fmt=".1f", annot_kws={"size": 8})
                axes[0, i].set_yticklabels(axes[0, i].get_yticklabels(), rotation=45, ha="right")
                axes[0, i].set_title(f"Intervallo {i+1}")
                
                # Grafico lineare sotto la heatmap con tempo normalizzato in minuti (0-5)
                timestamps = df_interval["Timestamp"].values
                time_in_minutes = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0]) * 5  # Normalizza da 0 a 5 minuti
                mean_wave_values = df_interval[waves].applymap(np.mean).values
                for w_idx, wave in enumerate(waves):
                    axes[1, i].plot(time_in_minutes, mean_wave_values[:, w_idx], label=wave)
                
                axes[1, i].set_xlabel("Tempo (minuti)")
                axes[1, i].set_ylabel("EEG Mean Value")
                axes[1, i].legend()
                axes[1, i].set_title(f"Distribuzione EEG - Intervallo {i+1}")
            
            plt.show()
        
        plot_heatmaps(intervals_game1, f"Partecipante {participant_id} - Primo Gioco")
        plot_heatmaps(intervals_game2, f"Partecipante {participant_id} - Secondo Gioco")
        '''
    
    return segmented_dataframes

def normalize_eeg(segmented_dataframes):
    """
    Normalizza df_game1 e df_game2 per ogni partecipante utilizzando la Z-score normalization
    calcolata sugli ultimi 30 secondi di df_video1. 

    Parametri:
    segmented_dataframes (dict): Dizionario contenente i DataFrame di ciascun partecipante nel formato: {participant_id: [df_video1, df_game1, df_game2]}

    Ritorna:
    dict: Dizionario con i DataFrame df_game1 e df_game2 normalizzati per ogni partecipante.
    """
    normalized_dataframes = {}

    for participant_id, dfs in segmented_dataframes.items():
        df_video1, df_game1, df_game2 = dfs

        # Creare una copia per evitare modifiche in-place
        df_video1 = df_video1.copy()

        # Assicuriamoci che il Timestamp sia numerico e convertiamo da millisecondi a secondi
        df_video1["Timestamp"] = pd.to_numeric(df_video1["Timestamp"], errors="coerce") / 1000

        # Determinare il timestamp massimo
        max_time = df_video1["Timestamp"].max()

        # Estrarre solo gli ultimi 30 secondi eliminando il primo minuto e mezzo
        baseline_df = df_video1[df_video1["Timestamp"] >= (max_time - 30)].copy()
        

        # Debugging: Stampiamo i dati della baseline per verificare
        if baseline_df.empty:
            print(f"La baseline è vuota per il partecipante {participant_id}. Salto la normalizzazione.")
            continue
        else:
            print(f"Baseline trovata per il partecipante {participant_id}, contiene {len(baseline_df)} righe.")

        # Colonne EEG
        eeg_columns = ['alpha', 'beta', 'delta', 'gamma', 'theta']

        # Convertire le stringhe in liste di numeri
        for col in eeg_columns:
            baseline_df[col] = baseline_df[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))

        # Calcolare la media e deviazione standard della baseline per ogni canale EEG (otteniamo 8 valori per banda)
        baseline_means = {col: np.mean(np.vstack(baseline_df[col]), axis=0) for col in eeg_columns}
        baseline_stds = {col: np.std(np.vstack(baseline_df[col]), axis=0) for col in eeg_columns}

        # Funzione per normalizzare un DataFrame EEG con Z-score normalization
        def normalize_df(df):
            df_norm = df.copy()

            # Assicuriamoci che Timestamp sia numerico e convertiamo da millisecondi a secondi
            df_norm["Timestamp"] = pd.to_numeric(df_norm["Timestamp"], errors="coerce") / 1000.0

            # Convertiamo le stringhe in liste di numeri per ogni colonna EEG
            for col in eeg_columns:
                df_norm[col] = df_norm[col].apply(lambda x: np.array(ast.literal_eval(x)) if isinstance(x, str) else np.array(x))

            # Applichiamo la Z-score normalization "element-wise" (su tutti gli 8 valori per banda)
            for col in eeg_columns:
                df_norm[col] = df_norm[col].apply(lambda x: (x - baseline_means[col]) / baseline_stds[col] if np.any(baseline_stds[col] != 0) else x)

            return df_norm

        # Normalizzare df_game1 e df_game2
        df_game1_norm = normalize_df(df_game1)
        df_game2_norm = normalize_df(df_game2)

        # Salvare i DataFrame normalizzati
        normalized_dataframes[participant_id] = [df_game1_norm, df_game2_norm]

    return normalized_dataframes

def compute_aggregated_rms_amplitudes(normalized_segmented_dataframes, log_dir):
    """
    Calcola la Root Mean Square (RMS) Amplitude aggregata per ogni banda EEG e ogni canale,
    restituendo tre righe per ogni gioco (sei righe in totale per partecipante).

    Parametri:
    normalized_segmented_dataframes (dict): Dizionario contenente i DataFrame normalizzati per ogni partecipante.
                                            I timestamp nei DataFrame sono in **secondi**.
    log_dir (str): Percorso della cartella dei log. I timestamp nei log sono in **millisecondi**.

    Ritorna:
    dict: Dizionario con i DataFrame di RMS aggregata per ciascun partecipante.
    """
    aggregated_dataframes = {}
    eeg_columns = ['alpha', 'beta', 'delta', 'gamma', 'theta']

    for participant_id, dfs in normalized_segmented_dataframes.items():
        df_game1_norm, df_game2_norm = dfs
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")

        # Estrai i timestamp dal log (che sono in millisecondi)
        timestamps = extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue

        # Convertiamo i timestamp dei log da millisecondi a secondi
        timestamps = {key: value / 1000.0 for key, value in timestamps.items()}

        # Funzione per calcolare la RMS per un intervallo specifico
        def compute_rms(df, start, end, exclude_intervals=None, skip_start=1.0):
            """ Calcola la Root Mean Square su un intervallo specifico, saltando i primi 'skip_start' secondi
                 ed escludendo specifici sotto-intervalli """
            df_interval = df[(df["Timestamp"] >= start + skip_start) & (df["Timestamp"] <= end)]
            
            # Rimuovi i dati all'interno degli intervalli da escludere
            if exclude_intervals:
                for exc_start, exc_end in exclude_intervals:
                    df_interval = df_interval[(df_interval["Timestamp"] < exc_start) | (df_interval["Timestamp"] > exc_end)]
            
            if df_interval.empty:
                return {band: [np.nan] * 8 for band in eeg_columns}  # Se vuoto, restituisci NaN

            rms_values = {}
            for band in eeg_columns:
                values = np.array(df_interval[band].to_list())  # Converte direttamente in array NumPy
                rms_values[band] = np.sqrt(np.mean(np.square(values), axis=0))  # RMS per ogni canale

            return rms_values

        # Definizione degli intervalli per il primo gioco (timestamp ora in secondi)
        game1_intervals = [
            (timestamps["Avvio del primo gioco"], timestamps["primo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["primo iGEQ terminato, gioco ripreso"], timestamps["secondo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio del primo gioco"], timestamps["Chiusura del primo gioco"], [
                (timestamps["primo iGEQ mostrato, gioco messo in pausa"], timestamps["primo iGEQ terminato, gioco ripreso"]),
                (timestamps["secondo iGEQ mostrato, gioco messo in pausa"], timestamps["secondo iGEQ terminato, gioco ripreso"])
            ])
        ]

        # Definizione degli intervalli per il secondo gioco (timestamp ora in secondi)
        game2_intervals = [
            (timestamps["Avvio secondo gioco"], timestamps["terzo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["terzo iGEQ terminato, gioco ripreso"], timestamps["quarto iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio secondo gioco"], timestamps["Chiusura secondo gioco"], [
                (timestamps["terzo iGEQ mostrato, gioco messo in pausa"], timestamps["terzo iGEQ terminato, gioco ripreso"]),
                (timestamps["quarto iGEQ mostrato, gioco messo in pausa"], timestamps["quarto iGEQ terminato, gioco ripreso"])
            ])
        ]

        # Calcolo delle RMS per i tre intervalli di ogni gioco
        game1_rms = [compute_rms(df_game1_norm, *interval[:2], exclude_intervals=interval[2] if len(interval) > 2 else None) for interval in game1_intervals]
        game2_rms = [compute_rms(df_game2_norm, *interval[:2], exclude_intervals=interval[2] if len(interval) > 2 else None) for interval in game2_intervals]

        # Creazione dei DataFrame
        df_game1_rms = pd.DataFrame(game1_rms)
        df_game1_rms.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        df_game2_rms = pd.DataFrame(game2_rms)
        df_game2_rms.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        # Salviamo i DataFrame nel dizionario finale
        aggregated_dataframes[participant_id] = {
            "game1_rms": df_game1_rms,
            "game2_rms": df_game2_rms
        }

    return aggregated_dataframes

def compute_aggregated_ptp_amplitudes(normalized_segmented_dataframes, log_dir):
    """
    Calcola la Peak-to-Peak (PtP) Amplitude aggregata per ogni banda EEG e ogni canale,
    restituendo tre righe per ogni gioco (sei righe in totale per partecipante).

    Parametri:
    normalized_segmented_dataframes (dict): Dizionario contenente i DataFrame normalizzati per ogni partecipante.
                                            I timestamp nei DataFrame sono in **secondi**.
    log_dir (str): Percorso della cartella dei log. I timestamp nei log sono in **millisecondi**.

    Ritorna:
    dict: Dizionario con i DataFrame di PtP aggregata per ciascun partecipante.
    """
    aggregated_dataframes = {}
    eeg_columns = ['alpha', 'beta', 'delta', 'gamma', 'theta']

    for participant_id, dfs in normalized_segmented_dataframes.items():
        df_game1_norm, df_game2_norm = dfs
        log_file_path = os.path.join(log_dir, f"{participant_id}.txt")

        # Estrai i timestamp dal log (che sono in millisecondi)
        timestamps = extract_timestamps_from_log(log_file_path, participant_id)
        if timestamps is None:
            continue

        # Convertiamo i timestamp dei log da millisecondi a secondi
        timestamps = {key: value / 1000.0 for key, value in timestamps.items()}

        # Funzione per calcolare la Peak-to-Peak per un intervallo specifico
        def compute_ptp(df, start, end, exclude_intervals=None, skip_start=1.0):
            """ Calcola la Peak-to-Peak su un intervallo specifico, saltando i primi 'skip_start' secondi
                ed escludendo specifici sotto-intervalli """
            df_interval = df[(df["Timestamp"] >= start + skip_start) & (df["Timestamp"] <= end)]
            
            # Stampa il numero di righe prima dell'esclusione
            print(f"Partecipante: {participant_id}, Intervallo: {start} - {end}")
            print(f"Numero di righe prima dell'esclusione: {len(df_interval)}")
            
            # Rimuovi i dati all'interno degli intervalli da escludere
            if exclude_intervals:
                for exc_start, exc_end in exclude_intervals:
                    df_interval = df_interval[(df_interval["Timestamp"] < exc_start) | (df_interval["Timestamp"] > exc_end)]
            
            # Stampa il numero di righe dopo l'esclusione
            print(f"Numero di righe dopo l'esclusione: {len(df_interval)}")
            print("-" * 50)
            
            if df_interval.empty:
                return {band: [np.nan] * 8 for band in eeg_columns}  # Se vuoto, restituisci NaN

            ptp_values = {}
            for band in eeg_columns:
                values = np.array(df_interval[band].to_list())  # Converte direttamente in array NumPy
                ptp_values[band] = np.max(values, axis=0) - np.min(values, axis=0)  # Peak-to-Peak per ogni canale

            return ptp_values

        # Definizione degli intervalli per il primo gioco (timestamp ora in secondi)
        game1_intervals = [
            (timestamps["Avvio del primo gioco"], timestamps["primo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["primo iGEQ terminato, gioco ripreso"], timestamps["secondo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio del primo gioco"], timestamps["Chiusura del primo gioco"], [
                (timestamps["primo iGEQ mostrato, gioco messo in pausa"], timestamps["primo iGEQ terminato, gioco ripreso"]),
                (timestamps["secondo iGEQ mostrato, gioco messo in pausa"], timestamps["secondo iGEQ terminato, gioco ripreso"])
            ])
        ]

        # Definizione degli intervalli per il secondo gioco (timestamp ora in secondi)
        game2_intervals = [
            (timestamps["Avvio secondo gioco"], timestamps["terzo iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["terzo iGEQ terminato, gioco ripreso"], timestamps["quarto iGEQ mostrato, gioco messo in pausa"]),
            (timestamps["Avvio secondo gioco"], timestamps["Chiusura secondo gioco"], [
                (timestamps["terzo iGEQ mostrato, gioco messo in pausa"], timestamps["terzo iGEQ terminato, gioco ripreso"]),
                (timestamps["quarto iGEQ mostrato, gioco messo in pausa"], timestamps["quarto iGEQ terminato, gioco ripreso"])
            ])
        ]

        # Calcolo delle Peak-to-Peak per i tre intervalli di ogni gioco
        game1_ptp = [compute_ptp(df_game1_norm, *interval[:2], exclude_intervals=interval[2] if len(interval) > 2 else None) for interval in game1_intervals]
        game2_ptp = [compute_ptp(df_game2_norm, *interval[:2], exclude_intervals=interval[2] if len(interval) > 2 else None) for interval in game2_intervals]

        # Creazione dei DataFrame
        df_game1_ptp = pd.DataFrame(game1_ptp)
        df_game1_ptp.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        df_game2_ptp = pd.DataFrame(game2_ptp)
        df_game2_ptp.insert(0, "Interval", ["1st", "2nd", "Full w/o Pauses"])

        # Salviamo i DataFrame nel dizionario finale
        aggregated_dataframes[participant_id] = {
            "game1_ptp": df_game1_ptp,
            "game2_ptp": df_game2_ptp
        }

    return aggregated_dataframes
