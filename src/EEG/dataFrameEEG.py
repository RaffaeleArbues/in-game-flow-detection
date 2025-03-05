import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import numpy as np


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
        if not os.path.exists(log_file_path):
            print(f"File di log non trovato per {participant_id}")
            continue
        
        # Legge il file di log e trova i timestamp rilevanti
        with open(log_file_path, 'r') as log_file:
            log_lines = log_file.readlines()
        
        timestamps = {}
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
        
        # Stampa i timestamp estratti
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
        
        # Esclusione degli intervalli di tempo in cui il partecipante sta compilando l'igeq
        df_game1 = df_game1[~df_game1["Timestamp"].between(ts_first_igeq_start, ts_first_igeq_end)]
        df_game1 = df_game1[~df_game1["Timestamp"].between(ts_second_igeq_start, ts_second_igeq_end)]
        df_game1 = df_game1[~df_game1["Timestamp"].between(ts_third_igeq_start, ts_third_igeq_end)]
        df_game1 = df_game1[~df_game1["Timestamp"].between(ts_fourth_igeq_start, ts_fourth_igeq_end)]

        df_game2 = df_game2[~df_game2["Timestamp"].between(ts_first_igeq_start, ts_first_igeq_end)]
        df_game2 = df_game2[~df_game2["Timestamp"].between(ts_second_igeq_start, ts_second_igeq_end)]
        df_game2 = df_game2[~df_game2["Timestamp"].between(ts_third_igeq_start, ts_third_igeq_end)]
        df_game2 = df_game2[~df_game2["Timestamp"].between(ts_fourth_igeq_start, ts_fourth_igeq_end)]



        # Stampa il numero di righe nei segmenti per verificare
        '''
        print(f"{participant_id}: video1={len(df_video1)}, game1={len(df_game1)}, game2={len(df_game2)}")

        segmented_dataframes[participant_id] = [df_video1, df_game1, df_game2]
        
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
        '''
    
    return segmented_dataframes
