import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, combine_pvalues

def spearman_corr_with_p(aggregated_amplitudes, df_noto_dict, df_ignoto_dict, method):
    """
    Calcola la correlazione di Spearman tra le ampiezze EEG aggregate e le risposte ai questionari.
    Il calcolo della correlazione viene effettuato separatamente per ogni banda, sensore e domanda.

    Parametri:
    - aggregated_amplitudes (dict): Dizionario con i DataFrame delle ampiezze EEG per ogni partecipante.
    - df_noto_dict (dict): Dizionario con i dati dei questionari per il gioco noto.
    - df_ignoto_dict (dict): Dizionario con i dati dei questionari per il gioco ignoto.
    - method (str): Metodo di calcolo dell'ampiezza utilizzato.

    Ritorna:
    - DataFrame con i risultati della correlazione per ogni combinazione di banda, sensore e domanda.
    """    

    results = []  # Lista per salvare i risultati della correlazione
    
    # Mappa degli intervalli temporali ai rispettivi DataFrame dei questionari
    interval_map = {
        0: "df_selfreport_1",  # Primo intervallo (inizio gioco - 1 interruzione)
        1: "df_selfreport_2",  # Secondo intervallo (tra 1 e 2 interruzione)
        2: "df_selfreport_final"  # Terzo intervallo (dopo la 2 interruzione - fine gioco)
    }
    
    # Associa ogni intervallo al tipo di questionario somministrato
    interval_to_questionnaire = {
        0: "iGEQ",  
        1: "iGEQ",  
        2: "GEQ"    
    }

    # Bande EEG considerate nell'analisi
    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    
    # Sensori EEG utilizzati (8 elettrodi)
    sensors = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    
    # Domande dei questionari per ciascun intervallo temporale
    domande_per_intervallo = {
        0: [5, 10],  # Domande per il primo intervallo
        1: [5, 10],  # Domande per il secondo intervallo
        2: [5, 13, 25, 28, 31]  # Domande per il terzo intervallo
    }
    
    all_data = []  # Lista per raccogliere i dati da correlare
    
    # Itera su ogni partecipante nei dati EEG aggregati
    for participant, games in aggregated_amplitudes.items():
        # Determina se il partecipante appartiene al gruppo "A" o "B"
        if f"A_{participant}" in df_noto_dict:
            noto_dict = df_noto_dict  # Il dizionario "noto" è quello associato ad "A"
            ignoto_dict = df_ignoto_dict  # Il dizionario "ignoto" è quello associato ad "A"
            participant_key = f"A_{participant}"
            game_noto = f"game1_{method}"  # Il primo gioco è quello noto
            game_ignoto = f"game2_{method}"  # Il secondo gioco è quello ignoto
        elif f"B_{participant}" in df_noto_dict:
            noto_dict = df_ignoto_dict  # In questo caso, il gioco noto è il secondo
            ignoto_dict = df_noto_dict  # Il gioco ignoto è il primo
            participant_key = f"B_{participant}"
            game_noto = f"game2_{method}"
            game_ignoto = f"game1_{method}"
        else:
            continue  # Se il partecipante non è nei dizionari, viene saltato

        # Recupera i DataFrame EEG e questionari corrispondenti
        df_noto = games.get(game_noto)
        df_ignoto = games.get(game_ignoto)
        df_noto_q = noto_dict.get(participant_key)
        df_ignoto_q = ignoto_dict.get(participant_key)

        # Se mancano dati EEG o questionari per un partecipante, si salta l'iterazione
        if df_noto is None or df_ignoto is None or df_noto_q is None or df_ignoto_q is None:
            continue

        # Itera su ciascun intervallo temporale (0, 1, 2)
        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]  # Nome dell'intervallo
            questionnaire_type = interval_to_questionnaire[interval_idx]  # Tipo di questionario associato
            
            # Controlla se i dati del questionario sono disponibili per questo intervallo
            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue

            df_noto_q_interval = df_noto_q[interval_name]  # Estrai il DataFrame corrispondente al gioco noto
            df_ignoto_q_interval = df_ignoto_q[interval_name]  # Estrai il DataFrame corrispondente al gioco ignoto
            
            # Itera su ciascuna banda EEG
            for banda in bands:
                try:
                    # Estrai i valori EEG per l'intervallo corrente
                    noto_eeg_values = df_noto[banda].iloc[interval_idx]
                    ignoto_eeg_values = df_ignoto[banda].iloc[interval_idx]
                except:
                    continue  # Salta se l'accesso ai dati EEG fallisce

                # Itera su ciascun sensore EEG
                for sensor_idx, sensore in enumerate(sensors):
                    # Itera sulle domande del questionario per l'intervallo corrente
                    for domanda in domande_per_intervallo[interval_idx]:
                        try:
                            # Estrai il valore EEG per il sensore corrente
                            noto_sensor_value = noto_eeg_values[sensor_idx]
                            ignoto_sensor_value = ignoto_eeg_values[sensor_idx]
                        except:
                            continue  # Salta se non riesce ad accedere ai dati EEG per il sensore

                        # Estrai i punteggi del questionario per la domanda corrente
                        noto_score = df_noto_q_interval.loc[
                            (df_noto_q_interval["Domanda"].astype(str) == str(domanda)), 
                            "Punteggio"
                        ].values
                        
                        ignoto_score = df_ignoto_q_interval.loc[
                            (df_ignoto_q_interval["Domanda"].astype(str) == str(domanda)), 
                            "Punteggio"
                        ].values

                        # Salta se non sono presenti punteggi validi
                        if len(noto_score) == 0 or len(ignoto_score) == 0:
                            continue

                        # Aggiunge i dati alla lista per l'analisi della correlazione
                        all_data.append([interval_name, banda, sensore, f"{questionnaire_type}_Q{domanda}", noto_sensor_value, noto_score[0]])
                        all_data.append([interval_name, banda, sensore, f"{questionnaire_type}_Q{domanda}", ignoto_sensor_value, ignoto_score[0]])
    
    # Se sono stati raccolti dati, si procede con la correlazione
    if all_data:
        df_all_data = pd.DataFrame(all_data, columns=["Intervallo", "Banda", "Sensore", "Domanda", "EEG_Valore", "Punteggio"])
        df_all_data.to_csv("df_aggregated_ptp.csv", index=False)
        
        # Calcola la correlazione di Spearman per ogni combinazione di banda, sensore e domanda
        for (banda, sensore, domanda), group in df_all_data.groupby(["Banda", "Sensore", "Domanda"]):
            spearman_corr, p_value = spearmanr(group["EEG_Valore"], group["Punteggio"])
            results.append(["Aggregato", banda, sensore, domanda, spearman_corr, p_value, len(group)])
    
    # Creazione del DataFrame con i risultati della correlazione
    results_df = pd.DataFrame(results, columns=["Gioco", "Banda", "Sensore", "Domanda", "Spearman Corr", "p-value", "Num_Samples"])
    
    return results_df

def generate_correlation_table(correlation_df, output_corr_file, output_pval_file):
    """
    Genera una tabella aggregata delle correlazioni di Spearman tra i gruppi di sensori EEG e le risposte ai questionari,
    ed esporta i risultati sia per le correlazioni che per i p-values.
    
    Parametri:
    - correlation_df (DataFrame): DataFrame contenente i dati di correlazione unificati.
    - output_corr_file (str): Nome del file CSV per le correlazioni.
    - output_pval_file (str): Nome del file CSV per i p-values.
    
    Ritorna:
    - DataFrame con le correlazioni aggregate per ogni domanda e regione cerebrale.
    """
    # Definizione delle aree cerebrali
    sensor_groups = {
        "Frontale Left": ["F5"],
        "Frontale Right": ["F6"],
        "Centrale Left": ["C3", "CP3"],
        "Centrale Right": ["C4", "CP4"],
        "Occipitale Left": ["PO3"],
        "Occipitale Right": ["PO4"]
    }
    
    # Bande di frequenza
    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    
    # Domande da includere nella tabella
    domande_selezionate = ["iGEQ_Q5", "iGEQ_Q10", "GEQ_Q5", "GEQ_Q13", "GEQ_Q25", "GEQ_Q28", "GEQ_Q31"]
    
    table_corr_data = []  # Lista per salvare i dati della tabella delle correlazioni
    table_pval_data = []  # Lista per salvare i dati della tabella dei p-values
    
    for domanda in domande_selezionate:
        row_corr = {"Domanda": domanda}
        row_pval = {"Domanda": domanda}
        
        for band in bands:
            for region, sensors in sensor_groups.items():
                # Filtra i dati per banda, sensore e domanda
                region_data = correlation_df[
                    (correlation_df["Banda"] == band) & 
                    (correlation_df["Sensore"].isin(sensors)) & 
                    (correlation_df["Domanda"] == domanda)
                ]
                
                if region_data.empty:
                    row_corr[f"{band} {region}"] = np.nan
                    row_pval[f"{band} {region}"] = np.nan
                    continue
                
                # Calcola la media delle correlazioni per la regione
                region_corr = region_data["Spearman Corr"].mean()
                row_corr[f"{band} {region}"] = round(region_corr, 3) if not np.isnan(region_corr) else np.nan
                
                # Calcola la media dei p-values per la regione
                region_pval = region_data["p-value"].mean()
                row_pval[f"{band} {region}"] = round(region_pval, 3) if not np.isnan(region_pval) else np.nan
        
        table_corr_data.append(row_corr)
        table_pval_data.append(row_pval)
    
    # Creazione DataFrame per correlazioni e p-values
    table_corr_df = pd.DataFrame(table_corr_data)
    table_pval_df = pd.DataFrame(table_pval_data)
    
    if table_corr_df.empty:
        print("ERRORE: Nessun dato valido trovato per la tabella di correlazione.")
        return None
    
    # **Aggiunta della riga "Flow"** come media delle domande per entrambi i file
    flow_corr_row = {"Domanda": "Flow"}
    flow_pval_row = {"Domanda": "Flow"}
    for col in table_corr_df.columns[1:]:  # Escludiamo la colonna "Domanda"
        flow_corr_row[col] = round(table_corr_df[col].mean(), 3) if not np.isnan(table_corr_df[col].mean()) else np.nan
        flow_pval_row[col] = round(table_pval_df[col].mean(), 3) if not np.isnan(table_pval_df[col].mean()) else np.nan
    
    # **Aggiungiamo la riga "Flow" alla tabella**
    table_corr_df = pd.concat([table_corr_df, pd.DataFrame([flow_corr_row])], ignore_index=True)
    table_pval_df = pd.concat([table_pval_df, pd.DataFrame([flow_pval_row])], ignore_index=True)
    
    # Esporta in CSV
    table_corr_df.to_csv(output_corr_file, index=False)
    table_pval_df.to_csv(output_pval_file, index=False)
    print(f"Esportata tabella delle correlazioni: {output_corr_file}")
    print(f"Esportata tabella dei p-values: {output_pval_file}")
    
    return table_corr_df


def filter_significant_correlations(corr_file, pval_file, output_file, threshold=0.10):
    """
    Filtra un file di correlazioni, mantenendo solo le righe e colonne in cui almeno un valore di p-value
    è inferiore alla soglia specificata.
    
    Parametri:
    - corr_file (str): Path del file CSV contenente la matrice di correlazione.
    - pval_file (str): Path del file CSV contenente la matrice di p-values.
    - output_file (str): Path del file CSV in cui salvare i risultati filtrati.
    - threshold (float): Soglia di significatività per il p-value (default = 0.10).
    
    Ritorna:
    - None: Salva il risultato in output_file.
    """
    # Carica i dati
    corr_df = pd.read_csv(corr_file, index_col=0)
    pval_df = pd.read_csv(pval_file, index_col=0)
    
    # Maschera: True se almeno un p-value nella riga è sotto il threshold
    row_mask = (pval_df < threshold).any(axis=1)
    col_mask = (pval_df < threshold).any(axis=0)
    
    # Filtra righe e colonne
    filtered_corr = corr_df.loc[row_mask, col_mask]
    
    # Salva il file filtrato
    filtered_corr.to_csv(output_file)
    
    print(f"File salvato con successo: {output_file}")
