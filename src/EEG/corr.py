import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, combine_pvalues

def create_flow_dataframe(aggregated_amplitudes, df_noto_dict, df_ignoto_dict, method):
    """
    Crea un DataFrame con le informazioni di partecipanti, tipo di gioco, intervallo, punteggio di flow
    e valori dei sensori per ciascuna banda. Include anche il punteggio di flow normalizzato.

    Parametri:
    - aggregated_amplitudes (dict): Dizionario con i DataFrame delle ampiezze EEG per ogni partecipante.
    - df_noto_dict (dict): Dizionario con i dati dei questionari per il gioco noto.
    - df_ignoto_dict (dict): Dizionario con i dati dei questionari per il gioco ignoto.
    - method (str): Metodo di calcolo dell'ampiezza utilizzato.

    Ritorna:
    - DataFrame con i dati formattati secondo la struttura richiesta.
    """    
    # Importa pandas se non è già importato
    import pandas as pd
    import numpy as np
    
    # Configurazione delle mappature
    interval_map = {
        0: "df_selfreport_1",
        1: "df_selfreport_2",
        2: "df_selfreport_final"
    }
    
    interval_to_questionnaire = {
        0: "iGEQ_1",  
        1: "iGEQ_2",  
        2: "GEQ"    
    }

    # Domande relative al flow per ciascun intervallo temporale
    flow_questions = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }
    
    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    sensors = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    
    all_data = []
    
    # Dizionario per raccogliere temporaneamente tutti i punteggi di flow per ogni partecipante
    participant_flow_scores = {}
    
    # Prima passata: raccogli tutti i punteggi di flow per ogni partecipante
    for participant, games in aggregated_amplitudes.items():
        # Determina il gruppo del partecipante (A o B)
        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict
        
        if not (group_A or group_B):
            continue  # Partecipante non trovato nei dizionari
            
        participant_key = f"{'A' if group_A else 'B'}_{participant}"
        
        # Per gruppo B, il gioco noto è game2 ma i questionari sono in df_noto_dict
        game_noto_key = f"game{'1' if group_A else '2'}_{method}"
        game_ignoto_key = f"game{'2' if group_A else '1'}_{method}"
        
        # Recupera i dataframe dei questionari
        df_noto_q = df_noto_dict.get(participant_key)
        df_ignoto_q = df_ignoto_dict.get(participant_key)
        
        if df_noto_q is None or df_ignoto_q is None:
            continue
            
        # Inizializza l'elenco dei punteggi per questo partecipante se non esiste
        if participant not in participant_flow_scores:
            participant_flow_scores[participant] = []
            
        # Per ogni intervallo
        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]
            
            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue
                
            # Dataframe dei questionari per questo intervallo
            df_noto_interval = df_noto_q[interval_name]
            df_ignoto_interval = df_ignoto_q[interval_name]
            
            # Calcola media flow per entrambi i giochi
            flow_noto = calculate_flow_score(df_noto_interval, flow_questions[interval_idx])
            flow_ignoto = calculate_flow_score(df_ignoto_interval, flow_questions[interval_idx])
            
            # Aggiungi i punteggi validi alla lista dei punteggi del partecipante
            if flow_noto is not None:
                participant_flow_scores[participant].append(flow_noto)
            if flow_ignoto is not None:
                participant_flow_scores[participant].append(flow_ignoto)
    
    # Seconda passata: crea il dataframe con i punteggi normalizzati
    for participant, games in aggregated_amplitudes.items():
        # Salta se non abbiamo raccolto punteggi per questo partecipante
        if participant not in participant_flow_scores or len(participant_flow_scores[participant]) < 2:
            continue
            
        # Calcola media e deviazione standard per questo partecipante
        flow_scores = participant_flow_scores[participant]
        mean_flow = sum(flow_scores) / len(flow_scores)
        std_flow = (sum((x - mean_flow) ** 2 for x in flow_scores) / len(flow_scores)) ** 0.5
        
        # Se la deviazione standard è 0, impostiamo un valore minimo per evitare divisione per zero
        if std_flow == 0:
            std_flow = 1e-6
        
        # Determina il gruppo del partecipante (A o B)
        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict
        
        if not (group_A or group_B):
            continue
            
        participant_key = f"{'A' if group_A else 'B'}_{participant}"
        
        # Per gruppo B, il gioco noto è game2 ma i questionari sono in df_noto_dict
        game_noto_key = f"game{'1' if group_A else '2'}_{method}"
        game_ignoto_key = f"game{'2' if group_A else '1'}_{method}"
        
        # Recupera i dataframe
        df_noto = games.get(game_noto_key)
        df_ignoto = games.get(game_ignoto_key)
        df_noto_q = df_noto_dict.get(participant_key)
        df_ignoto_q = df_ignoto_dict.get(participant_key)
        
        if any(x is None or (isinstance(x, pd.DataFrame) and x.empty) for x in [df_noto, df_ignoto, df_noto_q, df_ignoto_q]):
            continue

        # Per ogni intervallo
        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]
            questionnaire_type = interval_to_questionnaire[interval_idx]
            
            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue
                
            # Dataframe dei questionari per questo intervallo
            df_noto_interval = df_noto_q[interval_name]
            df_ignoto_interval = df_ignoto_q[interval_name]
            
            # Calcola media flow per entrambi i giochi
            flow_noto = calculate_flow_score(df_noto_interval, flow_questions[interval_idx])
            flow_ignoto = calculate_flow_score(df_ignoto_interval, flow_questions[interval_idx])
            
            # Calcola i punteggi normalizzati usando la formula z-score: (x - μ) / σ
            normalized_flow_noto = (flow_noto - mean_flow) / std_flow if flow_noto is not None else None
            normalized_flow_ignoto = (flow_ignoto - mean_flow) / std_flow if flow_ignoto is not None else None
            
            if flow_noto is None and flow_ignoto is None:
                continue
                
            # Per ogni banda
            for banda in bands:
                try:
                    # Ottieni i valori EEG
                    noto_eeg = df_noto[banda].iloc[interval_idx]
                    ignoto_eeg = df_ignoto[banda].iloc[interval_idx]
                    
                    # Crea le righe per il dataframe
                    if flow_noto is not None:
                        row_noto = {
                            "Partecipant_ID": participant,
                            "Tipo_Gioco": "Gioco_Noto",
                            "Intervallo": questionnaire_type,
                            "Flow": flow_noto,
                            "Normalized_Flow": normalized_flow_noto,
                            "Banda": banda
                        }
                        # Aggiungi i valori dei sensori
                        for i, sensor in enumerate(sensors):
                            row_noto[sensor] = noto_eeg[i]
                            
                        all_data.append(row_noto)
                        
                    if flow_ignoto is not None:
                        row_ignoto = {
                            "Partecipant_ID": participant,
                            "Tipo_Gioco": "Gioco_Ignoto",
                            "Intervallo": questionnaire_type,
                            "Flow": flow_ignoto,
                            "Normalized_Flow": normalized_flow_ignoto,
                            "Banda": banda
                        }
                        # Aggiungi i valori dei sensori
                        for i, sensor in enumerate(sensors):
                            row_ignoto[sensor] = ignoto_eeg[i]
                            
                        all_data.append(row_ignoto)
                        
                except Exception as e:
                    print(f"Errore con {participant}, intervallo {interval_idx}, banda {banda}: {e}")
                    continue
    
    # Crea e salva il dataframe
    df_all_data = pd.DataFrame(all_data)
    
    # Verifica che ci siano dati prima di salvare
    if not df_all_data.empty:
        df_all_data = df_all_data.sort_values(by=["Partecipant_ID", "Tipo_Gioco"], ascending=[True, False])

        df_all_data.to_csv("df_flow_eeg_data.csv", index=False)
        print(f"Dataframe creato con successo con {len(df_all_data)} righe.")
    else:
        print("ATTENZIONE: Il dataframe risultante è vuoto!")
    
    return df_all_data

def calculate_flow_score(df_interval, questions):
    """Calcola il punteggio medio di flow per un set di domande."""
    scores = []
    for q in questions:
        q_scores = df_interval.loc[df_interval["Domanda"].astype(str) == str(q), "Punteggio"].values
        if len(q_scores) > 0:
            scores.append(q_scores[0])
    
    return sum(scores) / len(scores) if scores else None


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
                
                # Calcola la media delle correlazioni per la regione (e arrotondo a 3 cifre decimali)
                region_corr = region_data["Spearman Corr"].mean()
                row_corr[f"{band} {region}"] = round(region_corr, 3) if not np.isnan(region_corr) else np.nan
                
                # Calcola la media dei p-values per la regione (e arrotondo a 3 cifre decimali)
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
