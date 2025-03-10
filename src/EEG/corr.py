import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, kendalltau, combine_pvalues

def spearman_corr_with_p(aggregated_amplitudes, df_noto_dict, df_ignoto_dict, method):
    """
    Calcola la correlazione di Spearman tra le ampiezze EEG aggregate e le risposte ai questionari.
    
    Parametri:
    - aggregated_amplitudes (dict): Dizionario con i DataFrame delle ampiezze EEG per ogni partecipante.
    - df_noto_dict (dict): Dizionario con i dati dei questionari per il gioco noto.
    - df_ignoto_dict (dict): Dizionario con i dati dei questionari per il gioco ignoto.
    - method (str): Metodo di calcolo dell'ampiezza utilizzato.

    Ritorna:
    - DataFrame con i risultati della correlazione (Spearman e p-value).
    """    
    results = []

    interval_map = {
        0: "df_selfreport_1",
        1: "df_selfreport_2",
        2: "df_selfreport_final"
    }

    # Mappatura degli intervalli ai tipi di questionario
    interval_to_questionnaire = {
        0: "iGEQ",  # Intervallo 0 → iGEQ
        1: "iGEQ",  # Intervallo 1 → iGEQ
        2: "GEQ"    # Intervallo 2 → GEQ
    }

    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    sensors = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    # Domande da considerare per ciascun intervallo
    domande_per_intervallo = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }

    # Strutture dati per raccogliere le correlazioni
    eeg_data = {"Noto": {}, "Ignoto": {}}
    questionario_data = {"Noto": {}, "Ignoto": {}}

    # Log per conteggio partecipanti
    n_participants_processed = 0
    n_participants_skipped = 0
    
    for participant, games in aggregated_amplitudes.items():
        # Determina quale dizionario usare in base al prefisso del partecipante
        if f"A_{participant}" in df_noto_dict:
            noto_dict = df_noto_dict
            ignoto_dict = df_ignoto_dict
            participant_key_noto = f"A_{participant}"
            participant_key_ignoto = f"A_{participant}"
            game_noto = f"game1_{method}"
            game_ignoto = f"game2_{method}"
        elif f"B_{participant}" in df_noto_dict:
            noto_dict = df_ignoto_dict
            ignoto_dict = df_noto_dict
            participant_key_noto = f"B_{participant}"
            participant_key_ignoto = f"B_{participant}"
            game_noto = f"game2_{method}"
            game_ignoto = f"game1_{method}"
        else:
            print(f"Partecipante {participant} non trovato nei dizionari dei questionari.")
            n_participants_skipped += 1
            continue

        # Recupera i dataframes per i giochi noto e ignoto
        df_noto = games.get(game_noto)
        df_ignoto = games.get(game_ignoto)

        # Recupera i dati dei questionari
        df_noto_q = noto_dict.get(participant_key_noto)
        df_ignoto_q = ignoto_dict.get(participant_key_ignoto)

        if df_noto is None or df_ignoto is None:
            print(f"Partecipante {participant}: dati EEG mancanti per {method}.")
            n_participants_skipped += 1
            continue

        if df_noto_q is None or df_ignoto_q is None:
            print(f"Partecipante {participant}: dati questionario mancanti.")
            n_participants_skipped += 1
            continue

        n_participants_processed += 1

        # Processa ogni intervallo temporale
        for interval_idx in range(3):
            questionnaire_type = interval_to_questionnaire[interval_idx]
            interval_name = interval_map[interval_idx]
            
            # Verifica che i dataframe dei questionari esistano e non siano vuoti
            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                print(f"Intervallo {interval_name} mancante per partecipante {participant}")
                continue
                
            df_noto_q_interval = df_noto_q[interval_name]
            df_ignoto_q_interval = df_ignoto_q[interval_name]
            
            if df_noto_q_interval.empty or df_ignoto_q_interval.empty:
                print(f"Dati vuoti per intervallo {interval_name}, partecipante {participant}")
                continue

            # Processa ogni banda EEG
            for banda in bands:
                try:
                    # Estrai i valori delle ampiezze EEG per l'intervallo corrente
                    noto_eeg_values = df_noto[banda].iloc[interval_idx]
                    ignoto_eeg_values = df_ignoto[banda].iloc[interval_idx]
                except IndexError:
                    print(f"Errore indice per banda {banda}, partecipante {participant}, intervallo {interval_idx}")
                    continue
                except Exception as e:
                    print(f"Errore durante l'accesso ai dati EEG: {e}")
                    continue

                # Processa ogni sensore
                for sensor_idx, sensore in enumerate(sensors):
                    # Per ogni domanda nell'intervallo corrente
                    for domanda in domande_per_intervallo[interval_idx]:
                        # Crea una chiave unica per la domanda con il tipo di questionario
                        domanda_label = f"{questionnaire_type}_Q{domanda}"
                        
                        # Estrai il valore EEG per questo sensore
                        try:
                            noto_sensor_value = noto_eeg_values[sensor_idx]
                            ignoto_sensor_value = ignoto_eeg_values[sensor_idx]
                        except IndexError:
                            print(f"Errore indice sensore {sensor_idx} per banda {banda}, partecipante {participant}")
                            continue
                        
                        # Filtra i punteggi del questionario per la domanda corrente
                        try:
                            noto_score = df_noto_q_interval.loc[
                                (df_noto_q_interval["Domanda"].astype(str) == str(domanda)) & 
                                (df_noto_q_interval["Tipo_Questionario"] == questionnaire_type), 
                                "Punteggio"
                            ].values
                            
                            ignoto_score = df_ignoto_q_interval.loc[
                                (df_ignoto_q_interval["Domanda"].astype(str) == str(domanda)) & 
                                (df_ignoto_q_interval["Tipo_Questionario"] == questionnaire_type), 
                                "Punteggio"
                            ].values
                        except Exception as e:
                            print(f"Errore filtro questionari: {e}")
                            continue
                            
                        # Verifica che ci siano punteggi validi
                        if len(noto_score) == 0 or len(ignoto_score) == 0:
                            continue
                            
                        # Prendi solo il primo punteggio (dovrebbe essercene solo uno per domanda)
                        if len(noto_score) > 1:
                            print(f"ATTENZIONE: Più di un punteggio per domanda {domanda} in {questionnaire_type} per {participant} (gioco noto)")
                            noto_score = noto_score[0]
                        else:
                            noto_score = noto_score[0]
                            
                        if len(ignoto_score) > 1:
                            print(f"ATTENZIONE: Più di un punteggio per domanda {domanda} in {questionnaire_type} per {participant} (gioco ignoto)")
                            ignoto_score = ignoto_score[0]
                        else:
                            ignoto_score = ignoto_score[0]
                        
                        # Crea una chiave univoca per questo specifico contesto
                        key_noto = (banda, sensore, interval_name, domanda_label, "Noto", participant)
                        key_ignoto = (banda, sensore, interval_name, domanda_label, "Ignoto", participant)
                        
                        # Aggiungi i dati ai dizionari per la correlazione
                        if key_noto not in eeg_data["Noto"]:
                            eeg_data["Noto"][key_noto] = noto_sensor_value
                            questionario_data["Noto"][key_noto] = noto_score
                            
                        if key_ignoto not in eeg_data["Ignoto"]:
                            eeg_data["Ignoto"][key_ignoto] = ignoto_sensor_value
                            questionario_data["Ignoto"][key_ignoto] = ignoto_score

    print(f"Partecipanti elaborati: {n_participants_processed}")
    print(f"Partecipanti saltati: {n_participants_skipped}")
    
    # Raggruppa i dati per calcolare le correlazioni
    correlation_data = {}
    
    for gioco in ["Noto", "Ignoto"]:
        for key in eeg_data[gioco]:
            banda, sensore, intervallo, domanda_label, _, _ = key
            group_key = (gioco, intervallo, banda, sensore, domanda_label)
            
            if group_key not in correlation_data:
                correlation_data[group_key] = {
                    "eeg_values": [],
                    "questionnaire_scores": []
                }
                
            correlation_data[group_key]["eeg_values"].append(eeg_data[gioco][key])
            correlation_data[group_key]["questionnaire_scores"].append(questionario_data[gioco][key])
    
    # Calcola le correlazioni di Spearman
    for group_key, data in correlation_data.items():
        gioco, intervallo, banda, sensore, domanda = group_key
        
        eeg_values = data["eeg_values"]
        punteggi = data["questionnaire_scores"]
        
        # Verifica che ci siano abbastanza dati per calcolare la correlazione
        if len(eeg_values) > 1 and len(punteggi) > 1:  # Serve più di un dato per la correlazione
            try:
                # Rimuovi eventuali NaN
                valid_indices = ~(np.isnan(eeg_values) | np.isnan(punteggi))
                valid_eeg = [eeg_values[i] for i in range(len(eeg_values)) if valid_indices[i]]
                valid_scores = [punteggi[i] for i in range(len(punteggi)) if valid_indices[i]]
                
                if len(valid_eeg) > 1:  # Ricontrolla dopo aver rimosso i NaN
                    spearman_corr, p_value = spearmanr(valid_eeg, valid_scores)
                    results.append([gioco, intervallo, banda, sensore, domanda, spearman_corr, p_value, len(valid_eeg)])
            except Exception as e:
                print(f"Errore nel calcolo della correlazione: {e} per {group_key}")
    
    # Crea il DataFrame finale dei risultati
    if results:
        results_df = pd.DataFrame(
            results, 
            columns=["Gioco", "Intervallo", "Banda", "Sensore", "Domanda", "Spearman Corr", "p-value", "Num_Samples"]
        )
        
        # Ordina per p-value per evidenziare le correlazioni significative
        results_df = results_df.sort_values(by=["p-value", "Spearman Corr"], ascending=[True, False])
    else:
        results_df = pd.DataFrame(
            columns=["Gioco", "Intervallo", "Banda", "Sensore", "Domanda", "Spearman Corr", "p-value", "Num_Samples"]
        )
        print("ATTENZIONE: Nessun risultato di correlazione calcolato!")

    return results_df


def export_correlation_and_pvalue_tables(correlation_df, output_noto, output_ignoto):
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
    
    # Generiamo i nomi dei file per le correlazioni e i p-values
    output_noto_pval = output_noto.replace(".csv", "_pvalues.csv")
    output_ignoto_pval = output_ignoto.replace(".csv", "_pvalues.csv")
    
    # Creazione delle tabelle per gioco noto e ignoto
    for gioco, output_corr_file, output_pval_file in zip(
        ["Noto", "Ignoto"], 
        [output_noto, output_ignoto], 
        [output_noto_pval, output_ignoto_pval]
    ):
        table_corr_data = []  # Per le correlazioni
        table_pval_data = []  # Per i p-values
        
        # Seleziona solo i dati del gioco attuale
        df_filtered = correlation_df[correlation_df["Gioco"] == gioco]
        
        for domanda in sorted(df_filtered["Domanda"].unique()):
            row_corr = {"Domanda": domanda}
            row_pval = {"Domanda": domanda}
            p_values_storage = {}  # Per raccogliere i p-values per Fisher

            for band in bands:
                for region, sensors in sensor_groups.items():
                    # Seleziona solo le correlazioni per la regione e banda
                    region_data = df_filtered[
                        (df_filtered["Banda"] == band) & 
                        (df_filtered["Sensore"].isin(sensors)) & 
                        (df_filtered["Domanda"] == domanda)
                    ]

                    # Calcola la media delle correlazioni per la regione
                    region_corr = region_data["Spearman Corr"].mean()
                    row_corr[f"{band} {region}"] = round(region_corr, 3) if not np.isnan(region_corr) else np.nan

                    # Raccoglie i p-values per il metodo di Fisher
                    p_values = region_data["p-value"].dropna().values
                    if len(p_values) > 0:
                        p_values_storage[f"{band} {region}"] = p_values

            # Calcola i p-values combinati per ogni banda e regione
            for col, p_vals in p_values_storage.items():
                if len(p_vals) > 1:
                    _, fisher_p = combine_pvalues(p_vals, method='fisher')
                    row_pval[col] = round(fisher_p, 3)  # Approssima il p-value combinato
                else:
                    row_pval[col] = round(p_vals[0], 3) if len(p_vals) == 1 else np.nan  # Se c'è un solo valore, lo usiamo

            table_corr_data.append(row_corr)
            table_pval_data.append(row_pval)

        # Creazione DataFrame per correlazioni
        table_corr_df = pd.DataFrame(table_corr_data)

        # Creazione DataFrame per p-values
        table_pval_df = pd.DataFrame(table_pval_data)

        # **Aggiunta della riga "Flow"** come media delle domande
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

        print(f"Esportata tabella delle correlazioni per {gioco}: {output_corr_file}")
        print(f"Esportata tabella dei p-values per {gioco}: {output_pval_file}")


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
