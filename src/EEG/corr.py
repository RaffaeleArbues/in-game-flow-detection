import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, kendalltau, combine_pvalues

def kendall_corr_with_p(aggregated_amplitudes, df_noto_dict, df_ignoto_dict, method):
    """
    Calcola la correlazione di Kendall tra le ampiezze EEG aggregate e le risposte ai questionari.
    Supporta diversi tipi di ampiezza (RMS, Peak-to-Peak, Band-Specific).

    Parametri:
    - aggregated_amplitudes (dict): Dizionario con i DataFrame delle ampiezze EEG per ogni partecipante.
    - df_noto_dict (dict): Dizionario con i dati dei questionari per il gioco noto.
    - df_ignoto_dict (dict): Dizionario con i dati dei questionari per il gioco ignoto.
    - amplitude_type (str): Tipo di ampiezza da utilizzare ("RMS", "Peak-to-Peak", "Band-Specific").

    Ritorna:
    - DataFrame con i risultati della correlazione (Kendall Tau e p-value).
    """
    results = []
    
    interval_map = {
        0: "df_selfreport_1",
        1: "df_selfreport_2",
        2: "df_selfreport_final"
    }
    
    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    sensors = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    
    domande_per_intervallo = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }
    
    eeg_data = {"Noto": {}, "Ignoto": {}}
    questionario_data = {"Noto": {}, "Ignoto": {}}
    
    for gioco in ["Noto", "Ignoto"]:
        for banda in bands:
            for sensore in sensors:
                for i in range(3):
                    key = (banda, sensore, interval_map[i])
                    eeg_data[gioco][key] = []
                    questionario_data[gioco][key] = {domanda: [] for domanda in domande_per_intervallo[i]}
    
    for participant, games in aggregated_amplitudes.items():
        if f"A_{participant}" in df_noto_dict:
            noto_dict = df_noto_dict
            ignoto_dict = df_ignoto_dict
            game_noto = f"game1_{method}"
            game_ignoto = f"game2_{method}"
        elif f"B_{participant}" in df_noto_dict:
            noto_dict = df_ignoto_dict
            ignoto_dict = df_noto_dict
            game_noto = f"game2_{method}"
            game_ignoto = f"game1_{method}"
        else:
            print(f"Partecipante {participant} non trovato nei dizionari dei questionari.")
            continue
    
        df_noto = games.get(game_noto)
        df_ignoto = games.get(game_ignoto)
    
        df_noto_q = noto_dict.get(f"A_{participant}", noto_dict.get(f"B_{participant}"))
        df_ignoto_q = ignoto_dict.get(f"A_{participant}", ignoto_dict.get(f"B_{participant}"))
    
        if df_noto is None or df_ignoto is None:
            print(f"Partecipante {participant}: dati EEG mancanti per {method}.")
            continue
    
        for i in range(3):
            df_noto_q_sub = df_noto_q[interval_map[i]]
            df_ignoto_q_sub = df_ignoto_q[interval_map[i]]
    
            if df_noto_q_sub.empty or df_ignoto_q_sub.empty:
                continue
    
            for banda in bands:
                noto_values = df_noto[banda].iloc[i]
                ignoto_values = df_ignoto[banda].iloc[i]
    
                for j, sensore in enumerate(sensors):
                    key = (banda, sensore, interval_map[i])
    
                    eeg_data["Noto"][key].append(noto_values[j])
                    eeg_data["Ignoto"][key].append(ignoto_values[j])
    
                    for domanda in domande_per_intervallo[i]:
                        tipo_questionario = "iGEQ" if i < 2 else "GEQ"
                        domanda_label = f"{tipo_questionario}_Q{domanda}"
    
                        noto_scores = df_noto_q_sub[(df_noto_q_sub["Domanda"] == domanda) & (df_noto_q_sub["Tipo_Questionario"] == tipo_questionario)]["Punteggio"]
                        ignoto_scores = df_ignoto_q_sub[(df_ignoto_q_sub["Domanda"] == domanda) & (df_ignoto_q_sub["Tipo_Questionario"] == tipo_questionario)]["Punteggio"]
    
                        if domanda_label not in questionario_data["Noto"][key]:
                            questionario_data["Noto"][key][domanda_label] = []
                        if domanda_label not in questionario_data["Ignoto"][key]:
                            questionario_data["Ignoto"][key][domanda_label] = []
    
                        if not noto_scores.empty:
                            questionario_data["Noto"][key][domanda_label].extend(noto_scores.to_list())
                        if not ignoto_scores.empty:
                            questionario_data["Ignoto"][key][domanda_label].extend(ignoto_scores.to_list())
    
    for gioco in ["Noto", "Ignoto"]:
        for key, eeg_values in eeg_data[gioco].items():
            banda, sensore, intervallo = key
    
            for domanda, punteggi in questionario_data[gioco][key].items():
                if len(eeg_values) > 1 and len(punteggi) > 1:
                    tau, p_value = kendalltau(eeg_values, punteggi)
                    results.append([gioco, intervallo, banda, sensore, domanda, tau, p_value])
    
    results_df = pd.DataFrame(results, columns=["Gioco", "Intervallo", "Banda", "Sensore", "Domanda", "Kendall Tau", "p-value"])
    
    return results_df


def spearman_corr_with_p(aggregated_amplitudes, df_noto_dict, df_ignoto_dict, method):
    """
    Calcola la correlazione di Spearman tra le ampiezze EEG aggregate e le risposte ai questionari.
    Supporta diversi tipi di ampiezza (RMS, Peak-to-Peak, Band-Specific).

    Parametri:
    - aggregated_amplitudes (dict): Dizionario con i DataFrame delle ampiezze EEG per ogni partecipante.
    - df_noto_dict (dict): Dizionario con i dati dei questionari per il gioco noto.
    - df_ignoto_dict (dict): Dizionario con i dati dei questionari per il gioco ignoto.
    - amplitude_type (str): Tipo di ampiezza da utilizzare ("RMS", "Peak-to-Peak", "Band-Specific").

    Ritorna:
    - DataFrame con i risultati della correlazione (Spearman e p-value).
    """
    results = []

    interval_map = {
        0: "df_selfreport_1",
        1: "df_selfreport_2",
        2: "df_selfreport_final"
    }

    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    sensors = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']

    domande_per_intervallo = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }

    eeg_data = {"Noto": {}, "Ignoto": {}}
    questionario_data = {"Noto": {}, "Ignoto": {}}

    for gioco in ["Noto", "Ignoto"]:
        for banda in bands:
            for sensore in sensors:
                for i in range(3):
                    key = (banda, sensore, interval_map[i])
                    eeg_data[gioco][key] = []
                    questionario_data[gioco][key] = {domanda: [] for domanda in domande_per_intervallo[i]}

    for participant, games in aggregated_amplitudes.items():
        if f"A_{participant}" in df_noto_dict:
            noto_dict = df_noto_dict
            ignoto_dict = df_ignoto_dict
            game_noto = f"game1_{method}"
            game_ignoto = f"game2_{method}"
        elif f"B_{participant}" in df_noto_dict:
            noto_dict = df_ignoto_dict
            ignoto_dict = df_noto_dict
            game_noto = f"game2_{method}"
            game_ignoto = f"game1_{method}"
        else:
            print(f"Partecipante {participant} non trovato nei dizionari dei questionari.")
            continue

        df_noto = games.get(game_noto)
        df_ignoto = games.get(game_ignoto)

        df_noto_q = noto_dict.get(f"A_{participant}", noto_dict.get(f"B_{participant}"))
        df_ignoto_q = ignoto_dict.get(f"A_{participant}", ignoto_dict.get(f"B_{participant}"))

        if df_noto is None or df_ignoto is None:
            print(f"Partecipante {participant}: dati EEG mancanti per {method}.")
            continue

        for i in range(3):
            df_noto_q_sub = df_noto_q[interval_map[i]]
            df_ignoto_q_sub = df_ignoto_q[interval_map[i]]

            if df_noto_q_sub.empty or df_ignoto_q_sub.empty:
                continue

            for banda in bands:
                noto_values = df_noto[banda].iloc[i]
                ignoto_values = df_ignoto[banda].iloc[i]

                for j, sensore in enumerate(sensors):
                    key = (banda, sensore, interval_map[i])

                    eeg_data["Noto"][key].append(noto_values[j])
                    eeg_data["Ignoto"][key].append(ignoto_values[j])

                    for domanda in domande_per_intervallo[i]:
                        # Determina il tipo di questionario in base all'intervallo
                        if i < 2:
                            tipo_questionario = "iGEQ"  # Intervalli 0 e 1 → iGEQ
                        else:
                            tipo_questionario = "GEQ"   # Intervallo 2 → GEQ

                        # Creiamo una chiave unica per la domanda con il tipo di questionario
                        domanda_label = f"{tipo_questionario}_Q{domanda}"

                        # Filtriamo per il tipo di questionario e domanda
                        noto_scores = df_noto_q_sub[(df_noto_q_sub["Domanda"] == domanda) & (df_noto_q_sub["Tipo_Questionario"] == tipo_questionario)]["Punteggio"]
                        ignoto_scores = df_ignoto_q_sub[(df_ignoto_q_sub["Domanda"] == domanda) & (df_ignoto_q_sub["Tipo_Questionario"] == tipo_questionario)]["Punteggio"]

                        # Assicuriamoci che la chiave esista nel dizionario
                        if domanda_label not in questionario_data["Noto"][key]:
                            questionario_data["Noto"][key][domanda_label] = []
                        if domanda_label not in questionario_data["Ignoto"][key]:
                            questionario_data["Ignoto"][key][domanda_label] = []

                        # Aggiungiamo i dati nel dizionario mantenendo la distinzione tra questionari
                        if not noto_scores.empty:
                            questionario_data["Noto"][key][domanda_label].extend(noto_scores.to_list())
                        if not ignoto_scores.empty:
                            questionario_data["Ignoto"][key][domanda_label].extend(ignoto_scores.to_list())



    for gioco in ["Noto", "Ignoto"]:
        for key, eeg_values in eeg_data[gioco].items():
            banda, sensore, intervallo = key

            for domanda, punteggi in questionario_data[gioco][key].items():
                print(f"{gioco} - {banda} - {sensore} - {intervallo} - Domanda {domanda}: {len(eeg_values)} valori EEG, {len(punteggi)} punteggi")
                if len(eeg_values) > 1 and len(punteggi) > 1:
                    spearman_corr, p_value = spearmanr(eeg_values, punteggi)
                    results.append([gioco, intervallo, banda, sensore, domanda, spearman_corr, p_value])

    results_df = pd.DataFrame(results, columns=["Gioco", "Intervallo", "Banda", "Sensore", "Domanda", "Spearman Corr", "p-value"])

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
                    region_corr = region_data["Kendall Tau"].mean()
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
