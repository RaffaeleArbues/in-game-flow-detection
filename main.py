import src.EEG.preProcessing as pp
import src.EEG.dataFrameEEG as dfe
import src.EEG.corr as corr
import src.questionnaire.dataFrameQuest as quest
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
    
def main():
    path_to_neurosity = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//EEG"
    log_file_path = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//log"
    path_to_quest = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//questionnaire"

    # Preprocessing: Estrazione e correzione dei file JSON
    print("Inizio il preprocessing dei file ZIP...")
    processed_data = pp.extract_and_fix_json_files(path_to_neurosity)
    print("Preprocessing completato!")
    print(processed_data)
   
    df_power_by_band = dfe.create_power_by_band_dataframes(processed_data)

    segmented_dataframes = dfe.split_dataframes(df_power_by_band, log_file_path)

    normalized_segmented_dataframes = dfe.normalize_eeg(segmented_dataframes)

    # aggregati con i tre tipi di calcolo delle ampiezze
    aggregated_rms = dfe.compute_aggregated_rms_amplitudes(normalized_segmented_dataframes, log_file_path)
    aggregated_ptp = dfe.compute_aggregated_ptp_amplitudes(normalized_segmented_dataframes, log_file_path)

    df_noto_dict, df_ignoto_dict = quest.carica_questionari(path_to_quest)
    print(aggregated_rms)


    # Calcolo delle correlazioni per RMS
    df_flow = corr.create_flow_dataframe(aggregated_rms, df_noto_dict, df_ignoto_dict, "rms")
    #correlation_ptp.to_csv("df_spearman_ptp.csv", index=False)

    #print(correlation_rms)

    # Calcolo delle correlazioni per Peak-to-Peak
    #correlation_bs = corr.spearman_corr_with_p(aggregated_bs, df_noto_dict, df_ignoto_dict, "ptp")

    # Calcolo delle correlazioni per Band-Specific
    #correlation_band = corr.spearman_corr_with_p(aggregated_bs, df_noto_dict, df_ignoto_dict, "band")

    #corr.generate_correlation_table(correlation_ptp, "correlation_table.csv", "pvalue_table.csv")

    '''
    correlation_file = "C://Users//raffa//OneDrive//Desktop//Tesi//PLOTS//CORR_EEG//Intervalli cambiati//rms//correlation_table_ignoto.csv"
    pvalue_file = "C://Users//raffa//OneDrive//Desktop//Tesi//PLOTS//CORR_EEG//Intervalli cambiati//rms//correlation_table_ignoto_pvalues.csv"
    corr.filter_significant_correlations(correlation_file, pvalue_file, "filtered_correlation.csv")

    
    # Salvare ogni DataFrame normalizzato come CSV
    for participant_id, dfs in normalized_segmented_dataframes.items():
        df_game1_norm, df_game2_norm = dfs

        # Creare i percorsi dei file
        file_game1 = os.path.join(output_dir, f"{participant_id}_game1_normalized.csv")
        file_game2 = os.path.join(output_dir, f"{participant_id}_game2_normalized.csv")

        # Salvare i file CSV
        df_game1_norm.to_csv(file_game1, index=False)
        df_game2_norm.to_csv(file_game2, index=False)
    '''


    '''
    output_dir = "output_segmented_csv"
    os.makedirs(output_dir, exist_ok=True)  # Crea la cartella se non esiste

    for participant_id, segments in segmented_dataframes.items():
        segment_names = ["video1", "game1", "game2"]
        
        for segment, name in zip(segments, segment_names):
            file_path = os.path.join(output_dir, f"{participant_id}_{name}.csv")
            segment.to_csv(file_path, index=False)  # Salva il CSV senza indice
            print(f"Salvato: {file_path}")
    '''
if __name__ == "__main__":
    main()
