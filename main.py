import src.EEG.preProcessing as pp
import src.EEG.dataFrameEEG as dfe
import src.EEG.corr as corrEEG
import src.physiological.corr as corrPhys
import src.questionnaire.dataFrameQuest as quest
import src.physiological.dataFramePhysiological as phys
import src.facialAnalysis.dataFrameFacial as face
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
    
def main():
    path_to_neurosity = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//EEG"
    path_to_physiological = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//physiological_separated"
    path_to_video = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//video"
    log_file_path = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//log"
    path_to_quest = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//questionnaire"
    output_video = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//video//splittedIntervals"

    # PROCESSING EEG DATA
    '''
    # Preprocessing: Estrazione e correzione dei file JSON
    print("Inizio il preprocessing dei file ZIP...")
    processed_data = pp.extract_and_fix_json_files(path_to_neurosity)
    print("Preprocessing completato!")
    print(processed_data)
    # df_power_by_band è un dizionario che ha come chiavi i "partecipant_id" e associati a ciascuna delle chiavi un dataframe con i dati EEG 
    df_power_by_band = dfe.create_power_by_band_dataframes(processed_data)
    # segmented_dataframes è un dizionario che ha come chiavi i "partecipant_id" e associati 3 dataframe corrispondenti ai dati della baseline e i dati dei due giochi giocati
    segmented_dataframes = dfe.split_dataframes(df_power_by_band, log_file_path)
    # normalized_segmented_dataframes è un dizionario che ha come chiavi i "partecipant_id" e associati i 2 dataframe con i dati dei due giochi normalizzati
    normalized_segmented_dataframes = dfe.normalize_eeg(segmented_dataframes)

    # gli aggregati sono dizinoari che hanno come chiavi i "partecipant_id" e associati due dataframe per gioco con le seguenti righe:
    # "1st", "2nd", "Full w/o Pauses", dove sono contenute le ampiezze, calcolate con RMS e PTP
    aggregated_rms = dfe.compute_aggregated_rms_amplitudes(normalized_segmented_dataframes, log_file_path)
    aggregated_ptp = dfe.compute_aggregated_ptp_amplitudes(normalized_segmented_dataframes, log_file_path)

    # dizionari con le chiavi che sono i "partecipant_id" e associati i dizionari df_selfreport_1, df_selfreport_2, df_selfreport_final (uno per i self del gioco noto, l'altro per il gioco ignoto).
    df_noto_dict, df_ignoto_dict = quest.carica_questionari(path_to_quest)
    #print(aggregated_rms)

    # Estrazione dei dataframe finali -> eeg con accanto i dati self report corrispondenti. (uno per le ampiezze ptp e l'altro per le ampiezze rms).
    df_flow_ptp = corrEEG.create_EEG_flow_dataframe(aggregated_ptp, df_noto_dict, df_ignoto_dict, "ptp")
    df_flow_rms = corrEEG.create_EEG_flow_dataframe(aggregated_rms, df_noto_dict, df_ignoto_dict, "rms")
    
    # Calcolo dei mixed models
    corrEEG.run_mixed_models(df_flow_rms)

    #correlation_ptp.to_csv("df_spearman_ptp.csv", index=False)

    #print(correlation_rms)

    # Calcolo delle correlazioni per Peak-to-Peak
    #correlation_bs = corrEEG.spearman_corr_with_p(aggregated_bs, df_noto_dict, df_ignoto_dict, "ptp")

    # Calcolo delle correlazioni per Band-Specific
    #correlation_band = corrEEG.spearman_corr_with_p(aggregated_bs, df_noto_dict, df_ignoto_dict, "band")

    #corrEEG.generate_correlation_table(correlation_ptp, "correlation_table.csv", "pvalue_table.csv")
    '''
    
    # PROCESSING PERIFERIC PHYSIOLOGICAL DATA
    '''
    # segmented_dataframes è un dizionario che ha come chiavi i "partecipant_id" e associati 3 dataframe corrispondenti ai dati della baseline e i dati dei due giochi giocati
    segmented_dataframes_eda, segmented_dataframes_bvp = phys.split_dataframes(path_to_physiological, log_file_path)

    # dopo aver segmentato e filtrato i dati, estraggo la tonica e la fasica dall'eda
    segmented_eda_pt = phys.separate_eda_components(segmented_dataframes_eda)
    
    # Normalizzo tutto (sia eda che bvp)
    norm_eda, norm_bvp = phys.normalize_physio_dataframes(segmented_eda_pt, segmented_dataframes_bvp)
    
    # calcolo l'HR dai dati bvp normalizzati
    norm_bvp_hr =  phys.calculate_heart_rate(norm_bvp)

    # calcolo le metriche per EDA
    eda_metrics = phys.extract_eda_metrics(norm_eda, log_file_path)
    bvp_metrics = phys.extract_bvp_metrics(norm_bvp_hr, log_file_path)
    #print(bvp_metrics["Rocco_Mennea"][0])  # DataFrame del primo gioco
    #print(bvp_metrics["Rocco_Mennea"][1])  # DataFrame del secondo gioco
    df_noto_dict, df_ignoto_dict = quest.carica_questionari(path_to_quest)
    df_flow_phys = corrPhys.create_physiological_flow_dataframe(eda_metrics, bvp_metrics, df_noto_dict, df_ignoto_dict)
    # Calcolo dei mixed models
    corrPhys.run_physiological_mixed_model(df_flow_phys)
    
    #segmented_hrv = phys.calculate_hrv_per_segment(segmented_dataframes_bvp_hr)
    
    #segmented_physio_dataframes_hr = phys.calculate_heart_rate(segmented_physio_dataframes)
    #normalized_physio_dataframesphys = phys.normalize_physio_dataframes(segmented_physio_dataframes_hr)
    #print(normalized_physio_dataframesphys)

    # Scegli un participant_id da esportare
    participant_id = "Rocco_Mennea"  # Sostituiscilo con il nome corretto

    if participant_id in norm_bvp_hr:
        df_game1_normalized_bvp, df_game2_normalized_bvp = norm_bvp_hr[participant_id]

        # Esporta in CSV
        df_game1_normalized_bvp.to_csv(f"{participant_id}_game1_bvp_hr.csv", index=False)
        df_game2_normalized_bvp.to_csv(f"{participant_id}_game2_bvp_hr.csv", index=False)

        print(f"Dati di {participant_id} esportati correttamente!")
    else:
        print(f"Partecipante {participant_id} non trovato nel dataset.")
    
    if participant_id in norm_eda:
        df_game1_eda, df_game2_eda = norm_eda[participant_id]

        # Esporta in CSV
        df_game1_eda.to_csv(f"{participant_id}_game1_eda_pt.csv", index=False)
        df_game2_eda.to_csv(f"{participant_id}_game2_eda_pt.csv", index=False)

        print(f"Dati di {participant_id} esportati correttamente!")
    else:
        print(f"Partecipante {participant_id} non trovato nel dataset.")
    '''

    # Seleziona i file
    #face.cut_ffmpeg_segments(path_to_video, log_file_path, output_video)

    # Opzionale: percorso dove salvare i risultati
    output_folder = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//video//resultsFex"

    # Esegui l'analisi
    results = face.analyze_participant_videos(output_video, output_folder)

    
if __name__ == "__main__":
    main()
