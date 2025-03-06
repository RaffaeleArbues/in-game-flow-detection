import src.EEG.preProcessing as pp
import src.EEG.dataFrameEEG as dfe
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

    
def main():
    path_to_neurosity = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//EEG"
    log_file_path = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//log"

    # Preprocessing: Estrazione e correzione dei file JSON
    print("Inizio il preprocessing dei file ZIP...")
    processed_data = pp.extract_and_fix_json_files(path_to_neurosity)
    print("Preprocessing completato!")
    print(processed_data)
   
    df_power_by_band = dfe.create_power_by_band_dataframes(processed_data)

    segmented_dataframes = dfe.split_dataframes(df_power_by_band, log_file_path)


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
