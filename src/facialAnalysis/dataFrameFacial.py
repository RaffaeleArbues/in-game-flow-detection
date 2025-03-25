import src.EEG.dataFrameEEG as dfe
import os
import pandas as pd
import subprocess
from feat import Detector

def unix_to_seconds(csv_df, target_ts):
    """
    Converte un timestamp Unix in secondi rispetto all'inizio del video.
    Prende il primo timestamp come riferimento (0).
    """
    csv_df_sorted = csv_df.sort_values("Unix_Timestamp")
    start_ts = csv_df_sorted["Unix_Timestamp"].iloc[0]
    return int((target_ts - start_ts) / 1000)

def ffmpeg_cut(input_path, output_path, start_sec, duration):
    cmd = [
        "ffmpeg",
        "-ss", str(start_sec),
        "-i", input_path,
        "-t", str(duration),
        "-c", "copy",  # evita ricodifica
        output_path,
        "-y"  # sovrascrive
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def cut_ffmpeg_segments(video_dir, log_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(video_dir):
        if not file.endswith(".avi"):
            continue

        participant_id = file.replace(".avi", "")
        video_path = os.path.join(video_dir, file)
        csv_path = os.path.join(video_dir, f"{participant_id}.csv")
        log_path = os.path.join(log_dir, f"{participant_id}.txt")

        if not os.path.exists(csv_path) or not os.path.exists(log_path):
            print(f"File mancanti per {participant_id}, salto.")
            continue

        try:
            timestamps = dfe.extract_timestamps_from_log(log_path, participant_id)
            csv_df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Errore con {participant_id}: {e}")
            continue

        # Segmenti da creare
        segments = [
            ("baseline", timestamps["Inizio riproduzione primo video"], timestamps["Fine riproduzione primo video"], []),
            ("game1_part1", timestamps["Avvio del primo gioco"], timestamps["primo iGEQ mostrato, gioco messo in pausa"], []),
            ("game1_part2", timestamps["primo iGEQ terminato, gioco ripreso"], timestamps["secondo iGEQ mostrato, gioco messo in pausa"], []),
            ("game1_part3", timestamps["secondo iGEQ terminato, gioco ripreso"], timestamps["Chiusura del primo gioco"], []),
            ("game2_part1", timestamps["Avvio secondo gioco"], timestamps["terzo iGEQ mostrato, gioco messo in pausa"], []),
            ("game2_part2", timestamps["terzo iGEQ terminato, gioco ripreso"], timestamps["quarto iGEQ mostrato, gioco messo in pausa"], []),
            ("game2_part3", timestamps["quarto iGEQ terminato, gioco ripreso"], timestamps["Chiusura secondo gioco"], [])
        ]

        # Cartella di output
        participant_out = os.path.join(output_dir, participant_id)
        os.makedirs(participant_out, exist_ok=True)

        for name, start_ts, end_ts, exclusions in segments:
            try:
                if not exclusions:
                    start_sec = unix_to_seconds(csv_df, start_ts)
                    end_sec = unix_to_seconds(csv_df, end_ts)
                    duration = end_sec - start_sec
                    output_path = os.path.join(participant_out, f"{name}.mp4")
                    ffmpeg_cut(video_path, output_path, start_sec, duration)
                else:
                    # Caso: rimuovere intervalli dal video → taglia le parti valide e uniscile
                    clips = []
                    current_ts = start_ts
                    for excl_start, excl_end in exclusions:
                        if excl_start > current_ts:
                            s = unix_to_seconds(csv_df, current_ts)
                            e = unix_to_seconds(csv_df, excl_start)
                            d = e - s
                            tmp_clip = os.path.join(participant_out, f"tmp_{s}_{e}.mp4")
                            ffmpeg_cut(video_path, tmp_clip, s, d)
                            clips.append(tmp_clip)
                        current_ts = excl_end
                    # Aggiungi la parte finale
                    if current_ts < end_ts:
                        s = unix_to_seconds(csv_df, current_ts)
                        e = unix_to_seconds(csv_df, end_ts)
                        d = e - s
                        tmp_clip = os.path.join(participant_out, f"tmp_{s}_{e}.mp4")
                        ffmpeg_cut(video_path, tmp_clip, s, d)
                        clips.append(tmp_clip)

                    # Unisci tutto in un unico video
                    concat_list_path = os.path.join(participant_out, "concat_list.txt")
                    with open(concat_list_path, "w") as f:
                        for clip_path in clips:
                            f.write(f"file '{clip_path}'\n")
                    final_path = os.path.join(participant_out, f"{name}.mp4")
                    subprocess.run([
                        "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list_path,
                        "-c", "copy", final_path, "-y"
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                    # Pulisci i file temporanei
                    for clip in clips:
                        os.remove(clip)
                    os.remove(concat_list_path)

            except Exception as e:
                print(f"Errore nel taglio {name} per {participant_id}: {e}")

def analyze_participant_videos(main_folder, output_folder=None): 
    """
    Analizza tutti i video dei partecipanti usando py-feat, salvando AU, emozioni e pose.

    Args:
        main_folder (str): Cartella principale con le sottocartelle dei partecipanti.
        output_folder (str, optional): Cartella in cui salvare i risultati. Se None, non salva su disco.

    Returns:
        dict: Risultati organizzati per partecipante e nome video.
    """
    # Inizializza il detector di py-feat
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose"
    )

    results = {}

    # Scorri le cartelle dei partecipanti
    for participant_folder in os.listdir(main_folder):
        participant_path = os.path.join(main_folder, participant_folder)
        if not os.path.isdir(participant_path) or "_" not in participant_folder:
            continue

        print(f"Partecipante: {participant_folder}")
        results[participant_folder] = {}

        # Cartella di output per il partecipante
        if output_folder:
            participant_output = os.path.join(output_folder, participant_folder)
            os.makedirs(participant_output, exist_ok=True)

        # Trova i video
        video_files = [f for f in os.listdir(participant_path) if f.lower().endswith(('.mp4', '.avi', '.mov'))]

        if len(video_files) != 7:
            print(f"ATTENZIONE: trovati {len(video_files)} video invece di 7")

        for video_file in video_files:
            video_path = os.path.join(participant_path, video_file)
            video_name = os.path.splitext(video_file)[0]
            print(f"Analisi video: {video_file}")

            try:
                # Analizza direttamente il video
                fex = detector.detect_video(video_path, skip_frames=15)

                # Estrai i dataframe
                aus_df = fex.aus
                emotions_df = fex.emotions
                poses_df = fex.poses

                results[participant_folder][video_name] = {
                    "aus": aus_df,
                    "emotions": emotions_df,
                    "poses": poses_df
                }

                # Salva i CSV se richiesto
                if output_folder:
                    video_output = os.path.join(participant_output, video_name)
                    os.makedirs(video_output, exist_ok=True)

                    aus_df.to_csv(os.path.join(video_output, "aus.csv"), index=False)
                    emotions_df.to_csv(os.path.join(video_output, "emotions.csv"), index=False)
                    poses_df.to_csv(os.path.join(video_output, "poses.csv"), index=False)

            except Exception as e:
                print(f"ERRORE durante l'analisi di {video_file}: {str(e)}")

    return results