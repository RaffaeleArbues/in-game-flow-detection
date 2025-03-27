import pandas as pd
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import EEG.corrEEG as flowCal

def create_facial_flow_dataframe(face_metrics, df_noto_dict, df_ignoto_dict):
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

    flow_questions = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }

    all_data = []
    participant_flow_scores = {}

    # Prima passata: raccogli i punteggi di flow per la normalizzazione
    for participant_key, dataframes in face_metrics.items():
        participant = participant_key
        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict

        if not (group_A or group_B):
            continue

        participant_key_full = f"{'A' if group_A else 'B'}_{participant}"
        df_noto_q = df_noto_dict.get(participant_key_full)
        df_ignoto_q = df_ignoto_dict.get(participant_key_full)

        if df_noto_q is None or df_ignoto_q is None:
            continue

        if participant not in participant_flow_scores:
            participant_flow_scores[participant] = []

        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]

            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue

            flow_noto = calculate_flow_score(df_noto_q[interval_name], flow_questions[interval_idx])
            flow_ignoto = calculate_flow_score(df_ignoto_q[interval_name], flow_questions[interval_idx])

            if flow_noto is not None:
                participant_flow_scores[participant].append(flow_noto)
            if flow_ignoto is not None:
                participant_flow_scores[participant].append(flow_ignoto)

    # Seconda passata: costruzione del dataframe finale
    for participant, dataframes in face_metrics.items():
        if participant not in participant_flow_scores or len(participant_flow_scores[participant]) < 2:
            continue

        mean_flow = sum(participant_flow_scores[participant]) / len(participant_flow_scores[participant])
        std_flow = (sum((x - mean_flow) ** 2 for x in participant_flow_scores[participant]) / len(participant_flow_scores[participant])) ** 0.5
        if std_flow == 0:
            std_flow = 1e-6

        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict
        participant_key_full = f"{'A' if group_A else 'B'}_{participant}"
        df_noto_q = df_noto_dict.get(participant_key_full)
        df_ignoto_q = df_ignoto_dict.get(participant_key_full)

        for game_idx, df_game in enumerate(dataframes):
            tipo_gioco = "Gioco_Noto" if (group_A and game_idx == 0) or (group_B and game_idx == 1) else "Gioco_Ignoto"
            df_questionari = df_noto_q if tipo_gioco == "Gioco_Noto" else df_ignoto_q

            for interval_idx in range(3):
                interval_name = interval_map[interval_idx]
                questionnaire_type = interval_to_questionnaire[interval_idx]

                if interval_name not in df_questionari:
                    continue

                flow = calculate_flow_score(df_questionari[interval_name], flow_questions[interval_idx])
                if flow is None:
                    continue

                normalized_flow = (flow - mean_flow) / std_flow
                try:
                    features = df_game.iloc[interval_idx].to_dict()
                except Exception as e:
                    print(f"Errore per {participant}, Gioco {game_idx + 1}, Intervallo {interval_idx}: {e}")
                    continue

                row = {
                    "Partecipant_ID": participant,
                    "Tipo_Gioco": tipo_gioco,
                    "Intervallo": questionnaire_type,
                    "Flow": flow,
                    "Normalized_Flow": normalized_flow
                }
                row.update(features)
                all_data.append(row)

    df_all_data = pd.DataFrame(all_data)

    if not df_all_data.empty:
        df_all_data = df_all_data.sort_values(by=["Partecipant_ID", "Tipo_Gioco"], ascending=[True, False])
        df_all_data.to_csv("df_flow_face_data.csv", index=False)
        print(f"Dataframe creato con successo con {len(df_all_data)} righe.")
    else:
        print("ATTENZIONE: Il dataframe risultante è vuoto!")

    cleaned_df_all_data = clean_missing_values(df_all_data)
    return cleaned_df_all_data

def calculate_flow_score(df_interval, questions):
    """Calcola il punteggio medio di flow per un set di domande."""
    scores = []
    for q in questions:
        q_scores = df_interval.loc[df_interval["Domanda"].astype(str) == str(q), "Punteggio"].values
        if len(q_scores) > 0:
            scores.append(q_scores[0])
    return sum(scores) / len(scores) if scores else None


def clean_missing_values(df):
    """
    Riempie i NaN nel DataFrame con:
    - Interpolazione lineare lungo le righe
    - Media mobile centrata (rolling window di 3)
    - Media della colonna se necessario

    Le colonne categoriche (Partecipant_ID, Tipo_Gioco, Intervallo) sono escluse.
    """
    import pandas as pd

    # Copia di lavoro
    df_cleaned = df.copy()

    # Ordina per coerenza temporale
    df_cleaned = df_cleaned.sort_values(by=["Partecipant_ID", "Tipo_Gioco", "Intervallo"])

    # Escludi le colonne categoriche
    cat_cols = ["Partecipant_ID", "Tipo_Gioco", "Intervallo", "Flow", "Normalized_Flow"]
    num_cols = [col for col in df_cleaned.columns if col not in cat_cols and pd.api.types.is_numeric_dtype(df_cleaned[col])]

    # Interpolazione lineare (asse 0 = per colonna)
    df_cleaned[num_cols] = df_cleaned[num_cols].interpolate(method='linear', axis=0, limit_direction='both')

    # Media mobile centrata su finestra 3 (per riga)
    for col in num_cols:
        if df_cleaned[col].isna().any():
            df_cleaned[col] = df_cleaned[col].fillna(
                df_cleaned[col].rolling(window=3, min_periods=1, center=True).mean()
            )

    # Riempimento finale con la media della colonna
    df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].mean())

    return df_cleaned


# Attiva la conversione automatica tra Pandas e R
rpy2.robjects.pandas2ri.activate()

def run_mixed_models_face(df):
    from rpy2.robjects import pandas2ri
    import rpy2.robjects as ro

    # Mappatura Intervallo e Tipo_Gioco
    df["Intervallo"] = df["Intervallo"].map({"iGEQ_1": 1, "iGEQ_2": 2, "GEQ": 3})
    df["Tipo_Gioco"] = df["Tipo_Gioco"].map({"Gioco_Noto": 1, "Gioco_Ignoto": 0})
    df_cleaned = df.dropna()

    if df_cleaned.empty:
        raise ValueError("Il DataFrame è vuoto dopo la pulizia.")

    # Passa a R
    ro.globalenv['df'] = pandas2ri.py2rpy(df_cleaned)

    # Esegui codice R
    ro.r("""
        library(lme4)
        library(lmerTest)
        library(MuMIn)
        library(simr)

        output_files <- list(
            "AU" = "au_model_results.txt",
            "Emotion" = "emotion_model_results.txt",
            "Pose" = "pose_model_results.txt"
        )

        # Feature per gruppo
        au_features <- grep("^AU", names(df), value=TRUE)
        emotion_features <- grep("^(anger|fear|happiness|disgust|sadness|surprise|neutral)_", names(df), value=TRUE)
        pose_features <- grep("^(Pitch|Yaw|Roll)_", names(df), value=TRUE)

        feature_sets <- list(
            "AU" = au_features,
            "Emotion" = emotion_features,
            "Pose" = pose_features
        )

        calc_devianza <- function(model, var, file) {
            formula_ridotta <- as.formula(paste(". ~ . -", var))
            model_ridotto <- update(model, formula_ridotta)
            LRT <- anova(model_ridotto, model)
            out <- capture.output(LRT)
            cat("\nDeviance Explained for", var, "\n", out, file=file, sep="\n", append=TRUE)
        }

        for (group in names(feature_sets)) {
            features <- feature_sets[[group]]
            formula <- paste("Normalized_Flow ~ (", paste(features, collapse=" + "), "+ Intervallo + Tipo_Gioco) + (1 | Partecipant_ID)")
            model <- lmer(as.formula(formula), data=df, REML=FALSE)

            file <- output_files[[group]]
            out <- capture.output(summary(model))
            cat(paste0("\n*Linear Mixed Model - ", group, " Features*\n"), out, file=file, sep="\n", append=FALSE)

            out <- capture.output(r.squaredGLMM(model))
            cat("\nR-square\n", out, file=file, sep="\n", append=TRUE)

            out <- capture.output(summary(model)$coefficients)
            cat("\nCoefficients\n", out, file=file, sep="\n", append=TRUE)

            for (var in names(fixef(model))) {
                if (var != "(Intercept)") {
                    calc_devianza(model, var, file)
                }
            }

            cat(paste0("\nCalcolo del powerSim per il modello ", group, "...\n"), file=file, append=TRUE)
            power_model <- powerSim(model, test=fixed, nsim=200)
            out <- capture.output(power_model)
            cat("\nPower Analysis\n", out, file=file, sep="\n", append=TRUE)
        }
    """)

    print("Risultati salvati nei file:")
    print("- au_model_results.txt")
    print("- emotion_model_results.txt")
    print("- pose_model_results.txt")
