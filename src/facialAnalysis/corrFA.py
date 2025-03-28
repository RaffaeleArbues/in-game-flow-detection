import pandas as pd
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import src.EEG.corrEEG as corrEEG

def create_facial_flow_dataframe(face_metrics, df_noto_dict, df_ignoto_dict):
    """
        Creates a DataFrame linking facial expression features to self-reported flow scores during gameplay.

        This function processes precomputed facial features (from AU, emotion, and head pose data)
        and aligns them with questionnaire-derived flow scores collected during two gameplay sessions
        ("Gioco_Noto" and "Gioco_Ignoto") across three intervals.

        Main Steps:
        1. For each participant, determine group membership (A or B) to assign game order.
        2. Compute the mean and standard deviation of all flow scores for each participant, 
        used for Z-score normalization.
        3. For each interval (first, second, and post-game), retrieve:
        - The corresponding row of facial features.
        - The flow score from the correct questionnaire.
        - The normalized flow score.
        4. Build a row per interval containing:
        - Participant ID
        - Game type ("Gioco_Noto" or "Gioco_Ignoto")
        - Interval label
        - Flow and normalized flow
        - All facial features for that segment
        5. Append all rows to a single DataFrame.
        6. Clean missing values with interpolation, smoothing, and fallback strategies.
        7. Save the final DataFrame to "df_flow_face_data.csv".

        Args:
            face_metrics (dict): Dictionary mapping participant IDs to lists of two DataFrames
                                (one per game, each with three rows for intervals).
            df_noto_dict (dict): Dictionary of questionnaire DataFrames for the known game.
            df_ignoto_dict (dict): Dictionary of questionnaire DataFrames for the unknown game.

        Returns:
            pd.DataFrame: A cleaned and structured DataFrame ready for statistical analysis.
    """
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

    # Game Experience Questionnaire (both GEQ - 2 - and iGEQ - 0,1 - questions for Flow score)
    flow_questions = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }

    all_data = []
    participant_flow_scores = {}

    # Takes every participant's flow score
    for participant_key, dataframes in face_metrics.items():
        # Check if participant belongs to the A or B group
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

            flow_noto = corrEEG.calculate_flow_score(df_noto_q[interval_name], flow_questions[interval_idx])
            flow_ignoto = corrEEG.calculate_flow_score(df_ignoto_q[interval_name], flow_questions[interval_idx])

            if flow_noto is not None:
                participant_flow_scores[participant].append(flow_noto)
            if flow_ignoto is not None:
                participant_flow_scores[participant].append(flow_ignoto)

    # create df with normalized flow scores
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

                flow = corrEEG.calculate_flow_score(df_questionari[interval_name], flow_questions[interval_idx])
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
        print(f"Dataframe successfully created with {len(df_all_data)} rows.")
    else:
        print("WARNING: dataframe is empty!")

    cleaned_df_all_data = clean_missing_values(df_all_data)
    return cleaned_df_all_data


def clean_missing_values(df):
    """
    Fills NaN values in the DataFrame using:
    - Linear interpolation within each participant
    - Centered moving average (rolling window of 3)
    - Column mean as a fallback if needed

    Categorical columns (Partecipant_ID, Tipo_Gioco, Intervallo) are excluded from the process.
    """
    df_cleaned = df.copy()
    df_cleaned = df_cleaned.sort_values(by=["Partecipant_ID", "Tipo_Gioco", "Intervallo"])

    cat_cols = ["Partecipant_ID", "Tipo_Gioco", "Intervallo", "Flow", "Normalized_Flow"]
    num_cols = [col for col in df_cleaned.columns if col not in cat_cols and pd.api.types.is_numeric_dtype(df_cleaned[col])]

    # Interpolate separately for each participant
    df_cleaned[num_cols] = (
        df_cleaned
        .groupby("Partecipant_ID")[num_cols]
        .apply(lambda group: group.interpolate(method='linear', limit_direction='both'))
        .reset_index(level=0, drop=True)
    )

    # Centered rolling mean per column if still missing
    for col in num_cols:
        if df_cleaned[col].isna().any():
            df_cleaned[col] = df_cleaned[col].fillna(
                df_cleaned[col].rolling(window=3, min_periods=1, center=True).mean()
            )

    # Final fallback: column mean
    df_cleaned[num_cols] = df_cleaned[num_cols].fillna(df_cleaned[num_cols].mean())

    return df_cleaned

rpy2.robjects.pandas2ri.activate()

def run_mixed_models_face(df):

    df["Intervallo"] = df["Intervallo"].map({"iGEQ_1": 1, "iGEQ_2": 2, "GEQ": 3})
    df["Tipo_Gioco"] = df["Tipo_Gioco"].map({"Gioco_Noto": 1, "Gioco_Ignoto": 0})
    df_cleaned = df.dropna()

    if df_cleaned.empty:
        raise ValueError("Il DataFrame è vuoto dopo la pulizia.")

    ro.globalenv['df'] = pandas2ri.py2rpy(df_cleaned)

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
