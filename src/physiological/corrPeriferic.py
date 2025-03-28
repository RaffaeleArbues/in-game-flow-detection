import pandas as pd
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import src.EEG.corrEEG as corrEEG

def create_physiological_flow_dataframe(segmented_metrics_eda, segmented_metrics_bvp, df_noto_dict, df_ignoto_dict):
    """
        Function: create_physiological_flow_dataframe

        Description:
        This function integrates physiological metrics (EDA and BVP) with self-reported flow questionnaire scores 
        to build a structured DataFrame for statistical analysis. The goal is to correlate biometric signals 
        with flow experiences during gameplay in two types of games: "Gioco_Noto" (known, familiar) and "Gioco_Ignoto" (unknown or unfamiliar).

        Inputs:
        - segmented_metrics_eda: Dictionary mapping participant IDs to a list of two DataFrames (EDA metrics), 
        one per game. Each DataFrame has 3 rows (one per game interval) and 14 columns (interval + eda metrics).
        - segmented_metrics_bvp: Dictionary similar to segmented_metrics_eda but for BVP metrics.
        - df_noto_dict: Dictionary containing questionnaire responses for each participant during the known game. 
        Each value is a dictionary of DataFrames (e.g., 'df_selfreport_1', 'df_selfreport_2', 'df_selfreport_final').
        - df_ignoto_dict: Same as df_noto_dict, but for the unknown game.

        Processing Steps:
        1. For each participant, identify group membership (A or B) and retrieve corresponding questionnaire data.
        2. Compute flow scores for three intervals using predefined flow-related questions:
        - Interval 0 and 1: in-game GEQ (iGEQ_1, iGEQ_2)
        - Interval 2: post-game GEQ (GEQ)
        3. Normalize the flow scores per participant (z-score across all their questionnaire scores).
        4. For each interval and each game type, merge the flow score (raw and normalized) with EDA and BVP metrics.
        5. Build a row per interval containing:
        - Participant ID
        - Game Type ("Gioco_Noto" or "Gioco_Ignoto")
        - Interval Label
        - Flow Score
        - Normalized Flow Score
        - Corresponding EDA and BVP features (one row per interval)

        Output:
        - A pandas DataFrame with combined flow questionnaire scores and physiological data.
        - A CSV file ("df_flow_physiological_data.csv") is saved with the complete data.
        - Returns the DataFrame for further analysis.

        Note:
        - Participants without sufficient data (e.g., missing intervals or questionnaire responses) are skipped.
        - If all participants are skipped, a warning message is printed.
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

    # Dictionary to temporarily collect all flow scores for each participant
    flow_questions = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }

    all_data = []

    flow_scores_dict = {}
    all_raw_flow_scores = {}

    # Takes every participant's flow score
    for participant in set(segmented_metrics_eda.keys()) | set(segmented_metrics_bvp.keys()):
        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict

        if not (group_A or group_B):
            continue
        
        participant_key = f"{'A' if group_A else 'B'}_{participant}"
        game_noto = "Gioco_Noto"
        game_ignoto = "Gioco_Ignoto"

        df_noto_q = df_noto_dict.get(participant_key)
        df_ignoto_q = df_ignoto_dict.get(participant_key)
        if df_noto_q is None or df_ignoto_q is None:
            continue

        # initialization of flow scores
        flow_scores_dict[participant] = {game_noto: {}, game_ignoto: {}}
        all_raw_flow_scores[participant] = []

        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]
            questionnaire_type = interval_to_questionnaire[interval_idx]

            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue

            df_noto_interval = df_noto_q[interval_name]
            df_ignoto_interval = df_ignoto_q[interval_name]

            flow_noto = corrEEG.calculate_flow_score(df_noto_interval, flow_questions[interval_idx])
            flow_ignoto = corrEEG.calculate_flow_score(df_ignoto_interval, flow_questions[interval_idx])

            if flow_noto is not None:
                flow_scores_dict[participant][game_noto][questionnaire_type] = flow_noto
                all_raw_flow_scores[participant].append(flow_noto)
            if flow_ignoto is not None:
                flow_scores_dict[participant][game_ignoto][questionnaire_type] = flow_ignoto
                all_raw_flow_scores[participant].append(flow_ignoto)

    # Building Dataframe
    for participant in flow_scores_dict:
        flow_list = all_raw_flow_scores[participant]
        if len(flow_list) < 2:
            continue

        mean_flow = sum(flow_list) / len(flow_list)
        std_flow = (sum((x - mean_flow) ** 2 for x in flow_list) / len(flow_list)) ** 0.5
        if std_flow == 0:
            std_flow = 1e-6

        group_A = f"A_{participant}" in df_noto_dict
        participant_key = f"{'A' if group_A else 'B'}_{participant}"

        noto_game_index = f"game{'1' if group_A else '2'}"

        eda_metrics_list = segmented_metrics_eda.get(participant, [])
        bvp_metrics_list = segmented_metrics_bvp.get(participant, [])
        df_game1_eda, df_game2_eda = (eda_metrics_list if len(eda_metrics_list) == 2 else (None, None))
        df_game1_bvp, df_game2_bvp = (bvp_metrics_list if len(bvp_metrics_list) == 2 else (None, None))

        df_noto_eda = df_game1_eda if noto_game_index == "game1" else df_game2_eda
        df_ignoto_eda = df_game2_eda if noto_game_index == "game1" else df_game1_eda
        df_noto_bvp = df_game1_bvp if noto_game_index == "game1" else df_game2_bvp
        df_ignoto_bvp = df_game2_bvp if noto_game_index == "game1" else df_game1_bvp

        for interval_idx in range(3):
            questionnaire_type = interval_to_questionnaire[interval_idx]

            for tipo_gioco, df_eda, df_bvp in [
                ("Gioco_Noto", df_noto_eda, df_noto_bvp),
                ("Gioco_Ignoto", df_ignoto_eda, df_ignoto_bvp)
            ]:
                flow_value = flow_scores_dict[participant][tipo_gioco].get(questionnaire_type)
                if flow_value is None:
                    continue

                normalized = (flow_value - mean_flow) / std_flow

                row = {
                    "Partecipant_ID": participant,
                    "Tipo_Gioco": tipo_gioco,
                    "Intervallo": questionnaire_type,
                    "Flow": flow_value,
                    "Normalized_Flow": normalized
                }

                # EDA
                if df_eda is not None and not df_eda.empty and interval_idx < len(df_eda):
                    try:
                        eda_row = df_eda.iloc[interval_idx]
                        for col in df_eda.columns:
                            row[f"EDA_{col}"] = eda_row[col]
                    except Exception as e:
                        print(f"Error with EDA {participant}, {tipo_gioco}, {questionnaire_type}: {e}")
                # BVP
                if df_bvp is not None and not df_bvp.empty and interval_idx < len(df_bvp):
                    try:
                        bvp_row = df_bvp.iloc[interval_idx]
                        for col in df_bvp.columns:
                            row[f"BVP_{col}"] = bvp_row[col]
                    except Exception as e:
                        print(f"Error with BVP {participant}, {tipo_gioco}, {questionnaire_type}: {e}")

                all_data.append(row)

    df_all_data = pd.DataFrame(all_data)

    if not df_all_data.empty:
        df_all_data = df_all_data.sort_values(by=["Partecipant_ID", "Tipo_Gioco"], ascending=[True, False])
        # Removes useless columns
        columns_to_drop = ["EDA_Interval", "BVP_Interval"]
        df_all_data = df_all_data.drop(columns=[col for col in columns_to_drop if col in df_all_data.columns])
        df_all_data.to_csv("df_flow_physiological_data.csv", index=False)
        print(f"Dataframe successfully created with {len(df_all_data)} rows.")
    else:
        print("WARNING: dataframe is empty!")

    return df_all_data

# Attiva la conversione automatica tra Pandas e R
rpy2.robjects.pandas2ri.activate()

def run_physiological_mixed_model(df):
    """
        Performs a Linear Mixed Model in R using all physiological metrics (EDA and BVP) 
        as fixed effects, along with Interval and Game Type, and Participant_ID as a random effect.
        The results are saved to the file 'physio_model_results.txt'.
    """

    # converting Intervals and game type in numbers
    mapping_intervallo = {"iGEQ_1": 1, "iGEQ_2": 2, "GEQ": 3}
    mapping_tipo_gioco = {"Gioco_Noto": 1, "Gioco_Ignoto": 0}

    df["Intervallo"] = df["Intervallo"].map(mapping_intervallo)
    df["Tipo_Gioco"] = df["Tipo_Gioco"].map(mapping_tipo_gioco)

    # Remove NaN rows
    df_cleaned = df.dropna()
    if df_cleaned.empty:
        raise ValueError("Il DataFrame è vuoto dopo la rimozione dei NaN.")

    # passing DataFrame to R
    ro.globalenv['df'] = pandas2ri.py2rpy(df_cleaned)

    # executing the R script
    ro.r("""
        library(lme4)
        library(lmerTest)
        library(MuMIn)
        library(simr)

        output_physio <- "physio_model_results.txt"

        calc_devianza <- function(model, var, file) {
            formula_ridotta <- as.formula(paste(". ~ . -", var))
            model_ridotto <- update(model, formula_ridotta)
            LRT <- anova(model_ridotto, model)
            out <- capture.output(LRT)
            cat("\\nDeviance Explained for", var, "\\n", out, file=file, sep="\\n", append=TRUE)
        }

        model_physio <- lmer(
            Normalized_Flow ~ (
                EDA_min_eda + EDA_max_eda + EDA_avg_eda +
                EDA_min_eda_tonic + EDA_max_eda_tonic + EDA_avg_eda_tonic +
                EDA_min_eda_phasic + EDA_max_eda_phasic + EDA_avg_eda_phasic +
                EDA_delta_eda + EDA_f_DecRate_eda + EDA_f_DecTime_eda + EDA_f_NbPeaks_eda +
                BVP_mu_bvp + BVP_sigma_bvp +
                BVP_mu_hr + BVP_delta_hr + BVP_sigma_hr +
                BVP_SDNN + BVP_RMSSD +
                Intervallo + Tipo_Gioco
            ) + (1 | Partecipant_ID),
            data=df,
            REML=FALSE
        )

        out <- capture.output(summary(model_physio))
        cat("\\n*Linear Mixed Model - Physiological Model*\\n", out, file=output_physio, sep="\\n", append=FALSE)

        out <- capture.output(r.squaredGLMM(model_physio))
        cat("\\nR-square - Physiological Model\\n", out, file=output_physio, sep="\\n", append=TRUE)

        out <- capture.output(summary(model_physio)$coefficients)
        cat("\\nCoefficients - Physiological Model\\n", out, file=output_physio, sep="\\n", append=TRUE)

        for (var in names(fixef(model_physio))) {
            if (var != "(Intercept)") {
                calc_devianza(model_physio, var, output_physio)
            }
        }

        print("Calcolando powerSim per il modello fisiologico...")
        power_physio <- powerSim(model_physio, test = fixed, nsim = 200)
        print("PowerSim completato.")
        out <- capture.output(power_physio)
        cat("\\nPower Analysis - Physiological Model\\n", out, file=output_physio, sep="\\n", append=TRUE)
    """)

    print("Results saved in this file: 'physio_model_results.txt'")