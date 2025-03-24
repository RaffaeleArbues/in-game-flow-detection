import pandas as pd
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import src.EEG.corr as flowCal


def create_physiological_flow_dataframe(segmented_metrics_eda, segmented_metrics_bvp, df_noto_dict, df_ignoto_dict):
    import pandas as pd

    # Mappature intervalli e questionari
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

    # Domande di flow
    flow_questions = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }

    all_data = []

    # Nuovo dizionario: punteggi per ogni partecipante, gioco e intervallo
    flow_scores_dict = {}
    all_raw_flow_scores = {}

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

        # Inizializza
        flow_scores_dict[participant] = {game_noto: {}, game_ignoto: {}}
        all_raw_flow_scores[participant] = []

        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]
            questionnaire_type = interval_to_questionnaire[interval_idx]

            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue

            df_noto_interval = df_noto_q[interval_name]
            df_ignoto_interval = df_ignoto_q[interval_name]

            flow_noto = flowCal.calculate_flow_score(df_noto_interval, flow_questions[interval_idx])
            flow_ignoto = flowCal.calculate_flow_score(df_ignoto_interval, flow_questions[interval_idx])

            if flow_noto is not None:
                flow_scores_dict[participant][game_noto][questionnaire_type] = flow_noto
                all_raw_flow_scores[participant].append(flow_noto)
            if flow_ignoto is not None:
                flow_scores_dict[participant][game_ignoto][questionnaire_type] = flow_ignoto
                all_raw_flow_scores[participant].append(flow_ignoto)

    # Seconda passata: costruzione DataFrame
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

        game_noto_key = f"game{'1' if group_A else '2'}"
        game_ignoto_key = f"game{'2' if group_A else '1'}"

        eda_metrics_list = segmented_metrics_eda.get(participant, [])
        bvp_metrics_list = segmented_metrics_bvp.get(participant, [])
        df_game1_eda, df_game2_eda = (eda_metrics_list if len(eda_metrics_list) == 2 else (None, None))
        df_game1_bvp, df_game2_bvp = (bvp_metrics_list if len(bvp_metrics_list) == 2 else (None, None))

        df_noto_eda = df_game1_eda if game_noto_key == "game1" else df_game2_eda
        df_ignoto_eda = df_game2_eda if game_noto_key == "game1" else df_game1_eda
        df_noto_bvp = df_game1_bvp if game_noto_key == "game1" else df_game2_bvp
        df_ignoto_bvp = df_game2_bvp if game_noto_key == "game1" else df_game1_bvp

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
                        print(f"Errore con EDA {participant}, {tipo_gioco}, {questionnaire_type}: {e}")
                # BVP
                if df_bvp is not None and not df_bvp.empty and interval_idx < len(df_bvp):
                    try:
                        bvp_row = df_bvp.iloc[interval_idx]
                        for col in df_bvp.columns:
                            row[f"BVP_{col}"] = bvp_row[col]
                    except Exception as e:
                        print(f"Errore con BVP {participant}, {tipo_gioco}, {questionnaire_type}: {e}")

                all_data.append(row)

    df_all_data = pd.DataFrame(all_data)

    if not df_all_data.empty:
        df_all_data = df_all_data.sort_values(by=["Partecipant_ID", "Tipo_Gioco"], ascending=[True, False])
        # Rimuovi le colonne non necessarie, se presenti
        columns_to_drop = ["EDA_Interval", "BVP_Interval"]
        df_all_data = df_all_data.drop(columns=[col for col in columns_to_drop if col in df_all_data.columns])
        df_all_data.to_csv("df_flow_physiological_data.csv", index=False)
        print(f"Dataframe creato con successo con {len(df_all_data)} righe.")
    else:
        print("ATTENZIONE: Il dataframe risultante è vuoto!")

    return df_all_data

# Attiva la conversione automatica tra Pandas e R
rpy2.robjects.pandas2ri.activate()

def run_physiological_mixed_model(df):
    """
    Esegue un Linear Mixed Model in R con tutte le metriche fisiologiche EDA e BVP
    come effetti fissi, oltre a Intervallo e Tipo_Gioco, e Partecipant_ID come effetto casuale.
    I risultati vengono salvati nel file 'physio_model_results.txt'.
    """
    # Attiva la conversione da pandas a R
    pandas2ri.activate()

    # Mappa le variabili categoriali come numeri (se non già fatto)
    mapping_intervallo = {"iGEQ_1": 1, "iGEQ_2": 2, "GEQ": 3}
    mapping_tipo_gioco = {"Gioco_Noto": 1, "Gioco_Ignoto": 0}
    df["Intervallo"] = df["Intervallo"].map(mapping_intervallo)
    df["Tipo_Gioco"] = df["Tipo_Gioco"].map(mapping_tipo_gioco)

    # Pulisce i NaN
    df_cleaned = df.dropna()
    if df_cleaned.empty:
        raise ValueError("Il DataFrame è vuoto dopo la rimozione dei NaN.")

    # Passa il DataFrame a R
    ro.globalenv['df'] = pandas2ri.py2rpy(df_cleaned)

    # Script R da eseguire
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

    print("Modello misto eseguito. Risultati salvati in 'physio_model_results.txt'")