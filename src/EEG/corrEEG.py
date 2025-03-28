import pandas as pd
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri

def create_EEG_flow_dataframe(aggregated_amplitudes, df_noto_dict, df_ignoto_dict, method):
    """
        Constructs a final DataFrame that links EEG amplitudes (for each frequency band and electrode),
        subjective flow scores (from questionnaires), and metadata (participant ID, game type, interval).
        This structure is used to analyze the relationship between EEG activity and perceived flow state.

        The function takes EEG data (already aggregated by game and band, in a dictionary which has 2 df per key with 3rows (interval) x 6 col
        (which are lists of 8 amplitudes, one value per channel)), questionnaire responses (split across intervals), 
        and the amplitude method used ("rms" or "ptp").

        Main Steps:
        1. Iterate over all participants to extract their raw flow scores from the questionnaires.
        These scores are used to compute the mean and standard deviation per participant,
        in order to normalize the flow values using Z-score normalization.

        2. For each participant, for each of the three intervals (iGEQ_1, iGEQ_2, GEQ):
        - Extract the corresponding EEG segment for both the known and unknown games.
        - Compute the average flow score using specific questionnaire items.
        - Normalize the flow score using the participant's own mean and standard deviation.
        - Create a dictionary (row) that includes:
            * Participant ID
            * Game type (known/unknown)
            * Interval type
            * Raw and normalized flow score
            * EEG amplitude values for each frequency band and sensor
        - Append this row to the final dataset.

        3. Once all rows are collected:
        - A final DataFrame is created and sorted.
        - The structure includes one row per game, per interval, per participant (6 rows total per participant).
        - The final DataFrame is saved to "df_flow_eeg_data.csv" for further analysis.

        Supporting Function:
        - `calculate_flow_score(df_interval, questions)`: receives a questionnaire DataFrame and a list of question numbers,
        and returns the average score given by the participant for those questions.

        Notes:
        - The function dynamically assigns the correct game order based on the participant's group (A or B).
        - If any EEG or questionnaire data is missing or inconsistent, the corresponding row is skipped.
        - The function is flexible and supports multiple amplitude extraction methods (e.g., ptp, rms).

        Returns:
        - A Pandas DataFrame with participant-wise, interval-wise EEG features and flow scores.
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
    
    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    sensors = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    
    all_data = []
    
    # Dictionary to temporarily collect all flow scores for each participant
    participant_flow_scores = {}
    
    # Takes every participant's flow score
    for participant, games in aggregated_amplitudes.items():
        # Check if participant belongs to the A or B group
        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict
        
        if not (group_A or group_B):
            continue  # Participant not found
            
        participant_key = f"{'A' if group_A else 'B'}_{participant}"
        
        # Assign the correct game order arrangement based on the participant's group affiliation (A or B). 
        game_noto_key = f"game{'1' if group_A else '2'}_{method}"
        game_ignoto_key = f"game{'2' if group_A else '1'}_{method}"
        
        # retrive questionnaire dfs
        df_noto_q = df_noto_dict.get(participant_key)
        df_ignoto_q = df_ignoto_dict.get(participant_key)
        
        if df_noto_q is None or df_ignoto_q is None:
            continue
            
        if participant not in participant_flow_scores:
            participant_flow_scores[participant] = []
            
        # For each interval
        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]
            
            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue
                
            # Assigns questionnaire df fot this interval
            df_noto_interval = df_noto_q[interval_name]
            df_ignoto_interval = df_ignoto_q[interval_name]
            
            # flow mean for both games
            flow_noto = calculate_flow_score(df_noto_interval, flow_questions[interval_idx])
            flow_ignoto = calculate_flow_score(df_ignoto_interval, flow_questions[interval_idx])
            
            # Add valid scores to the participant's scores list (for both game1 and game 2)
            if flow_noto is not None:
                participant_flow_scores[participant].append(flow_noto)
            if flow_ignoto is not None:
                participant_flow_scores[participant].append(flow_ignoto)
    
    # create df with normalized flow scores
    for participant, games in aggregated_amplitudes.items():
        # skip if we have not collected scores for this participant
        if participant not in participant_flow_scores or len(participant_flow_scores[participant]) < 2:
            continue
            
        # mean e standard deviation for this participant
        flow_scores = participant_flow_scores[participant]
        mean_flow = sum(flow_scores) / len(flow_scores)
        std_flow = (sum((x - mean_flow) ** 2 for x in flow_scores) / len(flow_scores)) ** 0.5
        
        # If standard deviation is 0, we set a min value to avoid division by 0
        if std_flow == 0:
            std_flow = 1e-6
        
        # determine the participant's group (A o B)
        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict
        
        if not (group_A or group_B):
            continue
            
        participant_key = f"{'A' if group_A else 'B'}_{participant}"
    
        game_noto_key = f"game{'1' if group_A else '2'}_{method}"
        game_ignoto_key = f"game{'2' if group_A else '1'}_{method}"
        
        df_noto = games.get(game_noto_key)
        df_ignoto = games.get(game_ignoto_key)
        df_noto_q = df_noto_dict.get(participant_key)
        df_ignoto_q = df_ignoto_dict.get(participant_key)
        
        if any(x is None or (isinstance(x, pd.DataFrame) and x.empty) for x in [df_noto, df_ignoto, df_noto_q, df_ignoto_q]):
            continue

        # For each interval
        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]
            questionnaire_type = interval_to_questionnaire[interval_idx]
            
            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue
                
            df_noto_interval = df_noto_q[interval_name]
            df_ignoto_interval = df_ignoto_q[interval_name]
            
            flow_noto = calculate_flow_score(df_noto_interval, flow_questions[interval_idx])
            flow_ignoto = calculate_flow_score(df_ignoto_interval, flow_questions[interval_idx])
            
            # apply z-score: (x - μ) / σ
            normalized_flow_noto = (flow_noto - mean_flow) / std_flow if flow_noto is not None else None
            normalized_flow_ignoto = (flow_ignoto - mean_flow) / std_flow if flow_ignoto is not None else None
            
            if flow_noto is None and flow_ignoto is None:
                continue
                
            # Creates the rows for final df
            if flow_noto is not None:
                row_noto = {
                    "Partecipant_ID": participant,
                    "Tipo_Gioco": "Gioco_Noto",
                    "Intervallo": questionnaire_type,
                    "Flow": flow_noto,
                    "Normalized_Flow": normalized_flow_noto
                }
                # Add the value for channel name for each band
                for banda in bands:
                    try:
                        noto_eeg = df_noto[banda].iloc[interval_idx]
                        for i, sensor in enumerate(sensors):
                            row_noto[f"{sensor}_{banda}"] = noto_eeg[i]
                    except Exception as e:
                        print(f"Errore con {participant}, intervallo {interval_idx}, banda {banda}: {e}")
                        for i, sensor in enumerate(sensors):
                            row_noto[f"{sensor}_{banda}"] = None
                
                all_data.append(row_noto)
                
            if flow_ignoto is not None:
                row_ignoto = {
                    "Partecipant_ID": participant,
                    "Tipo_Gioco": "Gioco_Ignoto",
                    "Intervallo": questionnaire_type,
                    "Flow": flow_ignoto,
                    "Normalized_Flow": normalized_flow_ignoto
                }
                # Add the value for channel name for each band
                for banda in bands:
                    try:
                        ignoto_eeg = df_ignoto[banda].iloc[interval_idx]
                        for i, sensor in enumerate(sensors):
                            row_ignoto[f"{sensor}_{banda}"] = ignoto_eeg[i]
                    except Exception as e:
                        print(f"Error occurred for {participant}, interval {interval_idx}, band {banda}: {e}")
                        for i, sensor in enumerate(sensors):
                            row_ignoto[f"{sensor}_{banda}"] = None
                
                all_data.append(row_ignoto)
    
    df_all_data = pd.DataFrame(all_data)
    
    # check if there's data in there
    if not df_all_data.empty:
        df_all_data = df_all_data.sort_values(by=["Partecipant_ID", "Tipo_Gioco"], ascending=[True, False])

        df_all_data.to_csv("df_flow_eeg_data.csv", index=False)
        print(f"Dataframe successfully created with {len(df_all_data)} rows.")
    else:
        print("WARNING: dataframe is empty!")
    
    return df_all_data

def calculate_flow_score(df_interval, questions):
    """Calculate the mean Flow score for a set of questions."""
    scores = []
    for q in questions:
        q_scores = df_interval.loc[df_interval["Domanda"].astype(str) == str(q), "Punteggio"].values
        if len(q_scores) > 0:
            scores.append(q_scores[0])
    
    return sum(scores) / len(scores) if scores else None

# Activates the automatic conversion between pandas and R
rpy2.robjects.pandas2ri.activate()

def run_mixed_models(df):
    """
        Runs the full model and individual models for each band (Alpha, Beta, Delta, Gamma, Theta) in R.
        Calculates the explained deviance for each fixed effect and performs a power test.
        Saves the results into separate files.
    """

    # converting Intervals and game type in numbers
    mapping_intervallo = {"iGEQ_1": 1, "iGEQ_2": 2, "GEQ": 3}
    mapping_tipo_gioco = {"Gioco_Noto": 1, "Gioco_Ignoto": 0}

    df["Intervallo"] = df["Intervallo"].map(mapping_intervallo)
    df["Tipo_Gioco"] = df["Tipo_Gioco"].map(mapping_tipo_gioco)

    # Remove NaN rows
    df_cleaned = df.dropna()

    if df_cleaned.empty:
        raise ValueError("Errore: il DataFrame è vuoto dopo la rimozione dei NaN!")

    # passing DataFrame to R
    ro.globalenv['df'] = rpy2.robjects.pandas2ri.py2rpy(df_cleaned)

    # executing the R script
    ro.r("""
        library(lme4)
        library(lmerTest)
        library(MuMIn)  # Per calcolare R² dei modelli misti
        library(simr)   # Per calcolare il power test

        # File di output
        output_full <- "full_model_results.txt"
        output_alpha <- "alpha_model_results.txt"
        output_beta <- "beta_model_results.txt"
        output_delta <- "delta_model_results.txt"
        output_gamma <- "gamma_model_results.txt"
        output_theta <- "theta_model_results.txt"

        # Funzione per calcolare la devianza spiegata #
        calc_devianza <- function(model, var, file) {
            formula_ridotta <- as.formula(paste(". ~ . -", var))
            model_ridotto <- update(model, formula_ridotta)
            LRT <- anova(model_ridotto, model)
            out <- capture.output(LRT)
            cat("\nDeviance Explained for", var, "\n", out, file=file, sep="\n", append=TRUE)
        }

        #  Modello COMPLETO #
        model_full <- lmer(Normalized_Flow ~ (CP3_alpha + C3_alpha + F5_alpha + PO3_alpha + PO4_alpha + 
                                   F6_alpha + C4_alpha + CP4_alpha + CP3_beta + C3_beta + 
                                   F5_beta + PO3_beta + PO4_beta + F6_beta + C4_beta + CP4_beta +
                                   CP3_delta + C3_delta + F5_delta + PO3_delta + PO4_delta + 
                                   F6_delta + C4_delta + CP4_delta + CP3_gamma + C3_gamma + 
                                   F5_gamma + PO3_gamma + PO4_gamma + F6_gamma + C4_gamma + 
                                   CP4_gamma + CP3_theta + C3_theta + F5_theta + PO3_theta + 
                                   PO4_theta + F6_theta + C4_theta + CP4_theta + Intervallo + Tipo_Gioco) + 
                                   (1 | Partecipant_ID), data=df, REML=FALSE)

        # Salviamo il summary del modello completo
        out <- capture.output(summary(model_full))
        cat("\n*Linear Mixed Model - Full Model*\n", out, file=output_full, sep="\n", append=FALSE)
        
        # Calcoliamo R^2
        out <- capture.output(r.squaredGLMM(model_full))
        cat("\nR-square - Full Model\n", out, file=output_full, sep="\n", append=TRUE)
        
        # Salviamo i coefficienti
        out <- capture.output(summary(model_full)$coefficients)
        cat("\nCoefficients - Full Model\n", out, file=output_full, sep="\n", append=TRUE)

        # Calcoliamo la devianza spiegata per ogni variabile nel modello completo
        for (var in names(fixef(model_full))) {
            if (var != "(Intercept)") {
                calc_devianza(model_full, var, output_full)
            }
        }

        # Calcoliamo il power test per il modello completo
        print("Calcolando powerSim per il modello completo...")
        power_full <- powerSim(model_full, test = fixed, nsim = 200) 
        print("PowerSim per il modello completo completato.")
        out <- capture.output(power_full)
        cat("\nPower Analysis - Full Model\n", out, file=output_full, sep="\n", append=TRUE)

        # Modelli Separati per Banda #

        # Lista delle bande e dei rispettivi output file
        bande <- list(
            "Alpha" = c("CP3_alpha", "C3_alpha", "F5_alpha", "PO3_alpha", "PO4_alpha", "F6_alpha", "C4_alpha", "CP4_alpha"),
            "Beta"  = c("CP3_beta", "C3_beta", "F5_beta", "PO3_beta", "PO4_beta", "F6_beta", "C4_beta", "CP4_beta"),
            "Delta" = c("CP3_delta", "C3_delta", "F5_delta", "PO3_delta", "PO4_delta", "F6_delta", "C4_delta", "CP4_delta"),
            "Gamma" = c("CP3_gamma", "C3_gamma", "F5_gamma", "PO3_gamma", "PO4_gamma", "F6_gamma", "C4_gamma", "CP4_gamma"),
            "Theta" = c("CP3_theta", "C3_theta", "F5_theta", "PO3_theta", "PO4_theta", "F6_theta", "C4_theta", "CP4_theta")
        )

        output_files <- list(
            "Alpha" = output_alpha,
            "Beta" = output_beta,
            "Delta" = output_delta,
            "Gamma" = output_gamma,
            "Theta" = output_theta
        )

        # Creiamo ed eseguiamo i modelli per ogni banda e calcoliamo la devianza spiegata + power test
        for (banda in names(bande)) {
            formula <- paste("Normalized_Flow ~ (", paste(bande[[banda]], collapse=" + "), " + Intervallo + Tipo_Gioco) + (1 | Partecipant_ID)")
            model <- lmer(as.formula(formula), data=df, REML=FALSE)
            
            # Salviamo il summary
            out <- capture.output(summary(model))
            cat("\n*Linear Mixed Model -", banda, "Model*\n", out, file=output_files[[banda]], sep="\n", append=FALSE)
            
            # Calcoliamo R² e lo salviamo
            out <- capture.output(r.squaredGLMM(model))
            cat("\nR-square -", banda, "Model\n", out, file=output_files[[banda]], sep="\n", append=TRUE)
            
            # Salviamo i coefficienti
            out <- capture.output(summary(model)$coefficients)
            cat("\nCoefficients -", banda, "Model\n", out, file=output_files[[banda]], sep="\n", append=TRUE)
            
            # Calcoliamo la devianza spiegata per ogni variabile nel modello
            for (var in names(fixef(model))) {
                if (var != "(Intercept)") {
                    calc_devianza(model, var, output_files[[banda]])
                }
            }

            # Calcoliamo la devianza spiegata per ogni variabile nel modello
            for (var in names(fixef(model))) {
                if (var != "(Intercept)") {
                    calc_devianza(model, var, output_files[[banda]])
                }
            }

            print(paste("Calcolando powerSim per il modello:", banda, "..."))
            power_model <- powerSim(model, test = fixed, nsim = 200)  # Ridotto a 200 simulazioni
            print(paste("PowerSim per", banda, "completato."))

            out <- capture.output(power_model)
            cat("\nPower Analysis -", banda, "Model\n", out, file=output_files[[banda]], sep="\n", append=TRUE)
        }
    """)

    print("Results saved in these files:")
    print("- full_model_results.txt")
    print("- alpha_model_results.txt")
    print("- beta_model_results.txt")
    print("- delta_model_results.txt")
    print("- gamma_model_results.txt")