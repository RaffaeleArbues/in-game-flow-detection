import pandas as pd
import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri


def create_flow_dataframe(aggregated_amplitudes, df_noto_dict, df_ignoto_dict, method):
    """
    Crea un DataFrame con le informazioni di partecipanti, tipo di gioco, intervallo, punteggio di flow
    e valori dei sensori per ciascuna banda. Include anche il punteggio di flow normalizzato.
    La funzione è stata modificata per integrare l'informazione della banda nei nomi delle colonne dei sensori.

    Parametri:
    - aggregated_amplitudes (dict): Dizionario con i DataFrame delle ampiezze EEG per ogni partecipante.
    - df_noto_dict (dict): Dizionario con i dati dei questionari per il gioco noto.
    - df_ignoto_dict (dict): Dizionario con i dati dei questionari per il gioco ignoto.
    - method (str): Metodo di calcolo dell'ampiezza utilizzato.

    Ritorna:
    - DataFrame con i dati formattati secondo la struttura richiesta.
    """    
    
    # Configurazione delle mappature
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

    # Domande relative al flow per ciascun intervallo temporale
    flow_questions = {
        0: [5, 10],
        1: [5, 10],
        2: [5, 13, 25, 28, 31]
    }
    
    bands = ['alpha', 'beta', 'delta', 'gamma', 'theta']
    sensors = ['CP3', 'C3', 'F5', 'PO3', 'PO4', 'F6', 'C4', 'CP4']
    
    all_data = []
    
    # Dizionario per raccogliere temporaneamente tutti i punteggi di flow per ogni partecipante
    participant_flow_scores = {}
    
    # Prima passata: raccogli tutti i punteggi di flow per ogni partecipante
    for participant, games in aggregated_amplitudes.items():
        # Determina il gruppo del partecipante (A o B)
        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict
        
        if not (group_A or group_B):
            continue  # Partecipante non trovato nei dizionari
            
        participant_key = f"{'A' if group_A else 'B'}_{participant}"
        
        # Per gruppo B, il gioco noto è game2 ma i questionari sono in df_noto_dict
        game_noto_key = f"game{'1' if group_A else '2'}_{method}"
        game_ignoto_key = f"game{'2' if group_A else '1'}_{method}"
        
        # Recupera i dataframe dei questionari
        df_noto_q = df_noto_dict.get(participant_key)
        df_ignoto_q = df_ignoto_dict.get(participant_key)
        
        if df_noto_q is None or df_ignoto_q is None:
            continue
            
        # Inizializza l'elenco dei punteggi per questo partecipante se non esiste
        if participant not in participant_flow_scores:
            participant_flow_scores[participant] = []
            
        # Per ogni intervallo
        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]
            
            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue
                
            # Dataframe dei questionari per questo intervallo
            df_noto_interval = df_noto_q[interval_name]
            df_ignoto_interval = df_ignoto_q[interval_name]
            
            # Calcola media flow per entrambi i giochi
            flow_noto = calculate_flow_score(df_noto_interval, flow_questions[interval_idx])
            flow_ignoto = calculate_flow_score(df_ignoto_interval, flow_questions[interval_idx])
            
            # Aggiungi i punteggi validi alla lista dei punteggi del partecipante
            if flow_noto is not None:
                participant_flow_scores[participant].append(flow_noto)
            if flow_ignoto is not None:
                participant_flow_scores[participant].append(flow_ignoto)
    
    # Seconda passata: crea il dataframe con i punteggi normalizzati
    for participant, games in aggregated_amplitudes.items():
        # Salta se non abbiamo raccolto punteggi per questo partecipante
        if participant not in participant_flow_scores or len(participant_flow_scores[participant]) < 2:
            continue
            
        # Calcola media e deviazione standard per questo partecipante
        flow_scores = participant_flow_scores[participant]
        mean_flow = sum(flow_scores) / len(flow_scores)
        std_flow = (sum((x - mean_flow) ** 2 for x in flow_scores) / len(flow_scores)) ** 0.5
        
        # Se la deviazione standard è 0, impostiamo un valore minimo per evitare divisione per zero
        if std_flow == 0:
            std_flow = 1e-6
        
        # Determina il gruppo del partecipante (A o B)
        group_A = f"A_{participant}" in df_noto_dict
        group_B = f"B_{participant}" in df_noto_dict
        
        if not (group_A or group_B):
            continue
            
        participant_key = f"{'A' if group_A else 'B'}_{participant}"
        
        # Per gruppo B, il gioco noto è game2 ma i questionari sono in df_noto_dict
        game_noto_key = f"game{'1' if group_A else '2'}_{method}"
        game_ignoto_key = f"game{'2' if group_A else '1'}_{method}"
        
        # Recupera i dataframe
        df_noto = games.get(game_noto_key)
        df_ignoto = games.get(game_ignoto_key)
        df_noto_q = df_noto_dict.get(participant_key)
        df_ignoto_q = df_ignoto_dict.get(participant_key)
        
        if any(x is None or (isinstance(x, pd.DataFrame) and x.empty) for x in [df_noto, df_ignoto, df_noto_q, df_ignoto_q]):
            continue

        # Per ogni intervallo
        for interval_idx in range(3):
            interval_name = interval_map[interval_idx]
            questionnaire_type = interval_to_questionnaire[interval_idx]
            
            if interval_name not in df_noto_q or interval_name not in df_ignoto_q:
                continue
                
            # Dataframe dei questionari per questo intervallo
            df_noto_interval = df_noto_q[interval_name]
            df_ignoto_interval = df_ignoto_q[interval_name]
            
            # Calcola media flow per entrambi i giochi
            flow_noto = calculate_flow_score(df_noto_interval, flow_questions[interval_idx])
            flow_ignoto = calculate_flow_score(df_ignoto_interval, flow_questions[interval_idx])
            
            # Calcola i punteggi normalizzati usando la formula z-score: (x - μ) / σ
            normalized_flow_noto = (flow_noto - mean_flow) / std_flow if flow_noto is not None else None
            normalized_flow_ignoto = (flow_ignoto - mean_flow) / std_flow if flow_ignoto is not None else None
            
            if flow_noto is None and flow_ignoto is None:
                continue
                
            # Crea le righe per il dataframe con tutte le bande in un'unica riga
            if flow_noto is not None:
                row_noto = {
                    "Partecipant_ID": participant,
                    "Tipo_Gioco": "Gioco_Noto",
                    "Intervallo": questionnaire_type,
                    "Flow": flow_noto,
                    "Normalized_Flow": normalized_flow_noto
                }
                # Aggiungi i valori dei sensori per ogni banda
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
                # Aggiungi i valori dei sensori per ogni banda
                for banda in bands:
                    try:
                        ignoto_eeg = df_ignoto[banda].iloc[interval_idx]
                        for i, sensor in enumerate(sensors):
                            row_ignoto[f"{sensor}_{banda}"] = ignoto_eeg[i]
                    except Exception as e:
                        print(f"Errore con {participant}, intervallo {interval_idx}, banda {banda}: {e}")
                        for i, sensor in enumerate(sensors):
                            row_ignoto[f"{sensor}_{banda}"] = None
                
                all_data.append(row_ignoto)
    
    # Crea e salva il dataframe
    df_all_data = pd.DataFrame(all_data)
    
    # Verifica che ci siano dati prima di salvare
    if not df_all_data.empty:
        df_all_data = df_all_data.sort_values(by=["Partecipant_ID", "Tipo_Gioco"], ascending=[True, False])

        df_all_data.to_csv("df_flow_eeg_data.csv", index=False)
        print(f"Dataframe creato con successo con {len(df_all_data)} righe.")
    else:
        print("ATTENZIONE: Il dataframe risultante è vuoto!")
    
    return df_all_data

def calculate_flow_score(df_interval, questions):
    """Calcola il punteggio medio di flow per un set di domande."""
    scores = []
    for q in questions:
        q_scores = df_interval.loc[df_interval["Domanda"].astype(str) == str(q), "Punteggio"].values
        if len(q_scores) > 0:
            scores.append(q_scores[0])
    
    return sum(scores) / len(scores) if scores else None

# Attiva la conversione automatica tra Pandas e R
rpy2.robjects.pandas2ri.activate()

def run_mixed_models(df):
    """
    Esegue il modello completo e i modelli per ogni banda (Alpha, Beta, Delta, Gamma, Theta) in R.
    Calcola la devianza spiegata per ogni fixed effect e il power test.
    Salva i risultati in file separati.
    """

    # **Convertiamo Intervallo e Tipo_Gioco in numeri**
    mapping_intervallo = {"iGEQ_1": 1, "iGEQ_2": 2, "GEQ": 3}
    mapping_tipo_gioco = {"Gioco_Noto": 1, "Gioco_Ignoto": 0}

    df["Intervallo"] = df["Intervallo"].map(mapping_intervallo)
    df["Tipo_Gioco"] = df["Tipo_Gioco"].map(mapping_tipo_gioco)

    # **Rimuoviamo righe con NaN**
    df_cleaned = df.dropna()

    if df_cleaned.empty:
        raise ValueError("Errore: il DataFrame è vuoto dopo la rimozione dei NaN!")

    # **Passa il DataFrame a R**
    ro.globalenv['df'] = rpy2.robjects.pandas2ri.py2rpy(df_cleaned)

    # **Esegui lo script R**
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

    print("Risultati salvati nei file:")
    print("- full_model_results.txt")
    print("- alpha_model_results.txt")
    print("- beta_model_results.txt")
    print("- delta_model_results.txt")
    print("- gamma_model_results.txt")