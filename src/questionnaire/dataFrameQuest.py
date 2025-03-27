import pandas as pd
import os

# Def of classes for each questionnaire
class Questionario:
    def __init__(self, nome, domande):
        self.nome = nome
        self.domande = domande

    def calcola_punteggio(self, risposte):
        return risposte[self.domande].mean(axis=1)

# category's item corrisponding to The Game Experience Questionnaire categories
class InGameGEQ(Questionario):
    def __init__(self):
        super().__init__("In-game GEQ", {
            "Competence": [2, 9],
            "Immersion": [1, 4],
            "Flow": [5, 10],
            "Tension": [6, 8],
            "Challenge": [12, 13],
            "Negative affect": [3, 7],
            "Positive affect": [11, 14]
        })

class CoreGEQ(Questionario):
    def __init__(self):
        super().__init__("Core GEQ", {
            "Competence": [2, 10, 15, 17, 21],
            "Immersion": [3, 12, 18, 19, 27, 30],
            "Flow": [5, 13, 25, 28, 31],
            "Tension": [22, 24, 29],
            "Challenge": [11, 23, 26, 32, 33],
            "Negative affect": [7, 8, 9, 16],
            "Positive affect": [1, 4, 6, 14, 20]
        })

class PostGameGEQ(Questionario):
    def __init__(self):
        super().__init__("Post-game GEQ", {
            "Positive Experience": [1, 5, 7, 8, 12, 16],
            "Negative Experience": [2, 4, 6, 11, 14, 15],
            "Tiredness": [10, 13],
            "Returning to Reality": [3, 9, 17]
        })

def carica_questionari(cartella_principale):
    partecipanti = sorted(os.listdir(cartella_principale))
    
    questionari = {
        "iGEQ": InGameGEQ(),
        "GEQ": CoreGEQ(),
        "Post-game": PostGameGEQ()
    }

    dataframes = []
    for partecipante in partecipanti:
        percorso_partecipante = os.path.join(cartella_principale, partecipante)
        if os.path.isdir(percorso_partecipante):
            for tipo, questionario in questionari.items():
                try:
                    df = pd.read_csv(os.path.join(percorso_partecipante, f"risposte_{tipo.lower().replace('-', '')}.csv"), header=None)
                    df = df.iloc[:, 1:]  # Rimuove timestamp from csv
                    df["Partecipante"] = partecipante
                    df["Tipo_Questionario"] = tipo
                    
                    # finds if it's "A" or "B"
                    gruppo = "A" if partecipante.startswith("A") else "B"
                    
                    # Label the rows based on the session (known or unknown)
                    if tipo == "iGEQ":
                        df["Sessione"] = ["Gioco Noto"] * 2 + ["Gioco Ignoto"] * 2 if gruppo == "A" else ["Gioco Ignoto"] * 2 + ["Gioco Noto"] * 2
                    elif tipo in ["GEQ", "Post-game"]:
                        df["Sessione"] = ["Gioco Noto", "Gioco Ignoto"] if gruppo == "A" else ["Gioco Ignoto", "Gioco Noto"]
                    
                    dataframes.append(df)
                except Exception as e:
                    print(f"Errore caricamento dati {tipo} per {partecipante}: {e}")

    df = pd.concat(dataframes, ignore_index=True)

    df_long = df.melt(id_vars=["Partecipante", "Tipo_Questionario", "Sessione"], var_name="Domanda", value_name="Punteggio")
    df_long["Domanda"] = df_long["Domanda"].astype(int)

    # assign categories
    def assegna_categoria(domanda, tipo_questionario):
        for categoria, domande in questionari[tipo_questionario].domande.items():
            if domanda in domande:
                return categoria
        return None

    df_long["Categoria"] = df_long.apply(lambda row: assegna_categoria(row["Domanda"], row["Tipo_Questionario"]), axis=1)
    df_long = df_long[df_long["Categoria"].notnull()]

    # separating based on the type of game
    df_noto = df_long[df_long["Sessione"] == "Gioco Noto"]
    df_ignoto = df_long[df_long["Sessione"] == "Gioco Ignoto"]

    # Filtering datas only for iGEQ and GEQ
    df_noto_filtrato = df_noto[df_noto["Tipo_Questionario"].isin(["iGEQ", "GEQ"])]
    df_ignoto_filtrato = df_ignoto[df_ignoto["Tipo_Questionario"].isin(["iGEQ", "GEQ"])]

    # Filtering data only for "Flow" Category
    df_noto_filtrato = df_noto_filtrato[df_noto_filtrato["Categoria"] == "Flow"]
    df_ignoto_filtrato = df_ignoto_filtrato[df_ignoto_filtrato["Categoria"] == "Flow"]

    # Creating dictionary for each partecipant
    def crea_dizionario(df, nome_file):
        dizionario = {}
        for partecipante in df["Partecipante"].unique():
            df_partecipante = df[df["Partecipante"] == partecipante] 
            df_igeq = df_partecipante[df_partecipante["Tipo_Questionario"] == "iGEQ"]
            df_selfreport_1 = pd.concat([df_igeq[df_igeq["Domanda"] == 5].iloc[:1], df_igeq[df_igeq["Domanda"] == 10].iloc[:1]])
            df_selfreport_2 = pd.concat([df_igeq[df_igeq["Domanda"] == 5].iloc[1:2], df_igeq[df_igeq["Domanda"] == 10].iloc[1:2]])
            df_selfreport_final = df_partecipante[df_partecipante["Tipo_Questionario"] == "GEQ"] 
            
            dizionario[partecipante] = {
                "df_selfreport_1": df_selfreport_1,
                "df_selfreport_2": df_selfreport_2,
                "df_selfreport_final": df_selfreport_final
            }
            
        return dizionario
    
    df_noto_dict = crea_dizionario(df_noto_filtrato, "df_noto")
    df_ignoto_dict = crea_dizionario(df_ignoto_filtrato, "df_ignoto")

    return df_noto_dict, df_ignoto_dict