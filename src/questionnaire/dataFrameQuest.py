import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


# Definizione delle classi per i questionari
class Questionario:
    def __init__(self, nome, domande):
        self.nome = nome
        self.domande = domande

    def calcola_punteggio(self, risposte):
        return risposte[self.domande].mean(axis=1)

# gli item di cui si compongono le categorie sono quelli dei questionari, e sono categorie che fanno riferimento al The Game Experience Questionnaire
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

# Caricamento dei dati
cartella_principale = "D://Git//PARTECIPANTI Studio di Tesi - Dati Acquisizioni//rawDatasForPreProcessing//questionnaire"
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
                df = df.iloc[:, 1:]  # Rimuove il timestamp dal csv
                df["Partecipante"] = partecipante
                df["Tipo_Questionario"] = tipo
                
                # Determina se è "A" o "B" (cioè se è un partecipante che ha giocato prima al gioco noto e poi a quello ignoto o viceversa).
                gruppo = "A" if partecipante.startswith("A") else "B"
                
                # Etichetta le righe in base alla sessione (noto/ignoto)
                if tipo == "iGEQ":
                    df["Sessione"] = ["Gioco Noto"] * 2 + ["Gioco Ignoto"] * 2 if gruppo == "A" else ["Gioco Ignoto"] * 2 + ["Gioco Noto"] * 2
                elif tipo in ["GEQ", "Post-game"]:
                    df["Sessione"] = ["Gioco Noto", "Gioco Ignoto"] if gruppo == "A" else ["Gioco Ignoto", "Gioco Noto"]
                
                dataframes.append(df)
            except Exception as e:
                print(f"Errore caricamento dati {tipo} per {partecipante}: {e}")

# Concatenazione dei DataFrame
df = pd.concat(dataframes, ignore_index=True)

# Conversione in formato lungo
df_long = df.melt(id_vars=["Partecipante", "Tipo_Questionario", "Sessione"], var_name="Domanda", value_name="Punteggio")
df_long["Domanda"] = df_long["Domanda"].astype(int)

# Funzione per assegnare le categorie
def assegna_categoria(domanda, tipo_questionario):
    for categoria, domande in questionari[tipo_questionario].domande.items():
        if domanda in domande:
            return categoria
    return None

df_long["Categoria"] = df_long.apply(lambda row: assegna_categoria(row["Domanda"], row["Tipo_Questionario"]), axis=1)
df_long = df_long[df_long["Categoria"].notnull()]

# Separazione per gioco noto e ignoto
df_noto = df_long[df_long["Sessione"] == "Gioco Noto"]
df_ignoto = df_long[df_long["Sessione"] == "Gioco Ignoto"]

# Filtriamo solo i dati relativi a iGEQ e GEQ
df_noto_filtrato = df_noto[df_noto["Tipo_Questionario"].isin(["iGEQ", "GEQ"])]
df_ignoto_filtrato = df_ignoto[df_ignoto["Tipo_Questionario"].isin(["iGEQ", "GEQ"])]

# Creazione della figura con due subplot per confronto
plt.figure(figsize=(12, 6))

# Boxplot per il Gioco Noto
plt.subplot(1, 2, 1)
sns.boxplot(data=df_noto_filtrato, x="Categoria", y="Punteggio", hue="Tipo_Questionario", palette='Set2')
plt.title("Distribuzione Punteggi - Gioco Noto")
plt.xticks(rotation=45)
plt.xlabel("Categoria")
plt.ylabel("Punteggio")

# Boxplot per il Gioco Ignoto
plt.subplot(1, 2, 2)
sns.boxplot(data=df_ignoto_filtrato, x="Categoria", y="Punteggio", hue="Tipo_Questionario", palette='Set2')
plt.title("Distribuzione Punteggi - Gioco Ignoto")
plt.xticks(rotation=45)
plt.xlabel("Categoria")
plt.ylabel("Punteggio")

# Mostra il grafico
plt.tight_layout()
plt.show()

'''
# Creazione del plot senza la separazione dei questionari
plt.figure(figsize=(12, 6))

# Boxplot combinato
sns.boxplot(data=df_long[df_long["Tipo_Questionario"].isin(["iGEQ", "GEQ"])], 
            x="Categoria", y="Punteggio", 
            hue="Sessione", palette='Set2')

# Personalizzazione del grafico
plt.title("Distribuzione Punteggi per Categoria - Confronto Gioco Noto vs Ignoto")
plt.xticks(rotation=45)
plt.xlabel("Categoria")
plt.ylabel("Punteggio")
plt.legend(title="Sessione")

# Mostra il grafico
plt.tight_layout()
plt.show()
'''