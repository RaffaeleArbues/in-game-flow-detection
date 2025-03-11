## Metodo di Analisi (EEG)
Per ogni partecipante, la funzione **`spearman_corr_with_p`** esegue i seguenti passi:
1. **Segmentazione dei dati EEG** in tre intervalli temporali corrispondenti alle somministrazioni dei questionari.
2. **Associazione dei dati EEG con i questionari** relativi allo stesso partecipante.
3. **Estrazione delle ampiezze EEG** per ogni banda e elettrodo.
4. **Recupero dei punteggi dei questionari** somministrati nello stesso intervallo.
5. **Calcolo della correlazione di Spearman** tra i valori EEG e le risposte ai questionari.
6. **Strutturazione dei risultati** in un DataFrame per l'analisi finale.

## Struttura della funzione `spearman_corr_with_p`

### Input
La funzione accetta quattro parametri:

```python
def spearman_corr_with_p(aggregated_amplitudes, df_noto_dict, df_ignoto_dict, method):
```

| Parametro                 | Tipo  | Descrizione |
|---------------------------|-------|-------------|
| `aggregated_amplitudes`   | `dict` | Dizionario contenente i DataFrame delle ampiezze EEG per ogni partecipante e per ogni intervallo. |
| `df_noto_dict`            | `dict` | Dizionario con i punteggi dei questionari per il gioco noto per ogni partecipante. |
| `df_ignoto_dict`          | `dict` | Dizionario con i punteggi dei questionari per il gioco ignoto per ogni partecipante. |
| `method`                  | `str`  | Metodo utilizzato per calcolare l'ampiezza (es. "rms", "ptp"). |

### Flusso di esecuzione della funzione

#### **Segmentazione degli intervalli temporali**
- I dati EEG vengono suddivisi in *tre intervalli* corrispondenti ai momenti di somministrazione dei questionari.
- Ogni intervallo viene associato al rispettivo questionario:

```python
interval_map = {
    0: "df_selfreport_1",  # iGEQ - primo intervallo
    1: "df_selfreport_2",  # iGEQ - secondo intervallo
    2: "df_selfreport_final"  # GEQ - terzo intervallo
}
```

#### **Associazione dei dati EEG con i questionari dello stesso partecipante**
La funzione verifica a quale gruppo appartiene il partecipante di `aggregated_amplitudes` (**A o B**) e associa correttamente il gioco noto e ignoto.

```python
for participant, games in aggregated_amplitudes.items():
    if f"A_{participant}" in df_noto_dict:
        noto_dict = df_noto_dict
        ignoto_dict = df_ignoto_dict
        game_noto = f"game1_{method}"
        game_ignoto = f"game2_{method}"
    elif f"B_{participant}" in df_noto_dict:
        noto_dict = df_ignoto_dict
        ignoto_dict = df_noto_dict
        game_noto = f"game2_{method}"
        game_ignoto = f"game1_{method}"
```
#### **Recupera i DataFrame EEG e questionari corrispondenti**

```python
    df_noto = games.get(game_noto)
    df_ignoto = games.get(game_ignoto)
    df_noto_q = noto_dict.get(participant_key)
    df_ignoto_q = ignoto_dict.get(participant_key)
```
#### **Estrazione dei dati EEG per ogni intervallo**

```python
for banda in bands:
    noto_eeg_values = df_noto[banda].iloc[interval_idx]
    ignoto_eeg_values = df_ignoto[banda].iloc[interval_idx]
```

Per ogni **banda EEG** (alpha, beta, delta, gamma, theta), vengono estratti i valori delle ampiezze EEG corrispondenti all'intervallo temporale analizzato.

#### **Associazione EEG → Sensore e Punteggio Questionario**
Per ogni **sensore EEG**, viene estratto il valore dell'ampiezza EEG e associato alla risposta del questionario.

```python
noto_score = df_noto_q_interval.loc[
    (df_noto_q_interval["Domanda"].astype(str) == str(domanda)),
    "Punteggio"
].values
```

Se il punteggio del questionario **non esiste per quel partecipante**, il dato viene scartato.

#### **Aggregazione di tutti i dati alla lista per l'analisi della correlazione**
```python
all_data.append([interval_name, banda, sensore, f"{questionnaire_type}_Q{domanda}", noto_sensor_value, noto_score[0]])
all_data.append([interval_name, banda, sensore, f"{questionnaire_type}_Q{domanda}", ignoto_sensor_value, ignoto_score[0]])
```

#### **Calcolo della correlazione di Spearman**
Una volta ottenuto il dataframe `df_all_data`, raggruppo le colonne `["Banda", "Sensore", "Domanda"]` e calcolo la correlazione e il p-value tra ampiezza EEG e risposte ai questionari:

```python
for (banda, sensore, domanda), group in df_all_data.groupby(["Banda", "Sensore", "Domanda"]):
    spearman_corr, p_value = spearmanr(group["EEG_Valore"], group["Punteggio"])
```

#### **Strutturazione dei risultati in un DataFrame**

```python
results.append(["Aggregato", banda, sensore, domanda, spearman_corr, p_value, len(group)])
```

Il DataFrame risultante avrà la seguente struttura:

| Gioco     | Banda  | Sensore | Domanda | Spearman Corr | p-value | Num_Samples |
|-----------|--------|---------|---------|---------------|---------|-------------|
| Aggregato | alpha  | C3      | GEQ_Q13 | -0.42        | 0.065   | 20          |
| Aggregato | beta   | F5      | iGEQ_Q5 | 0.31         | 0.045   | 20          |
| Aggregato | delta  | CP3     | GEQ_Q31 | -0.27        | 0.09    | 20          |

