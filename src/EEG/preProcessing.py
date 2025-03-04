import os
import json
import zipfile

def extract_and_fix_json_files(directory_path):
    """
    Estrae solo il file "power_by_band.json" dai file .zip presenti in una directory,
    corregge il suo formato e restituisce il percorso del file JSON corretto.

    Parametri:
        directory_path (str): Percorso della directory contenente i file .zip.

    Ritorna:
        list: Una lista di percorsi assoluti dei file "power_by_band.json" corretti.
    """
    json_file_paths = []

    for file in os.listdir(directory_path):
        if file.endswith(".zip"):
            zip_file_path = os.path.join(directory_path, file)
            extracted_path = os.path.splitext(zip_file_path)[0]

            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                if "power_by_band.json" in zip_ref.namelist():
                    zip_ref.extract("power_by_band.json", extracted_path)
                    json_file_path = os.path.join(extracted_path, "power_by_band.json")
                    
                    if fix_json_format(json_file_path):
                        json_file_paths.append(json_file_path)

    return json_file_paths


def fix_json_format(file_path):
    """
    Corregge il formato di un file JSON sostituendo '][' con '],['.

    Parametri:
        file_path (str): Percorso del file JSON da correggere.

    Ritorna:
        bool: True se il file è stato corretto, False altrimenti.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()

        if "}][{" in content:
            corrected_content = content.replace("][", ",")
            with open(file_path, 'w') as f:
                f.write(corrected_content)
            print(f"Corretto il formato del file: {file_path}")
            return True

    except Exception as e:
        print(f"Errore durante la correzione di {file_path}: {e}")

    return False

'''
def load_json_with_id(file_path, participant_id):
    """
    Carica i dati JSON da un file e aggiunge il participant ID solo al dizionario restituito.

    Parametri:
        file_path (str): Percorso del file JSON.
        participant_id (str): ID del partecipante.

    Ritorna:
        dict or list: I dati JSON caricati con il participant ID aggiunto.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Aggiungi il participant ID ai dati JSON (solo nel dizionario restituito)
        if isinstance(data, list):
            for entry in data:
                entry['participant_id'] = participant_id
        elif isinstance(data, dict):
            data['participant_id'] = participant_id

        return data

    except Exception as e:
        print(f"Errore durante il caricamento di {file_path}: {e}")
        return None
'''