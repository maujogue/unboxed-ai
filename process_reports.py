import pandas as pd
import os

# Le nom de fichier listé dans le dossier assets est 'new-reports.ods' (avec un tiret)
input_path = os.path.join('assets', 'new-reports.ods')
output_path = os.path.join('assets', 'processed_reports.csv')

print(f"Tentative de lecture de {input_path}...")

try:
    # Lecture du fichier ODS
    # Note: Pandas nécessite 'odfpy' pour lire les fichiers .ods.
    # Si 'odfpy' n'est pas installé, cela lèvera une erreur.
    df = pd.read_excel(input_path, engine='odf')
    
    # Affichage des colonnes pour vérification
    print("Colonnes disponibles :", df.columns.tolist())
    
    col_name = 'Clinical information data (Pseudo reports)'
    
    if col_name in df.columns:
        # Création de la colonne 'fait'
        # On convertit en string pour éviter les erreurs sur des valeurs vides/numériques
        df['fait'] = df[col_name].fillna('').astype(str).apply(lambda x: len(x) > 100)
        
        # Affichage du DataFrame complet
        print("\nDataFrame complet :")
        print(df)
        
        # Sauvegarde (en CSV pour éviter les dépendances d'écriture Excel si possible)
        df.to_csv(output_path, index=False)
        print(f"\nFichier sauvegardé : {output_path}")
    else:
        print(f"\nErreur : La colonne '{col_name}' n'existe pas dans le fichier.")
        print("Colonnes trouvées :", df.columns.tolist())

except ImportError:
    print("\nERREUR MANQUANTE : La bibliothèque 'odfpy' est nécessaire pour lire les fichiers .ods avec pandas.")
    print("Comme demandé, aucune installation n'a été effectuée.")
except Exception as e:
    print(f"\nUne erreur est survenue : {e}")
