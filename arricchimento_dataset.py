# Arricchimento Dataset per Matrici di Localizzazione

import json
import numpy as np
import random

class MatrixDatasetEnricher:
    def __init__(self, base_dataset_path='matrix_environment_dataset.json'):
        """
        Carica e arricchisce il dataset base di matrici
        
        Args:
            base_dataset_path (str): Percorso al file JSON del dataset base
        """
        # Carica dataset originale
        with open(base_dataset_path, 'r', encoding='utf-8') as f:
            self.base_dataset = json.load(f)
        
        # Liste di prompt e descrizioni
        self.location_prompts = [
            "Qual è la posizione di {object} nella matrice?",
            "Trova le coordinate di {object}",
            "In che cella si trova {object}?",
            "Localizza {object} nella griglia",
            "Dove si trova {object} nell'ambiente?"
        ]
        
        # Preparazione dataset arricchito
        self.enriched_dataset = []
    
    def enrich_dataset(self, augmentation_factor=5):
        """
        Arricchisce il dataset con variazioni e informazioni aggiuntive
        
        Args:
            augmentation_factor (int): Moltiplicatore per aumentare il dataset
        
        Returns:
            self: Istanza corrente per concatenazione di metodi
        """
        environments = self.base_dataset['environments']
        
        for env_index, environment in enumerate(environments):
            for _ in range(augmentation_factor):
                # Seleziona oggetto casuale
                object_location = self._find_random_object(environment)
                
                if object_location:
                    obj_type, (x, y) = object_location
                    
                    # Salta se è 'nothing'
                    if obj_type == 'nothing':
                        continue
                    
                    # Genera prompt
                    prompt = random.choice(self.location_prompts).format(object=obj_type)
                    
                    # Prepara esempio di training
                    training_example = {
                        "environment_id": env_index,
                        "input": prompt,
                        "object": obj_type,
                        "coordinates": {
                            "x": x,
                            "y": y
                        },
                        "full_environment": environment
                    }
                    
                    self.enriched_dataset.append(training_example)
        
        return self
    
    def _find_random_object(self, environment):
        """
        Trova un oggetto casuale nella matrice
        
        Args:
            environment (list): Matrice dell'ambiente
        
        Returns:
            tuple: (tipo_oggetto, (x, y)) o None
        """
        # Trova tutti gli oggetti non 'nothing'
        object_locations = [
            (env_type, (x, y)) 
            for x in range(len(environment))
            for y in range(len(environment[x]))
            if (env_type := environment[x][y]) != 'nothing'
        ]
        
        # Restituisci oggetto casuale se presente
        return random.choice(object_locations) if object_locations else None
    
    def save_enriched_dataset(self, output_path='enriched_matrix_dataset.json'):
        """
        Salva il dataset arricchito
        
        Args:
            output_path (str): Percorso di output
        
        Returns:
            self: Istanza corrente per concatenazione di metodi
        """
        # Prepara metadati
        metadata = self.base_dataset['metadata'].copy()
        metadata.update({
            'enriched_samples': len(self.enriched_dataset),
            'augmentation_method': 'matrix_location_sampling'
        })
        
        # Salva dataset
        enriched_dataset = {
            "dataset": self.enriched_dataset,
            "metadata": metadata
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset arricchito salvato: {output_path}")
        print(f"Numero di campioni: {len(self.enriched_dataset)}")
        
        return self

# Esempio di utilizzo
if __name__ == "__main__":
    # Carica e arricchisci il dataset
    enricher = MatrixDatasetEnricher('matrix_environment_dataset.json')
    
    # Genera dataset arricchito
    enricher.enrich_dataset(augmentation_factor=5)
    
    # Salva il dataset
    enricher.save_enriched_dataset('enriched_matrix_dataset.json')