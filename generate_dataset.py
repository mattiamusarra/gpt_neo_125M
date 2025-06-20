# Generatore di Dataset a Matrice per Localizzazione Oggetti

import numpy as np
import json
import random

class MatrixEnvironmentGenerator:
    def __init__(self, 
                 grid_size=(10, 10),  # Dimensione griglia
                 object_types=['nothing', 'water_bottle', 'obstacle', 'book', 'remote']):
        """
        Generatore di ambienti rappresentati come matrici
        
        Args:
            grid_size (tuple): Dimensioni della griglia (x, y)
            object_types (list): Tipi di oggetti possibili
        """
        self.grid_size = grid_size
        self.object_types = object_types
        
        # Probabilità di distribuzione degli oggetti
        self.object_distribution = {
            'nothing': 0.6,
            'water_bottle': 0.1,
            'obstacle': 0.2,
            'book': 0.05,
            'remote': 0.05
        }
    
    def generate_environment(self, num_environments=100):
        """
        Genera multipli ambienti come matrici
        
        Args:
            num_environments (int): Numero di ambienti da generare
        
        Returns:
            list: Lista di ambienti (matrici)
        """
        environments = []
        
        for _ in range(num_environments):
            # Genera una matrice vuota
            environment = np.full(self.grid_size, 'nothing', dtype=object)
            
            # Popola la matrice
            for x in range(self.grid_size[0]):
                for y in range(self.grid_size[1]):
                    # Determina tipo di oggetto basato su distribuzione
                    object_type = self._select_object_type()
                    environment[x, y] = object_type
            
            environments.append(environment.tolist())
        
        return environments
    
    def _select_object_type(self):
        """
        Seleziona un tipo di oggetto basato su distribuzione probabilistica
        
        Returns:
            str: Tipo di oggetto
        """
        return np.random.choice(
            list(self.object_distribution.keys()), 
            p=list(self.object_distribution.values())
        )
    
    def generate_labeled_dataset(self, num_environments=100):
        """
        Genera un dataset con etichette per task di localizzazione
        
        Args:
            num_environments (int): Numero di ambienti da generare
        
        Returns:
            dict: Dataset con ambienti e metadati
        """
        # Genera ambienti
        environments = self.generate_environment(num_environments)
        
        # Preparazione dataset
        dataset = {
            "environments": environments,
            "metadata": {
                "grid_size": self.grid_size,
                "total_environments": num_environments,
                "object_types": self.object_types,
                "object_distribution": self.object_distribution,
                "version": "1.0.0",
                "date_created": "2024-05-12"
            }
        }
        
        return dataset
    
    def save_dataset(self, dataset, filename='matrix_environment_dataset.json'):
        """
        Salva il dataset su file
        
        Args:
            dataset (dict): Dataset da salvare
            filename (str): Nome del file
        """
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"Dataset salvato: {filename}")
        print(f"Numero di ambienti: {len(dataset['environments'])}")
    
    def analyze_dataset(self, dataset):
        """
        Analizza la distribuzione degli oggetti nel dataset
        
        Args:
            dataset (dict): Dataset generato
        """
        # Calcola distribuzione oggetti
        object_counts = {}
        for env in dataset['environments']:
            for row in env:
                for cell in row:
                    object_counts[cell] = object_counts.get(cell, 0) + 1
        
        # Calcola percentuali
        total_cells = len(dataset['environments']) * self.grid_size[0] * self.grid_size[1]
        
        print("\n--- Analisi Distribuzione Oggetti ---")
        for obj_type, count in object_counts.items():
            percentage = (count / total_cells) * 100
            print(f"{obj_type}: {count} celle ({percentage:.2f}%)")

# Esempio di utilizzo
if __name__ == "__main__":
    # Genera dataset
    generator = MatrixEnvironmentGenerator(
        grid_size=(10, 10),
        object_types=['nothing', 'water_bottle', 'obstacle', 'book', 'remote']
    )
    
    # Genera e salva dataset
    dataset = generator.generate_labeled_dataset(num_environments=1000)
    generator.save_dataset(dataset)
    
    # Analizza dataset
    generator.analyze_dataset(dataset)