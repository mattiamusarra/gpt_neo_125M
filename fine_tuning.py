import os
import json
import torch
import numpy as np
from transformers import (
    GPTNeoForCausalLM, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset

class MatrixLocationFineTuning:
    def __init__(self, 
                 model_name='EleutherAI/gpt-neo-125M', 
                 dataset_path='enriched_matrix_dataset.json'):
        """
        Inizializzazione del fine-tuning per localizzazione matriciale
        
        Args:
            model_name (str): Nome del modello base
            dataset_path (str): Percorso al dataset arricchito
        """
        # Configurazione del device
        self.device = self._select_device()
        
        # Caricamento modello e tokenizer
        print(f"🔄 Caricamento modello: {model_name}")
        self.model = GPTNeoForCausalLM.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        
        # Configurazione tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Sposta modello sul device corretto
        self.model = self.model.to(self.device)
        
        # Caricamento dataset
        self.dataset = self._prepare_dataset(dataset_path)
    
    def _select_device(self):
        """
        Selezione ottimale del device per Machine Learning
        
        Returns:
            torch.device: Device di computazione
        """
        if torch.backends.mps.is_available():
            print("🍎 Utilizzo Metal Performance Shaders (MPS)")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("🚀 Utilizzo CUDA GPU")
            return torch.device("cuda")
        else:
            print("💻 Utilizzo CPU")
            return torch.device("cpu")
    
    def _prepare_dataset(self, dataset_path):
        """
        Carica e prepara il dataset per il training
        
        Args:
            dataset_path (str): Percorso al file del dataset
        
        Returns:
            Dataset: Dataset processato per Hugging Face Transformers
        """
        # Carica dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Estrai input e target - formato più esplicito
        formatted_texts = []
        for entry in data['dataset']:
            # Crea un testo che include sia input che output in formato istruzione-risposta
            text = (
                f"Input: {entry['input']}\n"
                f"Output: Oggetto: {entry['object']}\n"
                f"Coordinate: x={entry['coordinates']['x']}, "
                f"y={entry['coordinates']['y']}\n"
                f"Ambiente: Matrice con {len(entry['full_environment'])}x{len(entry['full_environment'][0])} celle"
            )
            formatted_texts.append(text)
        
        # Tokenizzazione
        print("🔤 Tokenizzazione del dataset")
        encodings = self.tokenizer(
            formatted_texts, 
            padding=True, 
            truncation=True, 
            max_length=256,  # Aumentato per accogliere input e output insieme
            return_tensors='pt'
        )
        
        # Converti in Hugging Face Dataset per Causal Language Modeling
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': encodings['input_ids'].clone()  # Per causal LM, le labels sono uguali agli input
        })
    
    def fine_tune(self, 
                  output_dir='./matrix_location_model',
                  learning_rate=5e-5,  # Leggermente aumentato
                  batch_size=4,
                  epochs=4):  # Leggermente aumentato
        """
        Esegue il fine-tuning del modello
        
        Args:
            output_dir (str): Directory per salvare il modello
            learning_rate (float): Velocità di apprendimento
            batch_size (int): Dimensione del batch
            epochs (int): Numero di epoche
        
        Returns:
            tuple: Modello e tokenizer fine-tuned
        """
        # Parametri di base compatibili con versioni più vecchie
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            warmup_steps=100,  # Aumentato per un warmup più graduale
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
            save_total_limit=2,
            dataloader_num_workers=0,  # Importante per MPS
            # Aggiunto gradient accumulation per stabilità
            gradient_accumulation_steps=2
        )
        
        # Preparazione data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=data_collator
        )
        
        # Fine-tuning
        print("🏋️ Inizio fine-tuning...")
        trainer.train()
        
        # Salvataggio
        print("💾 Salvataggio modello e tokenizer...")
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        return self.model, self.tokenizer
    
    def generate_location(self, input_text):
        """
        Genera posizione per un oggetto
        
        Args:
            input_text (str): Richiesta di localizzazione
        
        Returns:
            str: Posizione generata
        """
        # Formato più esplicito
        prompt = f"Input: {input_text}\nOutput: "
        
        # Preparazione input con attention mask
        inputs = self.tokenizer(
            prompt, 
            return_tensors='pt',
            padding=True
        ).to(self.device)
        
        # Generazione con parametri migliorati
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Importante!
                max_length=200,  # Aumentato
                min_length=20,   # Aggiunto per evitare risposte troppo brevi
                num_return_sequences=1,
                temperature=0.8,  # Aggiustato
                do_sample=True,   # Attivato sampling
                top_p=0.92,       # Nucleus sampling
                top_k=50,         # Top-k sampling
                no_repeat_ngram_size=2,  # Evita ripetizioni
                pad_token_id=self.tokenizer.eos_token_id  # Importante!
            )
        
        # Decodifica
        generated_text = self.tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        # Estrai solo la parte di output
        output_parts = generated_text.split("Output: ")
        if len(output_parts) > 1:
            return output_parts[1].strip()
        return generated_text.strip()

# Esempio di utilizzo
if __name__ == "__main__":
    # Verifica e notifica requisiti
    import transformers
    print(f"🔍 Versione di Transformers: {transformers.__version__}")
    
    try:
        import accelerate
        print(f"✅ Versione accelerate: {accelerate.__version__}")
    except ImportError:
        print("❌ Libreria 'accelerate' non trovata. Installala con:")
        print("  pip install 'accelerate>=0.26.0'")
        exit(1)  # Esci se manca accelerate
    
    # Istanza per fine-tuning
    fine_tuner = MatrixLocationFineTuning(
        dataset_path='enriched_matrix_dataset.json'
    )
    
    # Controlla se esiste già un modello addestrato
    model_path = './matrix_location_model'
    if os.path.exists(model_path) and os.path.isdir(model_path):
        print("🔄 Caricamento modello già addestrato...")
        fine_tuner.model = GPTNeoForCausalLM.from_pretrained(model_path)
        fine_tuner.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        fine_tuner.model = fine_tuner.model.to(fine_tuner.device)
    else:
        # Esegui fine-tuning
        print("🏋️ Addestramento nuovo modello...")
        fine_tuned_model, fine_tuned_tokenizer = fine_tuner.fine_tune(
            output_dir=model_path,
            learning_rate=5e-5,
            batch_size=4,
            epochs=4
        )
    
    # Test di generazione
    test_inputs = [
        "Dov'è la bottiglia nella matrice?",
        "Posizione dell'ostacolo",
        "Localizza il libro"
    ]
    
    print("\n📊 TEST GENERAZIONE:")
    print("====================")
    for test_input in test_inputs:
        result = fine_tuner.generate_location(test_input)
        print(f"\n🔍 Input: {test_input}")
        print(f"📍 Output: {result}")