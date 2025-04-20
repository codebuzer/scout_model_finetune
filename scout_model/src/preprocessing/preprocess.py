from pathlib import Path
import json
from transformers import AutoTokenizer
from Bio import SeqIO

class DataPreprocessor:
    def __init__(self, model_name, max_tokens=35000):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.max_tokens = max_tokens

    def check_sequence_length(self, sequence):
        tokens = self.tokenizer.encode(sequence)
        return len(tokens)

    def process_spike_sequences(self, data_dir):
        dataset = []
        data_path = Path(data_dir)
        
        for file_path in data_path.glob("spike_*.fasta"):
            country = file_path.stem.split('_')[1]
            
            for record in SeqIO.parse(file_path, "fasta"):
                sequence = str(record.seq)
                token_length = self.check_sequence_length(sequence)
                
                if token_length <= self.max_tokens:
                    dataset.append({
                        "sequence": sequence,
                        "country": country,
                        "instruction": "Based on this COVID-19 spike protein sequence, which country does this sample come from?",
                        "response": f"This COVID-19 sample originates from {country}. This classification is based on the genomic characteristics of the spike protein sequence.",
                        "token_length": token_length
                    })
                else:
                    print(f"Skipping sequence from {country} due to length: {token_length} tokens")

        return dataset

    def save_dataset(self, dataset, output_file):
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

if __name__ == "__main__":
    preprocessor = DataPreprocessor("meta-llama/Llama-4-Scout-17B-16E")
    dataset = preprocessor.process_spike_sequences("data/spike_sequences")
    preprocessor.save_dataset(dataset, "data/processed_dataset.json")
