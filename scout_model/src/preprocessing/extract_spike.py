from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import os
from pathlib import Path

class SpikeExtractor:
    SPIKE_START = 21563
    SPIKE_END = 25384

    @staticmethod
    def extract_spike_region(sequence):
        return sequence[SpikeExtractor.SPIKE_START:SpikeExtractor.SPIKE_END]

    @classmethod
    def process_fasta_file(cls, input_file, output_file):
        spike_sequences = []
        
        for record in SeqIO.parse(input_file, "fasta"):
            spike_seq = cls.extract_spike_region(str(record.seq))
            new_record = SeqRecord(
                Seq(spike_seq),
                id=record.id + "_spike",
                description=f"Spike protein sequence from {record.description}"
            )
            spike_sequences.append(new_record)
        
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        SeqIO.write(spike_sequences, output_file, "fasta")

    @classmethod
    def process_all_files(cls, input_dir, output_dir):
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for file_path in input_path.glob("*.fasta"):
            output_file = output_path / f"spike_{file_path.name}"
            cls.process_fasta_file(file_path, output_file)

if __name__ == "__main__":
    SpikeExtractor.process_all_files(
        "data/fasta_files",
        "data/spike_sequences"
    )
