import pytest
from pathlib import Path
import json
import shutil
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from unittest.mock import Mock, patch

from src.preprocessing.extract_spike import SpikeExtractor
from src.preprocessing.preprocess import DataPreprocessor
from src.utils.constants import (
    SPIKE_COORDINATES,
    MAX_TOKEN_LENGTH,
    SEQUENCE_TEMPLATES,
    VALIDATION
)

# Test data
TEST_SEQUENCE = "A" * SPIKE_COORDINATES["START"] + "SPIKE" * 760 + "G" * 1000
TEST_COUNTRIES = ["australia", "london", "america"]

@pytest.fixture
def test_data_dir(tmp_path):
    """Create temporary test data directory with sample FASTA files"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # Create sample FASTA files for each test country
    for country in TEST_COUNTRIES:
        fasta_path = data_dir / f"{country}.fasta"
        records = [
            SeqRecord(
                Seq(TEST_SEQUENCE),
                id=f"test_{country}_1",
                description=f"Test sequence for {country}"
            ),
            SeqRecord(
                Seq(TEST_SEQUENCE),
                id=f"test_{country}_2",
                description=f"Test sequence for {country}"
            )
        ]
        SeqIO.write(records, fasta_path, "fasta")
    
    return data_dir

@pytest.fixture
def test_spike_dir(tmp_path):
    """Create temporary directory for spike sequences"""
    spike_dir = tmp_path / "spike_sequences"
    spike_dir.mkdir()
    return spike_dir

@pytest.fixture
def mock_tokenizer():
    """Mock tokenizer for testing"""
    mock = Mock()
    mock.encode.return_value = ["token"] * 1000  # Simulate reasonable token length
    return mock

class TestSpikeExtractor:
    """Test class for spike sequence extraction"""

    def test_extract_spike_region(self):
        """Test extraction of spike region from sequence"""
        sequence = "A" * SPIKE_COORDINATES["START"] + "SPIKE" * 760 + "G" * 1000
        spike_sequence = SpikeExtractor.extract_spike_region(sequence)
        
        assert len(spike_sequence) == SPIKE_COORDINATES["END"] - SPIKE_COORDINATES["START"]
        assert spike_sequence.startswith("SPIKE")

    def test_process_single_fasta_file(self, tmp_path):
        """Test processing of a single FASTA file"""
        # Create test input file
        input_file = tmp_path / "test.fasta"
        record = SeqRecord(
            Seq(TEST_SEQUENCE),
            id="test_sequence",
            description="Test sequence"
        )
        SeqIO.write([record], input_file, "fasta")
        
        # Create output file
        output_file = tmp_path / "spike_test.fasta"
        
        # Process file
        SpikeExtractor.process_fasta_file(input_file, output_file)
        
        # Verify output
        assert output_file.exists()
        records = list(SeqIO.parse(output_file, "fasta"))
        assert len(records) == 1
        assert len(records[0].seq) == SPIKE_COORDINATES["END"] - SPIKE_COORDINATES["START"]

    def test_process_all_files(self, test_data_dir, test_spike_dir):
        """Test processing of multiple FASTA files"""
        # Process all files
        SpikeExtractor.process_all_files(test_data_dir, test_spike_dir)
        
        # Verify outputs
        for country in TEST_COUNTRIES:
            spike_file = test_spike_dir / f"spike_{country}.fasta"
            assert spike_file.exists()
            records = list(SeqIO.parse(spike_file, "fasta"))
            assert len(records) == 2  # Two sequences per country
            
            for record in records:
                assert len(record.seq) == SPIKE_COORDINATES["END"] - SPIKE_COORDINATES["START"]

    def test_invalid_sequence_handling(self, tmp_path):
        """Test handling of invalid sequences"""
        # Create test file with invalid sequence
        input_file = tmp_path / "invalid.fasta"
        record = SeqRecord(
            Seq("INVALID"),
            id="invalid_sequence",
            description="Invalid sequence"
        )
        SeqIO.write([record], input_file, "fasta")
        
        output_file = tmp_path / "spike_invalid.fasta"
        
        # Process should handle short sequence gracefully
        with pytest.raises(IndexError):
            SpikeExtractor.process_fasta_file(input_file, output_file)

class TestDataPreprocessor:
    """Test class for data preprocessing"""

    @pytest.fixture
    def preprocessor(self, mock_tokenizer):
        """Create DataPreprocessor instance with mock tokenizer"""
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
            mock_auto_tokenizer.return_value = mock_tokenizer
            return DataPreprocessor("test_model", max_tokens=MAX_TOKEN_LENGTH)

    def test_check_sequence_length(self, preprocessor):
        """Test sequence length checking"""
        # Test sequence within limits
        assert preprocessor.check_sequence_length("ATCG" * 100) == 1000
        
        # Modify mock for long sequence
        preprocessor.tokenizer.encode.return_value = ["token"] * (MAX_TOKEN_LENGTH + 1)
        length = preprocessor.check_sequence_length("LONG" * 10000)
        assert length > MAX_TOKEN_LENGTH

    def test_process_spike_sequences(self, preprocessor, test_spike_dir):
        """Test processing of spike sequences"""
        # Create test spike sequences
        for country in TEST_COUNTRIES:
            spike_file = test_spike_dir / f"spike_{country}.fasta"
            record = SeqRecord(
                Seq("ATCG" * 1000),
                id=f"test_{country}",
                description=f"Test spike sequence for {country}"
            )
            SeqIO.write([record], spike_file, "fasta")
        
        # Process sequences
        dataset = preprocessor.process_spike_sequences(test_spike_dir)
        
        # Verify dataset
        assert len(dataset) == len(TEST_COUNTRIES)
        for item in dataset:
            assert "sequence" in item
            assert "country" in item
            assert "instruction" in item
            assert "response" in item
            assert item["country"] in TEST_COUNTRIES
            assert SEQUENCE_TEMPLATES["instruction"] in item["instruction"]

    def test_save_dataset(self, preprocessor, tmp_path):
        """Test dataset saving functionality"""
        output_file = tmp_path / "test_dataset.json"
        test_dataset = [
            {
                "sequence": "ATCG" * 100,
                "country": "test_country",
                "instruction": SEQUENCE_TEMPLATES["instruction"],
                "response": "Test response"
            }
        ]
        
        # Save dataset
        preprocessor.save_dataset(test_dataset, output_file)
        
        # Verify saved file
        assert output_file.exists()
        with open(output_file) as f:
            loaded_dataset = json.load(f)
        assert loaded_dataset == test_dataset

    def test_sequence_validation(self, preprocessor):
        """Test sequence validation"""
        # Test invalid characters
        invalid_seq = "ATCG123"
        with pytest.raises(ValueError):
            preprocessor.process_spike_sequences([invalid_seq])
        
        # Test sequence length limits
        too_short = "A" * (VALIDATION["min_sequence_length"] - 1)
        too_long = "A" * (VALIDATION["max_sequence_length"] + 1)
        
        with pytest.raises(ValueError):
            preprocessor.process_spike_sequences([too_short])
        
        with pytest.raises(ValueError):
            preprocessor.process_spike_sequences([too_long])

class TestIntegration:
    """Integration tests for preprocessing pipeline"""

    def test_full_preprocessing_pipeline(self, test_data_dir, test_spike_dir, tmp_path):
        """Test the complete preprocessing pipeline"""
        # 1. Extract spike sequences
        SpikeExtractor.process_all_files(test_data_dir, test_spike_dir)
        
        # 2. Preprocess data
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = ["token"] * 1000
            mock_auto_tokenizer.return_value = mock_tokenizer
            
            preprocessor = DataPreprocessor("test_model", max_tokens=MAX_TOKEN_LENGTH)
            dataset = preprocessor.process_spike_sequences(test_spike_dir)
            
            output_file = tmp_path / "final_dataset.json"
            preprocessor.save_dataset(dataset, output_file)
        
        # Verify final output
        assert output_file.exists()
        with open(output_file) as f:
            final_dataset = json.load(f)
        
        assert len(final_dataset) == len(TEST_COUNTRIES) * 2  # 2 sequences per country
        for item in final_dataset:
            assert all(key in item for key in ["sequence", "country", "instruction", "response"])

    @pytest.mark.parametrize("country", TEST_COUNTRIES)
    def test_country_specific_processing(self, country, test_data_dir, test_spike_dir, tmp_path):
        """Test preprocessing for specific countries"""
        # Process single country
        country_file = test_data_dir / f"{country}.fasta"
        spike_file = test_spike_dir / f"spike_{country}.fasta"
        
        SpikeExtractor.process_fasta_file(country_file, spike_file)
        
        with patch('transformers.AutoTokenizer.from_pretrained') as mock_auto_tokenizer:
            mock_tokenizer = Mock()
            mock_tokenizer.encode.return_value = ["token"] * 1000
            mock_auto_tokenizer.return_value = mock_tokenizer
            
            preprocessor = DataPreprocessor("test_model", max_tokens=MAX_TOKEN_LENGTH)
            dataset = preprocessor.process_spike_sequences(test_spike_dir)
            
            # Verify country-specific data
            country_samples = [item for item in dataset if item["country"] == country]
            assert len(country_samples) > 0
            for sample in country_samples:
                assert sample["country"] == country

if __name__ == "__main__":
    pytest.main([__file__])
