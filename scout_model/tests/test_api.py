import pytest
from fastapi.testclient import TestClient
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch

# Import the FastAPI app and related components
from src.api.app import app, ModelInterface
from src.utils.constants import (
    SPIKE_COORDINATES,
    ERROR_MESSAGES,
    SEQUENCE_TEMPLATES,
    VALIDATION
)

# Create test client
client = TestClient(app)

# Test data
VALID_SPIKE_SEQUENCE = "".join(["A" * 1000, "T" * 1000, "G" * 1000, "C" * 1000])
INVALID_SEQUENCE = "INVALID123"
TEST_COUNTRY = "test_country"

@pytest.fixture
def mock_model_interface():
    with patch('src.api.app.ModelInterface') as MockModelInterface:
        mock_instance = Mock()
        mock_instance.generate_response.return_value = f"This COVID-19 sample originates from {TEST_COUNTRY}"
        MockModelInterface.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def sample_fasta_file(tmp_path):
    """Create a sample FASTA file for testing"""
    fasta_path = tmp_path / "test.fasta"
    record = SeqRecord(
        Seq(VALID_SPIKE_SEQUENCE),
        id="test_sequence",
        description="Test sequence"
    )
    SeqIO.write([record], fasta_path, "fasta")
    return fasta_path

class TestAPIEndpoints:
    """Test class for API endpoints"""

    def test_predict_valid_sequence(self, mock_model_interface):
        """Test prediction with valid sequence"""
        response = client.post(
            "/predict",
            json={"sequence": VALID_SPIKE_SEQUENCE}
        )
        
        assert response.status_code == 200
        assert TEST_COUNTRY in response.json()["response"]
        mock_model_interface.generate_response.assert_called_once()

    def test_predict_invalid_sequence(self):
        """Test prediction with invalid sequence"""
        response = client.post(
            "/predict",
            json={"sequence": INVALID_SEQUENCE}
        )
        
        assert response.status_code == 400
        assert "Invalid sequence format" in response.json()["detail"]

    def test_predict_empty_sequence(self):
        """Test prediction with empty sequence"""
        response = client.post(
            "/predict",
            json={"sequence": ""}
        )
        
        assert response.status_code == 400
        assert "Empty sequence" in response.json()["detail"]

    def test_predict_long_sequence(self):
        """Test prediction with sequence exceeding max length"""
        long_sequence = "A" * (VALIDATION["max_sequence_length"] + 1)
        response = client.post(
            "/predict",
            json={"sequence": long_sequence}
        )
        
        assert response.status_code == 400
        assert "exceeds maximum allowed length" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_predict_concurrent_requests(self, mock_model_interface):
        """Test handling multiple concurrent requests"""
        import asyncio
        
        async def make_request():
            response = client.post(
                "/predict",
                json={"sequence": VALID_SPIKE_SEQUENCE}
            )
            return response.status_code
        
        # Make 5 concurrent requests
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        assert all(status == 200 for status in results)

class TestModelInterface:
    """Test class for ModelInterface"""

    def test_model_initialization(self):
        """Test model interface initialization"""
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                interface = ModelInterface()
                assert interface.model is not None
                assert interface.tokenizer is not None

    def test_generate_response(self, mock_model_interface):
        """Test response generation"""
        response = mock_model_interface.generate_response(VALID_SPIKE_SEQUENCE)
        assert TEST_COUNTRY in response
        mock_model_interface.generate_response.assert_called_once()

    @patch('src.api.app.extract_spike_region')
    def test_spike_extraction(self, mock_extract_spike):
        """Test spike region extraction"""
        mock_extract_spike.return_value = VALID_SPIKE_SEQUENCE
        
        with patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
            with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer:
                interface = ModelInterface()
                interface.generate_response("FULL_GENOME_SEQUENCE")
                
                mock_extract_spike.assert_called_once()

class TestInputValidation:
    """Test class for input validation"""

    def test_sequence_validation(self):
        """Test sequence validation"""
        def validate_sequence(seq):
            response = client.post("/predict", json={"sequence": seq})
            return response.status_code
        
        # Test various invalid sequences
        assert validate_sequence("123456") == 400  # Invalid characters
        assert validate_sequence("ATCG" * 10000) == 400  # Too long
        assert validate_sequence("") == 400  # Empty
        assert validate_sequence("AT-CG") == 400  # Contains invalid characters

    def test_content_type_validation(self):
        """Test content type validation"""
        response = client.post(
            "/predict",
            data=VALID_SPIKE_SEQUENCE,  # Not JSON
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 415  # Unsupported Media Type

class TestErrorHandling:
    """Test class for error handling"""

    def test_model_error_handling(self, mock_model_interface):
        """Test handling of model errors"""
        mock_model_interface.generate_response.side_effect = Exception("Model error")
        
        response = client.post(
            "/predict",
            json={"sequence": VALID_SPIKE_SEQUENCE}
        )
        
        assert response.status_code == 500
        assert "Model error" in response.json()["detail"]

    def test_invalid_json_handling(self):
        """Test handling of invalid JSON"""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity

class TestPerformance:
    """Test class for performance metrics"""

    @pytest.mark.skipif(not os.getenv("RUN_PERFORMANCE_TESTS"), 
                       reason="Performance tests are disabled")
    def test_response_time(self, mock_model_interface):
        """Test API response time"""
        import time
        
        start_time = time.time()
        response = client.post(
            "/predict",
            json={"sequence": VALID_SPIKE_SEQUENCE}
        )
        end_time = time.time()
        
        assert (end_time - start_time) < 5  # Response should be under 5 seconds
        assert response.status_code == 200

if __name__ == "__main__":
    pytest.main([__file__])
