"""
Test for streamlit_app.py load_model() function (Task 2.2)
"""

import os
import pytest
import joblib
from unittest.mock import patch, MagicMock


def test_load_model_missing_model_file():
    """Test that load_model raises FileNotFoundError when model file is missing."""
    # Import here to avoid streamlit initialization issues
    from streamlit_app import load_model
    
    # Clear the cache before testing
    load_model.clear()
    
    with patch('os.path.exists') as mock_exists:
        # Simulate model file missing, vectorizer exists
        mock_exists.side_effect = lambda path: 'vectorizer' in path
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_model()
        
        assert "Model file not found" in str(exc_info.value)


def test_load_model_missing_vectorizer_file():
    """Test that load_model raises FileNotFoundError when vectorizer file is missing."""
    from streamlit_app import load_model
    
    # Clear the cache before testing
    load_model.clear()
    
    with patch('os.path.exists') as mock_exists:
        # Simulate model exists, vectorizer file missing
        mock_exists.side_effect = lambda path: 'best_model' in path
        
        with pytest.raises(FileNotFoundError) as exc_info:
            load_model()
        
        assert "Vectorizer file not found" in str(exc_info.value)


def test_load_model_success():
    """Test that load_model successfully loads model and vectorizer when files exist."""
    from streamlit_app import load_model
    
    # Clear the cache before testing
    load_model.clear()
    
    mock_model = MagicMock()
    mock_vectorizer = MagicMock()
    
    with patch('os.path.exists', return_value=True):
        with patch('joblib.load') as mock_joblib_load:
            # First call returns model, second call returns vectorizer
            mock_joblib_load.side_effect = [mock_model, mock_vectorizer]
            
            model, vectorizer = load_model()
            
            assert model is mock_model
            assert vectorizer is mock_vectorizer
            assert mock_joblib_load.call_count == 2


def test_load_model_joblib_error():
    """Test that load_model handles joblib loading errors gracefully."""
    from streamlit_app import load_model
    
    # Clear the cache before testing
    load_model.clear()
    
    with patch('os.path.exists', return_value=True):
        with patch('joblib.load', side_effect=Exception("Corrupted file")):
            with pytest.raises(Exception) as exc_info:
                load_model()
            
            assert "Failed to load model files" in str(exc_info.value)


def test_load_model_returns_tuple():
    """Test that load_model returns a tuple of two objects."""
    from streamlit_app import load_model
    
    # Clear the cache before testing
    load_model.clear()
    
    mock_model = MagicMock()
    mock_vectorizer = MagicMock()
    
    with patch('os.path.exists', return_value=True):
        with patch('joblib.load') as mock_joblib_load:
            mock_joblib_load.side_effect = [mock_model, mock_vectorizer]
            
            result = load_model()
            
            assert isinstance(result, tuple)
            assert len(result) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
