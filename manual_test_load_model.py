"""
Manual test for streamlit_app.py load_model() function (Task 2.2)
This test verifies the function behavior without requiring pytest.
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Create a proper mock for streamlit with cache_resource decorator
def mock_cache_resource(func):
    """Mock decorator that just returns the function unchanged."""
    func.clear = lambda: None  # Add clear method for cache clearing
    return func

streamlit_mock = MagicMock()
streamlit_mock.cache_resource = mock_cache_resource
streamlit_mock.error = MagicMock()

# Mock streamlit before importing streamlit_app
sys.modules['streamlit'] = streamlit_mock

def test_load_model_missing_model_file():
    """Test that load_model raises FileNotFoundError when model file is missing."""
    print("\n[TEST 1] Testing missing model file...")
    
    # Import after mocking streamlit
    from streamlit_app import load_model
    
    # Clear the cache
    load_model.clear()
    
    with patch('os.path.exists') as mock_exists:
        # Simulate model file missing, vectorizer exists
        mock_exists.side_effect = lambda path: 'vectorizer' in path
        
        try:
            load_model()
            print("❌ FAILED: Expected FileNotFoundError but none was raised")
            return False
        except FileNotFoundError as e:
            if "Model file not found" in str(e):
                print("✅ PASSED: Correctly raised FileNotFoundError for missing model")
                return True
            else:
                print(f"❌ FAILED: Wrong error message: {e}")
                return False


def test_load_model_missing_vectorizer_file():
    """Test that load_model raises FileNotFoundError when vectorizer file is missing."""
    print("\n[TEST 2] Testing missing vectorizer file...")
    
    from streamlit_app import load_model
    
    # Clear the cache
    load_model.clear()
    
    with patch('os.path.exists') as mock_exists:
        # Simulate model exists, vectorizer file missing
        mock_exists.side_effect = lambda path: 'best_model' in path
        
        try:
            load_model()
            print("❌ FAILED: Expected FileNotFoundError but none was raised")
            return False
        except FileNotFoundError as e:
            if "Vectorizer file not found" in str(e):
                print("✅ PASSED: Correctly raised FileNotFoundError for missing vectorizer")
                return True
            else:
                print(f"❌ FAILED: Wrong error message: {e}")
                return False


def test_load_model_success():
    """Test that load_model successfully loads model and vectorizer when files exist."""
    print("\n[TEST 3] Testing successful model loading...")
    
    from streamlit_app import load_model
    
    # Clear the cache
    load_model.clear()
    
    mock_model = MagicMock()
    mock_vectorizer = MagicMock()
    
    with patch('os.path.exists', return_value=True):
        with patch('joblib.load') as mock_joblib_load:
            # First call returns model, second call returns vectorizer
            mock_joblib_load.side_effect = [mock_model, mock_vectorizer]
            
            model, vectorizer = load_model()
            
            if model is mock_model and vectorizer is mock_vectorizer:
                print("✅ PASSED: Successfully loaded model and vectorizer")
                return True
            else:
                print("❌ FAILED: Returned objects don't match expected mocks")
                return False


def test_load_model_returns_tuple():
    """Test that load_model returns a tuple of two objects."""
    print("\n[TEST 4] Testing return type is tuple...")
    
    from streamlit_app import load_model
    
    # Clear the cache
    load_model.clear()
    
    mock_model = MagicMock()
    mock_vectorizer = MagicMock()
    
    with patch('os.path.exists', return_value=True):
        with patch('joblib.load') as mock_joblib_load:
            mock_joblib_load.side_effect = [mock_model, mock_vectorizer]
            
            result = load_model()
            
            if isinstance(result, tuple) and len(result) == 2:
                print("✅ PASSED: Returns tuple of 2 elements")
                return True
            else:
                print(f"❌ FAILED: Expected tuple of 2, got {type(result)} with {len(result) if hasattr(result, '__len__') else 'N/A'} elements")
                return False


def test_load_model_joblib_error():
    """Test that load_model handles joblib loading errors gracefully."""
    print("\n[TEST 5] Testing joblib loading error handling...")
    
    from streamlit_app import load_model
    
    # Clear the cache
    load_model.clear()
    
    with patch('os.path.exists', return_value=True):
        with patch('joblib.load', side_effect=Exception("Corrupted file")):
            try:
                load_model()
                print("❌ FAILED: Expected Exception but none was raised")
                return False
            except Exception as e:
                if "Failed to load model files" in str(e):
                    print("✅ PASSED: Correctly handled joblib loading error")
                    return True
                else:
                    print(f"❌ FAILED: Wrong error message: {e}")
                    return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing load_model() function - Task 2.2")
    print("=" * 60)
    
    tests = [
        test_load_model_missing_model_file,
        test_load_model_missing_vectorizer_file,
        test_load_model_success,
        test_load_model_returns_tuple,
        test_load_model_joblib_error,
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ FAILED: Test raised unexpected exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✅ All tests passed!")
        return 0
    else:
        print("\n❌ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
