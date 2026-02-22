"""
Test for streamlit_app.py render_model_details() function (Task 7)
"""

import pytest
from unittest.mock import patch, MagicMock, call


def test_render_model_details_with_fake_prediction():
    """Test that render_model_details correctly displays fake prediction (0)."""
    from streamlit_app import render_model_details
    
    # Mock result dictionary with fake prediction
    result = {
        "model_prediction": 0,
        "confidence": 85,
        "pattern_score": 0.75,
        "key_indicators": [
            "High sensational language detected",
            "Multiple vague source references",
            "Conspiracy framing present"
        ]
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_model_details(result)
        
        # Verify expander was created
        mock_st.expander.assert_called_once()
        assert "Model Prediction Details" in str(mock_st.expander.call_args)
        
        # Verify markdown calls were made
        assert mock_st.markdown.call_count > 0
        
        # Check that FAKE (0) interpretation is in the calls
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("FAKE (0)" in call for call in markdown_calls)
        assert any("not credible" in call for call in markdown_calls)


def test_render_model_details_with_real_prediction():
    """Test that render_model_details correctly displays real prediction (1)."""
    from streamlit_app import render_model_details
    
    # Mock result dictionary with real prediction
    result = {
        "model_prediction": 1,
        "confidence": 92,
        "pattern_score": 0.25,
        "key_indicators": [
            "Evidence-based reporting",
            "Clear source attribution",
            "Balanced perspective"
        ]
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_model_details(result)
        
        # Check that REAL (1) interpretation is in the calls
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("REAL (1)" in call for call in markdown_calls)
        assert any("credible" in call for call in markdown_calls)


def test_render_model_details_displays_confidence():
    """Test that render_model_details displays confidence score as percentage."""
    from streamlit_app import render_model_details
    
    result = {
        "model_prediction": 1,
        "confidence": 78,
        "pattern_score": 0.5,
        "key_indicators": ["Test indicator"]
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_model_details(result)
        
        # Check that confidence percentage is displayed
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("78%" in call for call in markdown_calls)
        assert any("confidence" in call.lower() for call in markdown_calls)


def test_render_model_details_displays_pattern_score():
    """Test that render_model_details displays pattern score with interpretation."""
    from streamlit_app import render_model_details
    
    # Test high pattern score (suspicious)
    result = {
        "model_prediction": 0,
        "confidence": 85,
        "pattern_score": 0.75,
        "key_indicators": ["Test"]
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_model_details(result)
        
        # Check that pattern score is displayed as percentage
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("75%" in call for call in markdown_calls)
        assert any("High suspicious pattern" in call or "red flags" in call for call in markdown_calls)


def test_render_model_details_displays_key_indicators():
    """Test that render_model_details displays all key indicators as bulleted list."""
    from streamlit_app import render_model_details
    
    result = {
        "model_prediction": 0,
        "confidence": 80,
        "pattern_score": 0.6,
        "key_indicators": [
            "Indicator 1",
            "Indicator 2",
            "Indicator 3"
        ]
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_model_details(result)
        
        # Check that all indicators are displayed
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Indicator 1" in call for call in markdown_calls)
        assert any("Indicator 2" in call for call in markdown_calls)
        assert any("Indicator 3" in call for call in markdown_calls)


def test_render_model_details_handles_empty_indicators():
    """Test that render_model_details handles empty key indicators list."""
    from streamlit_app import render_model_details
    
    result = {
        "model_prediction": 1,
        "confidence": 90,
        "pattern_score": 0.2,
        "key_indicators": []
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_model_details(result)
        
        # Check that a default message is displayed
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("No specific indicators" in call for call in markdown_calls)


def test_render_model_details_handles_missing_fields():
    """Test that render_model_details handles missing fields gracefully."""
    from streamlit_app import render_model_details
    
    # Empty result dictionary
    result = {}
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        # Should not raise an exception
        render_model_details(result)
        
        # Verify function completed
        assert mock_st.expander.called


def test_render_model_details_pattern_score_low():
    """Test pattern score interpretation for low values."""
    from streamlit_app import render_model_details
    
    result = {
        "model_prediction": 1,
        "confidence": 95,
        "pattern_score": 0.15,
        "key_indicators": ["Test"]
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_model_details(result)
        
        # Check for low pattern score interpretation
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Low suspicious pattern" in call or "few red flags" in call for call in markdown_calls)


def test_render_model_details_pattern_score_moderate():
    """Test pattern score interpretation for moderate values."""
    from streamlit_app import render_model_details
    
    result = {
        "model_prediction": 0,
        "confidence": 70,
        "pattern_score": 0.45,
        "key_indicators": ["Test"]
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock the expander context manager
        mock_expander = MagicMock()
        mock_st.expander.return_value.__enter__ = MagicMock(return_value=mock_expander)
        mock_st.expander.return_value.__exit__ = MagicMock(return_value=False)
        
        render_model_details(result)
        
        # Check for moderate pattern score interpretation
        markdown_calls = [str(call) for call in mock_st.markdown.call_args_list]
        assert any("Moderate suspicious pattern" in call or "some concerning" in call for call in markdown_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
