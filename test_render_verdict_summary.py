"""
Test for streamlit_app.py render_verdict_summary() function (Task 6.1)
"""

import pytest
from unittest.mock import patch, MagicMock, call


def test_render_verdict_summary_with_real_classification():
    """Test that render_verdict_summary displays REAL classification correctly."""
    from streamlit_app import render_verdict_summary
    
    # Mock result dictionary
    result = {
        "classification": "REAL",
        "credibility_score": 85,
        "risk_level": "Low Risk",
        "confidence": 92,
        "analysis_summary": "This article appears credible with strong evidence."
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return three mock column objects
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Verify subheader was called
        mock_st.subheader.assert_called_once_with("🎯 Verdict Summary")
        
        # Verify markdown was called (for classification display and metrics)
        assert mock_st.markdown.call_count >= 4  # Classification + 3 metrics + spacing
        
        # Verify info was called for analysis summary
        mock_st.info.assert_called_once_with("This article appears credible with strong evidence.")


def test_render_verdict_summary_with_fake_classification():
    """Test that render_verdict_summary displays FAKE classification correctly."""
    from streamlit_app import render_verdict_summary
    
    result = {
        "classification": "FAKE",
        "credibility_score": 15,
        "risk_level": "High Risk",
        "confidence": 88,
        "analysis_summary": "This article shows multiple indicators of misinformation."
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return three mock column objects
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Verify subheader was called
        mock_st.subheader.assert_called_once()
        
        # Verify info was called with correct summary
        mock_st.info.assert_called_once_with("This article shows multiple indicators of misinformation.")


def test_render_verdict_summary_with_misleading_classification():
    """Test that render_verdict_summary displays MISLEADING classification correctly."""
    from streamlit_app import render_verdict_summary
    
    result = {
        "classification": "MISLEADING",
        "credibility_score": 45,
        "risk_level": "Medium Risk",
        "confidence": 75,
        "analysis_summary": "This article contains some accurate information but also misleading claims."
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return three mock column objects
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Verify subheader was called
        mock_st.subheader.assert_called_once()
        
        # Verify markdown was called multiple times
        assert mock_st.markdown.call_count >= 4


def test_render_verdict_summary_with_unverified_classification():
    """Test that render_verdict_summary displays UNVERIFIED classification correctly."""
    from streamlit_app import render_verdict_summary
    
    result = {
        "classification": "UNVERIFIED",
        "credibility_score": 50,
        "risk_level": "Medium Risk",
        "confidence": 60,
        "analysis_summary": "Unable to determine credibility with confidence."
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return three mock column objects
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Verify subheader was called
        mock_st.subheader.assert_called_once()
        
        # Verify info was called
        mock_st.info.assert_called_once()


def test_render_verdict_summary_handles_missing_keys():
    """Test that render_verdict_summary handles missing keys gracefully with defaults."""
    from streamlit_app import render_verdict_summary
    
    # Empty result dictionary
    result = {}
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return three mock column objects
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Should not raise an error and should use defaults
        mock_st.subheader.assert_called_once()
        mock_st.info.assert_called_once_with("")  # Empty summary


def test_render_verdict_summary_displays_all_metrics():
    """Test that render_verdict_summary displays all required metrics."""
    from streamlit_app import render_verdict_summary
    
    result = {
        "classification": "REAL",
        "credibility_score": 90,
        "risk_level": "Low Risk",
        "confidence": 95,
        "analysis_summary": "Test summary"
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return mock column objects
        mock_col1 = MagicMock()
        mock_col2 = MagicMock()
        mock_col3 = MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Verify columns were created
        mock_st.columns.assert_called_once_with(3)
        
        # Verify markdown was called within each column context
        # (The actual calls happen within the 'with' context managers)
        assert mock_st.markdown.call_count >= 4


def test_render_verdict_summary_color_coding():
    """Test that render_verdict_summary uses correct color coding."""
    from streamlit_app import render_verdict_summary, UIConfig
    
    result = {
        "classification": "FAKE",
        "credibility_score": 20,
        "risk_level": "High Risk",
        "confidence": 85,
        "analysis_summary": "Test"
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return three mock column objects
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Check that markdown was called with HTML containing color codes
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list if call[0]]
        
        # At least one markdown call should contain the FAKE color
        assert any(UIConfig.COLOR_FAKE in str(call) for call in markdown_calls)
        
        # At least one markdown call should contain the High Risk color
        assert any(UIConfig.COLOR_HIGH_RISK in str(call) for call in markdown_calls)


def test_render_verdict_summary_displays_credibility_score_with_suffix():
    """Test that credibility score is displayed with /100 suffix."""
    from streamlit_app import render_verdict_summary
    
    result = {
        "classification": "REAL",
        "credibility_score": 78,
        "risk_level": "Low Risk",
        "confidence": 82,
        "analysis_summary": "Test"
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return three mock column objects
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Check that markdown was called with HTML containing the score and /100
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list if call[0]]
        
        # At least one markdown call should contain "78" and "/100"
        assert any("78" in str(call) and "/100" in str(call) for call in markdown_calls)


def test_render_verdict_summary_displays_confidence_as_percentage():
    """Test that confidence is displayed as a percentage."""
    from streamlit_app import render_verdict_summary
    
    result = {
        "classification": "REAL",
        "credibility_score": 80,
        "risk_level": "Low Risk",
        "confidence": 91,
        "analysis_summary": "Test"
    }
    
    with patch('streamlit_app.st') as mock_st:
        # Mock columns to return three mock column objects
        mock_col1, mock_col2, mock_col3 = MagicMock(), MagicMock(), MagicMock()
        mock_st.columns.return_value = [mock_col1, mock_col2, mock_col3]
        
        render_verdict_summary(result)
        
        # Check that markdown was called with HTML containing the confidence and %
        markdown_calls = [call[0][0] for call in mock_st.markdown.call_args_list if call[0]]
        
        # At least one markdown call should contain "91" and "%"
        assert any("91" in str(call) and "%" in str(call) for call in markdown_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
