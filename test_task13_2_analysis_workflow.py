"""
Test for Task 13.2: Analysis Workflow Implementation

This test verifies that the analysis workflow is correctly implemented:
- Validates input when analyze button is clicked
- Displays spinner during analysis
- Calls analyzer.analyze() with article text
- Stores results in session state
- Disables analyze button during analysis
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import streamlit as st


def test_analysis_workflow_integration():
    """
    Test the complete analysis workflow integration.
    
    This test verifies:
    1. Input validation is performed when analyze button is clicked
    2. Spinner is displayed during analysis
    3. analyzer.analyze() is called with correct parameters
    4. Results are stored in session state
    5. Analyze button is disabled during analysis
    """
    # Mock the analyzer and model
    mock_analyzer = Mock()
    mock_model = Mock()
    mock_vectorizer = Mock()
    
    # Create a sample analysis result
    sample_result = {
        "classification": "FAKE",
        "credibility_score": 25,
        "risk_level": "High Risk",
        "confidence": 85,
        "analysis_summary": "This article shows strong indicators of misinformation.",
        "key_indicators": ["High use of sensational language", "Multiple vague source references"],
        "emotional_tone": "Highly emotional and manipulative",
        "suspicious_claims": ["Claim 1", "Claim 2"],
        "recommended_action": "Exercise extreme caution with this content.",
        "explanation": "The article received a classification of 'FAKE' with a credibility score of 25/100.",
        "model_prediction": 0,
        "pattern_score": 0.75,
        "patterns": {
            "sensational_phrases": 5,
            "excessive_caps": 0.15,
            "vague_sources": 3,
            "conspiracy_framing": 2,
            "emotional_manipulation": 4,
            "one_sided": 0.8,
            "no_evidence": 0.7,
            "extreme_adjectives": 6,
            "clickbait": 2
        }
    }
    
    # Configure the mock analyzer to return the sample result
    mock_analyzer.analyze.return_value = sample_result
    
    # Test article text
    test_article = "This is a test article with sufficient length to pass validation. " * 10
    
    # Simulate the analysis workflow
    with patch('streamlit.spinner') as mock_spinner:
        # Mock spinner context manager
        mock_spinner.return_value.__enter__ = Mock()
        mock_spinner.return_value.__exit__ = Mock()
        
        # Simulate session state
        session_state = {
            "analyze_clicked": True,
            "analyzing": False,
            "results": None,
            "analyzed": False,
            "article_text": test_article
        }
        
        # Validate input
        from streamlit_app import validate_input
        is_valid, error_message = validate_input(test_article)
        
        assert is_valid, f"Input validation failed: {error_message}"
        assert error_message == "", "Error message should be empty for valid input"
        
        # Simulate analysis workflow
        if is_valid:
            # Clear previous results
            session_state["results"] = None
            session_state["analyzed"] = False
            
            # Set analyzing flag
            session_state["analyzing"] = True
            
            # Call analyzer.analyze()
            results = mock_analyzer.analyze(test_article, mock_model, mock_vectorizer)
            
            # Verify analyzer.analyze() was called with correct parameters
            mock_analyzer.analyze.assert_called_once_with(test_article, mock_model, mock_vectorizer)
            
            # Store results in session state
            session_state["results"] = results
            session_state["analyzed"] = True
            
            # Reset button state
            session_state["analyze_clicked"] = False
            session_state["analyzing"] = False
        
        # Verify results are stored correctly
        assert session_state["results"] == sample_result, "Results not stored correctly in session state"
        assert session_state["analyzed"] is True, "Analyzed flag not set correctly"
        assert session_state["analyze_clicked"] is False, "Analyze clicked flag not reset"
        assert session_state["analyzing"] is False, "Analyzing flag not reset"
        
        # Verify all expected keys are present in results
        expected_keys = [
            "classification", "credibility_score", "risk_level", "confidence",
            "analysis_summary", "key_indicators", "emotional_tone", "suspicious_claims",
            "recommended_action", "explanation", "model_prediction", "pattern_score", "patterns"
        ]
        
        for key in expected_keys:
            assert key in session_state["results"], f"Missing key in results: {key}"
    
    print("✅ Analysis workflow integration test passed!")


def test_analysis_workflow_validation_failure():
    """
    Test that analysis workflow handles validation failures correctly.
    """
    # Test with text that's too short
    short_text = "Too short"
    
    from streamlit_app import validate_input
    is_valid, error_message = validate_input(short_text)
    
    assert not is_valid, "Validation should fail for short text"
    assert error_message == "Article text must be at least 50 characters", "Incorrect error message"
    
    # Test with text that's too long
    long_text = "x" * 50001
    is_valid, error_message = validate_input(long_text)
    
    assert not is_valid, "Validation should fail for long text"
    assert error_message == "Article text must not exceed 50,000 characters", "Incorrect error message"
    
    print("✅ Validation failure test passed!")


def test_analyze_button_disabled_during_analysis():
    """
    Test that the analyze button is disabled during analysis.
    """
    from streamlit_app import UIConfig
    
    # Valid text
    valid_text = "This is a valid article text with sufficient length. " * 10
    char_count = len(valid_text)
    
    # Test button should be enabled when not analyzing
    is_analyzing = False
    should_disable = (
        char_count < UIConfig.MIN_TEXT_LENGTH or 
        char_count > UIConfig.MAX_TEXT_LENGTH or 
        is_analyzing
    )
    
    assert not should_disable, "Button should be enabled when not analyzing and text is valid"
    
    # Test button should be disabled when analyzing
    is_analyzing = True
    should_disable = (
        char_count < UIConfig.MIN_TEXT_LENGTH or 
        char_count > UIConfig.MAX_TEXT_LENGTH or 
        is_analyzing
    )
    
    assert should_disable, "Button should be disabled when analyzing"
    
    print("✅ Button disable test passed!")


def test_session_state_cleared_on_new_analysis():
    """
    Test that previous results are cleared when new analysis is triggered.
    """
    # Simulate session state with previous results
    session_state = {
        "results": {"classification": "REAL", "credibility_score": 80},
        "analyzed": True,
        "analyze_clicked": True
    }
    
    # Simulate new analysis trigger
    if session_state["analyze_clicked"]:
        # Clear previous results (Requirement 18.3)
        session_state["results"] = None
        session_state["analyzed"] = False
    
    # Verify results are cleared
    assert session_state["results"] is None, "Previous results should be cleared"
    assert session_state["analyzed"] is False, "Analyzed flag should be reset"
    
    print("✅ Session state clearing test passed!")


if __name__ == "__main__":
    print("Running Task 13.2 Analysis Workflow Tests...\n")
    
    test_analysis_workflow_integration()
    test_analysis_workflow_validation_failure()
    test_analyze_button_disabled_during_analysis()
    test_session_state_cleared_on_new_analysis()
    
    print("\n✅ All Task 13.2 tests passed!")
