"""
Integration test to verify analyzer returns model_prediction, pattern_score, and patterns fields.
"""

import pytest
from unittest.mock import MagicMock


def test_analyzer_returns_new_fields():
    """Test that CredibilityAnalyzer.analyze() returns the new fields."""
    from credibility_analyzer import CredibilityAnalyzer
    
    # Create analyzer instance
    analyzer = CredibilityAnalyzer()
    
    # Create mock model and vectorizer
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.2, 0.8]]
    
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]
    
    # Test article text
    test_text = "This is a test article with sufficient length to pass validation. " * 5
    
    # Perform analysis
    result = analyzer.analyze(test_text, mock_model, mock_vectorizer)
    
    # Verify all expected fields are present
    assert "classification" in result
    assert "credibility_score" in result
    assert "risk_level" in result
    assert "confidence" in result
    assert "analysis_summary" in result
    assert "key_indicators" in result
    assert "emotional_tone" in result
    assert "suspicious_claims" in result
    assert "recommended_action" in result
    assert "explanation" in result
    
    # Verify new fields are present
    assert "model_prediction" in result, "model_prediction field is missing"
    assert "pattern_score" in result, "pattern_score field is missing"
    assert "patterns" in result, "patterns field is missing"
    
    # Verify field types
    assert isinstance(result["model_prediction"], int)
    assert result["model_prediction"] in [0, 1]
    
    assert isinstance(result["pattern_score"], float)
    assert 0.0 <= result["pattern_score"] <= 1.0
    
    assert isinstance(result["patterns"], dict)


def test_analyzer_pattern_score_calculation():
    """Test that pattern_score is calculated correctly."""
    from credibility_analyzer import CredibilityAnalyzer
    
    analyzer = CredibilityAnalyzer()
    
    # Create mock model and vectorizer
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_model.predict_proba.return_value = [[0.9, 0.1]]
    
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]
    
    # Test article with suspicious patterns
    test_text = """
    SHOCKING DISCOVERY!!! The government doesn't want you to know this!
    Sources say that experts claim this is the biggest cover-up in history.
    Wake up sheeple! This is ABSOLUTELY TERRIFYING and COMPLETELY unbelievable.
    According to insiders, the mainstream media is hiding the truth.
    """ * 3
    
    result = analyzer.analyze(test_text, mock_model, mock_vectorizer)
    
    # Pattern score should be higher for suspicious text
    assert result["pattern_score"] > 0.3, f"Expected pattern_score > 0.3, got {result['pattern_score']}"


def test_analyzer_patterns_dict_structure():
    """Test that patterns dictionary contains expected keys."""
    from credibility_analyzer import CredibilityAnalyzer
    
    analyzer = CredibilityAnalyzer()
    
    # Create mock model and vectorizer
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]
    mock_model.predict_proba.return_value = [[0.3, 0.7]]
    
    mock_vectorizer = MagicMock()
    mock_vectorizer.transform.return_value = [[0.1, 0.2, 0.3]]
    
    test_text = "This is a normal article with proper sourcing and evidence. " * 10
    
    result = analyzer.analyze(test_text, mock_model, mock_vectorizer)
    
    # Verify patterns dictionary has expected keys
    patterns = result["patterns"]
    expected_keys = [
        "sensational_phrases",
        "excessive_caps",
        "vague_sources",
        "conspiracy_framing",
        "emotional_manipulation",
        "one_sided",
        "no_evidence",
        "extreme_adjectives",
        "clickbait"
    ]
    
    for key in expected_keys:
        assert key in patterns, f"Expected key '{key}' not found in patterns dictionary"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
