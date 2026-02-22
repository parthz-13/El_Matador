"""
Test for streamlit_app.py validate_input() function (Task 3.1)

Tests the input validation logic for article text, ensuring proper handling
of minimum and maximum length constraints.
"""

import pytest
from streamlit_app import validate_input, UIConfig


class TestValidateInput:
    """Test suite for validate_input() function."""
    
    def test_validate_input_too_short(self):
        """Test that text shorter than 50 characters is rejected."""
        # Test with empty string
        is_valid, error_msg = validate_input("")
        assert is_valid is False
        assert error_msg == "Article text must be at least 50 characters"
        
        # Test with 49 characters (just below minimum)
        short_text = "a" * 49
        is_valid, error_msg = validate_input(short_text)
        assert is_valid is False
        assert error_msg == "Article text must be at least 50 characters"
    
    def test_validate_input_minimum_length(self):
        """Test that text with exactly 50 characters is accepted."""
        # Test with exactly 50 characters (minimum valid length)
        min_text = "a" * 50
        is_valid, error_msg = validate_input(min_text)
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_input_valid_length(self):
        """Test that text within valid range is accepted."""
        # Test with 100 characters (well within valid range)
        valid_text = "a" * 100
        is_valid, error_msg = validate_input(valid_text)
        assert is_valid is True
        assert error_msg == ""
        
        # Test with 1000 characters (mid-range)
        valid_text = "a" * 1000
        is_valid, error_msg = validate_input(valid_text)
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_input_maximum_length(self):
        """Test that text with exactly 50,000 characters is accepted."""
        # Test with exactly 50,000 characters (maximum valid length)
        max_text = "a" * 50000
        is_valid, error_msg = validate_input(max_text)
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_input_too_long(self):
        """Test that text longer than 50,000 characters is rejected."""
        # Test with 50,001 characters (just above maximum)
        long_text = "a" * 50001
        is_valid, error_msg = validate_input(long_text)
        assert is_valid is False
        assert error_msg == "Article text must not exceed 50,000 characters"
        
        # Test with 100,000 characters (well above maximum)
        very_long_text = "a" * 100000
        is_valid, error_msg = validate_input(very_long_text)
        assert is_valid is False
        assert error_msg == "Article text must not exceed 50,000 characters"
    
    def test_validate_input_returns_tuple(self):
        """Test that validate_input always returns a tuple of (bool, str)."""
        # Test with valid input
        result = validate_input("a" * 100)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
        
        # Test with invalid input
        result = validate_input("short")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)
    
    def test_validate_input_uses_config_constants(self):
        """Test that validate_input uses UIConfig constants for validation."""
        # Verify that the function respects UIConfig.MIN_TEXT_LENGTH
        min_boundary_text = "a" * (UIConfig.MIN_TEXT_LENGTH - 1)
        is_valid, _ = validate_input(min_boundary_text)
        assert is_valid is False
        
        min_valid_text = "a" * UIConfig.MIN_TEXT_LENGTH
        is_valid, _ = validate_input(min_valid_text)
        assert is_valid is True
        
        # Verify that the function respects UIConfig.MAX_TEXT_LENGTH
        max_valid_text = "a" * UIConfig.MAX_TEXT_LENGTH
        is_valid, _ = validate_input(max_valid_text)
        assert is_valid is True
        
        max_boundary_text = "a" * (UIConfig.MAX_TEXT_LENGTH + 1)
        is_valid, _ = validate_input(max_boundary_text)
        assert is_valid is False
    
    def test_validate_input_with_realistic_article(self):
        """Test validation with realistic article text."""
        # Realistic short article (valid)
        article = """
        Breaking news: Scientists have discovered a new species of butterfly
        in the Amazon rainforest. The discovery was made by researchers from
        the University of São Paulo during a biodiversity survey.
        """
        is_valid, error_msg = validate_input(article)
        assert is_valid is True
        assert error_msg == ""
        
        # Realistic very short text (invalid)
        snippet = "Breaking news today"
        is_valid, error_msg = validate_input(snippet)
        assert is_valid is False
        assert "at least 50 characters" in error_msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
