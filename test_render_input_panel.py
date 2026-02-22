"""
Test for streamlit_app.py render_input_panel() function (Task 5.1)

This test verifies that the render_input_panel function is properly defined
and can be imported. Full UI testing would require Streamlit testing framework.
"""

import pytest
from unittest.mock import patch, MagicMock


class MockSessionState(dict):
    """Mock class for Streamlit session state that supports both dict and attribute access."""
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")


def test_render_input_panel_exists():
    """Test that render_input_panel function exists and is callable."""
    from streamlit_app import render_input_panel
    
    assert callable(render_input_panel)


def test_render_input_panel_returns_string():
    """Test that render_input_panel returns a string."""
    from streamlit_app import render_input_panel
    
    # Mock all Streamlit components
    with patch('streamlit_app.st') as mock_st:
        # Mock session state as a proper dict-like object
        mock_st.session_state = MockSessionState()
        
        # Mock text_area to return a test string
        mock_st.text_area.return_value = "Test article text"
        mock_st.selectbox.return_value = "-- Select an example --"
        mock_st.button.return_value = False
        
        result = render_input_panel()
        
        assert isinstance(result, str)
        assert result == "Test article text"


def test_render_input_panel_calls_streamlit_components():
    """Test that render_input_panel calls expected Streamlit components."""
    from streamlit_app import render_input_panel
    
    with patch('streamlit_app.st') as mock_st:
        # Mock session state
        mock_st.session_state = MockSessionState()
        mock_st.text_area.return_value = ""
        mock_st.selectbox.return_value = "-- Select an example --"
        mock_st.button.return_value = False
        
        render_input_panel()
        
        # Verify key components are called
        mock_st.header.assert_called()
        mock_st.text_area.assert_called()
        mock_st.selectbox.assert_called()
        mock_st.button.assert_called()
        mock_st.caption.assert_called()


def test_render_input_panel_loads_example_article():
    """Test that render_input_panel loads example article when selected."""
    from streamlit_app import render_input_panel, EXAMPLE_ARTICLES
    
    with patch('streamlit_app.st') as mock_st:
        # Mock session state
        mock_st.session_state = MockSessionState()
        
        # Simulate selecting an example article
        example_name = "Example Credible Article"
        mock_st.selectbox.return_value = example_name
        mock_st.text_area.return_value = EXAMPLE_ARTICLES[example_name]
        mock_st.button.return_value = False
        
        result = render_input_panel()
        
        # Verify the example article text is returned
        assert result == EXAMPLE_ARTICLES[example_name]
        assert len(result) > 0


def test_render_input_panel_character_count_display():
    """Test that render_input_panel displays character count correctly."""
    from streamlit_app import render_input_panel, UIConfig
    
    with patch('streamlit_app.st') as mock_st:
        # Mock session state
        mock_st.session_state = MockSessionState()
        
        # Test with short text (below minimum)
        short_text = "Short"
        mock_st.text_area.return_value = short_text
        mock_st.selectbox.return_value = "-- Select an example --"
        mock_st.button.return_value = False
        
        render_input_panel()
        
        # Verify caption is called (character count display)
        assert mock_st.caption.called
        
        # Get the caption call arguments
        caption_calls = [call[0][0] for call in mock_st.caption.call_args_list]
        
        # Verify character count is displayed
        assert any(str(len(short_text)) in str(call) for call in caption_calls)


def test_render_input_panel_button_disabled_for_invalid_input():
    """Test that analyze button is disabled for invalid input lengths."""
    from streamlit_app import render_input_panel, UIConfig
    
    with patch('streamlit_app.st') as mock_st:
        # Mock session state
        mock_st.session_state = MockSessionState()
        
        # Test with text below minimum length
        short_text = "x" * (UIConfig.MIN_TEXT_LENGTH - 1)
        mock_st.text_area.return_value = short_text
        mock_st.selectbox.return_value = "-- Select an example --"
        mock_st.button.return_value = False
        
        render_input_panel()
        
        # Verify button is called with disabled=True
        button_call = mock_st.button.call_args
        assert button_call is not None
        assert button_call[1].get('disabled') == True


def test_render_input_panel_button_enabled_for_valid_input():
    """Test that analyze button is enabled for valid input lengths."""
    from streamlit_app import render_input_panel, UIConfig
    
    with patch('streamlit_app.st') as mock_st:
        # Mock session state
        mock_st.session_state = MockSessionState()
        
        # Test with valid text length
        valid_text = "x" * (UIConfig.MIN_TEXT_LENGTH + 10)
        mock_st.text_area.return_value = valid_text
        mock_st.selectbox.return_value = "-- Select an example --"
        mock_st.button.return_value = False
        
        render_input_panel()
        
        # Verify button is called with disabled=False
        button_call = mock_st.button.call_args
        assert button_call is not None
        assert button_call[1].get('disabled') == False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
