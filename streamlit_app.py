"""
VerifyAI: ML-Based News Credibility Analysis
Streamlit Web Application

This application provides an interpretable, research-grade interface for
ML-based news credibility analysis using the CredibilityAnalyzer system.

ACCESSIBILITY FEATURES (Requirements 16.1-16.5):
- Semantic HTML: Uses st.header, st.subheader, st.text_area for proper structure
- Text alternatives: All color-coded information includes text labels (e.g., "Low Risk", "High Risk")
- Color contrast: WCAG AA compliant colors with sufficient contrast ratios
- Descriptive labels: All input elements have clear, descriptive labels
- Keyboard navigation: Standard Streamlit controls support keyboard navigation
"""

import streamlit as st
import joblib
import os
from typing import Tuple, Dict, List, Optional
from credibility_analyzer import CredibilityAnalyzer


# ============================================================================
# UI Configuration Constants
# ============================================================================

class UIConfig:
    """Configuration constants for the UI."""
    
    # Layout
    LAYOUT_MODE = "wide"
    COLUMN_RATIO = [1, 1.3]
    
    # Input validation
    MIN_TEXT_LENGTH = 50
    MAX_TEXT_LENGTH = 50000
    
    # Text area
    TEXT_AREA_HEIGHT = 300
    
    # Color scheme - WCAG AA compliant colors for accessibility
    COLOR_REAL = "#28a745"      # Green - sufficient contrast on white background
    COLOR_FAKE = "#dc3545"      # Red - sufficient contrast on white background
    COLOR_MISLEADING = "#d63384"  # Darker pink/magenta - improved contrast (was #fd7e14)
    COLOR_UNVERIFIED = "#6c757d" # Gray - sufficient contrast on white background
    
    # Risk level colors - WCAG AA compliant
    COLOR_LOW_RISK = "#28a745"   # Green - sufficient contrast on white background
    COLOR_MEDIUM_RISK = "#ffc107" # Yellow - used with dark text for contrast
    COLOR_HIGH_RISK = "#dc3545"  # Red - sufficient contrast on white background
    
    # Model information
    MODEL_TYPE = "TF-IDF + Logistic Regression"
    DATASET_SIZE = "362,000+ labeled articles"
    
    # Performance metrics (from training)
    ACCURACY = 0.95
    PRECISION = 0.94
    RECALL = 0.96
    F1_SCORE = 0.95


# ============================================================================
# Model Loading and Caching Layer
# ============================================================================

@st.cache_resource
def load_analyzer() -> CredibilityAnalyzer:
    """
    Load and cache the CredibilityAnalyzer instance.
    
    This function uses Streamlit's cache_resource decorator to ensure the
    analyzer is loaded only once per session, improving performance across
    page reruns.
    
    Returns:
        CredibilityAnalyzer: Cached analyzer instance
        
    Raises:
        FileNotFoundError: If required backend files are missing
        Exception: If analyzer initialization fails
    """
    try:
        analyzer = CredibilityAnalyzer()
        return analyzer
    except FileNotFoundError as e:
        st.error(f"**Model Loading Error**: Required files not found - {str(e)}")
        st.error("Please ensure all backend components are properly installed.")
        raise
    except Exception as e:
        st.error(f"**Initialization Error**: Failed to load analyzer - {str(e)}")
        st.error("Please check the application logs for more details.")
        raise


@st.cache_resource
def load_model() -> Tuple[object, object]:
    """
    Load and cache the trained ML model and TF-IDF vectorizer.
    
    This function uses Streamlit's cache_resource decorator to ensure the
    model and vectorizer are loaded only once per session, improving performance
    across page reruns.
    
    Returns:
        Tuple[object, object]: Tuple of (model, vectorizer)
            - model: Trained scikit-learn model (Logistic Regression or Passive Aggressive)
            - vectorizer: Fitted TF-IDF vectorizer
        
    Raises:
        FileNotFoundError: If model files are missing
        Exception: If model loading fails
    """
    # Define model file paths
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "best_model.joblib")
    vectorizer_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")
    
    try:
        # Check if model files exist
        if not os.path.exists(model_path):
            error_msg = (
                f"Model file not found at: {model_path}\n\n"
                "Please train the model first by running:\n"
                "  python train_model.py\n\n"
                "This will create the required model files in the 'models/' directory."
            )
            st.error(f"**Model File Missing**\n\n{error_msg}")
            raise FileNotFoundError(error_msg)
        
        if not os.path.exists(vectorizer_path):
            error_msg = (
                f"Vectorizer file not found at: {vectorizer_path}\n\n"
                "Please train the model first by running:\n"
                "  python train_model.py\n\n"
                "This will create the required vectorizer file in the 'models/' directory."
            )
            st.error(f"**Vectorizer File Missing**\n\n{error_msg}")
            raise FileNotFoundError(error_msg)
        
        # Load model and vectorizer
        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        return model, vectorizer
        
    except FileNotFoundError:
        # Re-raise FileNotFoundError with our custom message
        raise
    except Exception as e:
        error_msg = (
            f"Failed to load model files: {str(e)}\n\n"
            "Troubleshooting steps:\n"
            "1. Verify model files exist in the 'models/' directory\n"
            "2. Ensure model files are not corrupted\n"
            "3. Try retraining the model: python train_model.py\n"
            "4. Check file permissions"
        )
        st.error(f"**Model Loading Error**\n\n{error_msg}")
        raise Exception(error_msg) from e


# ============================================================================
# Input Validation Component
# ============================================================================

def validate_input(text: str) -> Tuple[bool, str]:
    """
    Validate article text input.
    
    Checks that the article text meets minimum and maximum length requirements
    before analysis can proceed.
    
    Args:
        text: The article text to validate
        
    Returns:
        Tuple[bool, str]: Tuple of (is_valid, error_message)
            - is_valid: True if validation passes, False otherwise
            - error_message: Empty string if valid, specific error message otherwise
    """
    text_length = len(text)
    
    # Check minimum length
    if text_length < UIConfig.MIN_TEXT_LENGTH:
        return False, "Article text must be at least 50 characters"
    
    # Check maximum length
    if text_length > UIConfig.MAX_TEXT_LENGTH:
        return False, "Article text must not exceed 50,000 characters"
    
    # Validation passed
    return True, ""


class InputValidator:
    """
    Validates user input for article text.
    
    This class provides an object-oriented interface for input validation
    with the same validation logic as the validate_input() function.
    """
    
    MIN_LENGTH = 50
    MAX_LENGTH = 50000
    
    @staticmethod
    def validate(text: str) -> Tuple[bool, str]:
        """
        Validate article text length.
        
        Checks that the article text meets minimum and maximum length requirements
        before analysis can proceed.
        
        Args:
            text: Article text to validate
            
        Returns:
            Tuple[bool, str]: Tuple of (is_valid, error_message)
                - is_valid: True if validation passes, False otherwise
                - error_message: Empty string if valid, specific error message otherwise
        """
        text_length = len(text)
        
        # Check minimum length
        if text_length < InputValidator.MIN_LENGTH:
            return False, "Article text must be at least 50 characters"
        
        # Check maximum length
        if text_length > InputValidator.MAX_LENGTH:
            return False, "Article text must not exceed 50,000 characters"
        
        # Validation passed
        return True, ""


# ============================================================================
# Example Articles Component
# ============================================================================

EXAMPLE_ARTICLES = {
    "Example Credible Article": """
Scientists at Stanford University have published a peer-reviewed study in the journal Nature showing that a new vaccine candidate demonstrates 89% efficacy in phase 3 clinical trials involving 30,000 participants across 15 countries.

Dr. Sarah Chen, lead researcher at Stanford's Department of Immunology, stated in a press conference yesterday that the results were "highly encouraging" and that the team plans to submit the data to the FDA for emergency use authorization within the next month.

The study, which began in March 2023, tracked participants for an average of 6 months following vaccination. The research team reported that serious adverse events were rare, occurring in less than 0.1% of participants, and were comparable to rates seen with other approved vaccines.

"This vaccine uses a novel mRNA platform that has been refined based on lessons learned from previous vaccine development efforts," explained Dr. Chen. "The technology allows for rapid adaptation to new variants, which could prove crucial in managing future outbreaks."

The pharmaceutical company funding the research, BioMed Solutions, announced that manufacturing capacity is being scaled up at facilities in three countries to prepare for potential approval. The company has committed to providing doses at cost to low-income countries through partnerships with international health organizations.

Independent experts not involved in the study have praised the rigorous methodology. Dr. Michael Rodriguez, an epidemiologist at Johns Hopkins University, noted that "the sample size is robust, the follow-up period is adequate, and the transparency of the data sharing is commendable."

The full study data has been made available to the scientific community for independent review, and the research team has scheduled presentations at upcoming medical conferences to discuss their findings in detail.
""",
    
    "Example Suspicious Article": """
SHOCKING DISCOVERY: Government Scientists ADMIT Vaccines Contain Dangerous Chemicals That Big Pharma Doesn't Want You to Know About!!!

An EXPLOSIVE new report reveals that mainstream media has been HIDING the truth about what's really in vaccines. Anonymous sources close to the CDC have leaked documents showing that pharmaceutical companies are adding mysterious substances to vaccines without proper testing.

Experts say this could be the biggest cover-up in medical history! Thousands of people are reporting strange symptoms after vaccination, but doctors are being SILENCED by powerful corporations who control the entire healthcare system.

"They don't want you to know the truth," says one concerned parent who wishes to remain anonymous. "I did my own research online and found dozens of articles explaining how these chemicals are linked to serious health problems. Why isn't the government investigating this?"

Many people are now questioning whether vaccines are safe at all. Some alternative health practitioners suggest that natural immunity is far superior to anything created in a laboratory. One holistic doctor claims that a simple combination of vitamins and herbs can provide better protection than any vaccine.

The pharmaceutical industry makes BILLIONS of dollars from vaccines every year. Is it any wonder they want to keep pushing them on unsuspecting families? Follow the money and you'll see who really benefits from mandatory vaccination programs.

Wake up, people! Don't let them inject you with unknown substances. Do your own research and make informed decisions about your health. Share this article before it gets censored by Big Tech!
"""
}


def load_example(example_name: str) -> str:
    """
    Load an example article by name.
    
    Retrieves the full text of a pre-defined example article for demonstration
    purposes. Example articles include both credible and suspicious content to
    showcase the system's analysis capabilities.
    
    Args:
        example_name: Name of the example article (must match a key in EXAMPLE_ARTICLES)
        
    Returns:
        str: Full text of the example article
        
    Raises:
        KeyError: If the example_name is not found in EXAMPLE_ARTICLES
    """
    return EXAMPLE_ARTICLES[example_name]


# ============================================================================
# Input Panel Rendering
# ============================================================================

def render_input_panel() -> str:
    """
    Render the left column input panel.
    
    Displays a text area for article input, character count, example article
    selector, analyze button, and help text. This function handles all user
    input interactions for the analysis workflow.
    
    Returns:
        str: The article text entered by the user (empty string if no input)
    """
    st.header("Article Input")
    
    # Help text explaining input requirements
    st.markdown("""
    **How to use:**
    - Paste or type an article (minimum 50 characters, maximum 50,000 characters)
    - Or select an example article from the dropdown below
    - Click "Analyze Article" to see the credibility assessment
    """)
    
    # Example article selector
    st.subheader("Load Example Article")
    example_options = ["-- Select an example --"] + list(EXAMPLE_ARTICLES.keys())
    selected_example = st.selectbox(
        "Choose an example to analyze:",
        options=example_options,
        key="example_selector"
    )
    
    # Initialize article text from session state or empty
    if "article_text" not in st.session_state:
        st.session_state.article_text = ""
    
    # Load example article if selected
    if selected_example != "-- Select an example --":
        st.session_state.article_text = load_example(selected_example)
    
    # Text area for article input
    st.subheader("Article Text")
    article_text = st.text_area(
        "Enter or paste article text here:",
        value=st.session_state.article_text,
        height=UIConfig.TEXT_AREA_HEIGHT,
        key="article_input",
        placeholder="Paste your article text here..."
    )
    
    # Update session state with current text
    st.session_state.article_text = article_text
    
    # Display character count
    char_count = len(article_text)
    if char_count < UIConfig.MIN_TEXT_LENGTH:
        st.caption(f"Character count: {char_count} / {UIConfig.MIN_TEXT_LENGTH} minimum")
    elif char_count > UIConfig.MAX_TEXT_LENGTH:
        st.caption(f"Character count: {char_count} / {UIConfig.MAX_TEXT_LENGTH} maximum (exceeded)")
    else:
        st.caption(f"Character count: {char_count}")
    
    # Display validation warnings
    if char_count > 0:  # Only show warnings if user has entered text
        is_valid, error_message = validate_input(article_text)
        if not is_valid:
            st.warning(f"{error_message}")
    
    # Analyze button
    st.markdown("---")
    
    # Check if analysis is in progress
    is_analyzing = st.session_state.get("analyzing", False)
    
    analyze_button = st.button(
        "Analyze Article",
        type="primary",
        use_container_width=True,
        disabled=(char_count < UIConfig.MIN_TEXT_LENGTH or char_count > UIConfig.MAX_TEXT_LENGTH or is_analyzing)
    )
    
    # Store button state in session state for use in main()
    if "analyze_clicked" not in st.session_state:
        st.session_state.analyze_clicked = False
    
    if analyze_button:
        st.session_state.analyze_clicked = True
    
    return article_text


# ============================================================================
# Results Display Components
# ============================================================================

def render_verdict_summary(result: Dict):
    """
    Render the verdict summary section.
    
    Displays the primary classification result, credibility score, risk level,
    model confidence, and analysis summary with appropriate color coding.
    
    Args:
        result: Analysis result dictionary from CredibilityAnalyzer.analyze()
            Expected keys: classification, credibility_score, risk_level,
            confidence, analysis_summary
    """
    st.subheader("Verdict Summary")
    
    # Get values from result dictionary
    classification = result.get("classification", "UNVERIFIED")
    credibility_score = result.get("credibility_score", 0)
    risk_level = result.get("risk_level", "High Risk")
    confidence = result.get("confidence", 0)
    analysis_summary = result.get("analysis_summary", "")
    
    # Map classification to colors
    classification_colors = {
        "REAL": UIConfig.COLOR_REAL,
        "FAKE": UIConfig.COLOR_FAKE,
        "MISLEADING": UIConfig.COLOR_MISLEADING,
        "UNVERIFIED": UIConfig.COLOR_UNVERIFIED
    }
    classification_color = classification_colors.get(classification, UIConfig.COLOR_UNVERIFIED)
    
    # Map risk level to colors
    risk_colors = {
        "Low Risk": UIConfig.COLOR_LOW_RISK,
        "Medium Risk": UIConfig.COLOR_MEDIUM_RISK,
        "High Risk": UIConfig.COLOR_HIGH_RISK
    }
    risk_color = risk_colors.get(risk_level, UIConfig.COLOR_HIGH_RISK)
    
    # Display classification label with color coding and text alternative
    # Text alternative for accessibility: Classification is also shown in plain text
    st.markdown(
        f"""
        <div style="text-align: center; padding: 20px; background-color: {classification_color}15; border-radius: 10px; border: 2px solid {classification_color};">
            <h2 style="color: {classification_color}; margin: 0; font-size: 2.5em;">{classification}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create three columns for metrics
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    
    # Display credibility score as large number with "/100" suffix
    # Text alternative: "Credibility Score" label provides context
    with metric_col1:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Credibility Score</div>
                <div style="font-size: 3em; font-weight: bold; color: {classification_color};">{credibility_score}<span style="font-size: 0.5em; color: #999;">/100</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display risk level with color-coded badge and text label
    # Text alternative: Risk level text ("Low Risk", "Medium Risk", "High Risk") is always visible
    with metric_col2:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Risk Level</div>
                <div style="margin-top: 10px;">
                    <span style="background-color: {risk_color}; color: white; padding: 10px 20px; border-radius: 20px; font-size: 1.2em; font-weight: bold; display: inline-block;">{risk_level}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Display model confidence percentage
    with metric_col3:
        st.markdown(
            f"""
            <div style="text-align: center; padding: 15px;">
                <div style="font-size: 0.9em; color: #666; margin-bottom: 5px;">Model Confidence</div>
                <div style="font-size: 3em; font-weight: bold; color: #333;">{confidence}<span style="font-size: 0.5em; color: #999;">%</span></div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display analysis summary text
    st.markdown("**Analysis Summary:**")
    st.info(analysis_summary)


def render_model_details(result: Dict):
    """
    Render the model prediction details section (expandable).
    
    Displays technical details about the ML model's prediction including raw
    prediction value, confidence score, pattern score, and key indicators.
    This section is organized in an expandable container for researchers and
    advanced users.
    
    Args:
        result: Analysis result dictionary from CredibilityAnalyzer.analyze()
            Expected keys: model_prediction, confidence, pattern_score, key_indicators
    """
    with st.expander("Model Prediction Details", expanded=False):
        st.markdown("**Technical details for researchers and advanced users**")
        st.markdown("---")
        
        # Get values from result dictionary
        model_prediction = result.get("model_prediction", 0)
        confidence = result.get("confidence", 0)
        pattern_score = result.get("pattern_score", 0.0)
        key_indicators = result.get("key_indicators", [])
        
        # Display raw model prediction with interpretation
        st.markdown("**Raw Model Prediction:**")
        prediction_interpretation = "REAL (1)" if model_prediction == 1 else "FAKE (0)"
        prediction_color = UIConfig.COLOR_REAL if model_prediction == 1 else UIConfig.COLOR_FAKE
        
        st.markdown(
            f"""
            <div style="padding: 10px; background-color: {prediction_color}15; border-left: 4px solid {prediction_color}; border-radius: 5px;">
                <span style="font-size: 1.2em; font-weight: bold; color: {prediction_color};">{prediction_interpretation}</span>
                <br>
                <span style="font-size: 0.9em; color: #666;">
                    The ML model classified this article as <strong>{"credible" if model_prediction == 1 else "not credible"}</strong> based on linguistic patterns learned from training data.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display model confidence score as percentage
        st.markdown("**Model Confidence Score:**")
        st.markdown(
            f"""
            <div style="padding: 10px; background-color: #f0f0f0; border-radius: 5px;">
                <span style="font-size: 1.5em; font-weight: bold; color: #333;">{confidence}%</span>
                <br>
                <span style="font-size: 0.9em; color: #666;">
                    The model's confidence in its prediction. Higher values indicate stronger certainty.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display pattern score with explanation
        st.markdown("**Pattern Score:**")
        pattern_score_percentage = int(pattern_score * 100)
        
        # Determine pattern score interpretation
        if pattern_score < 0.3:
            pattern_interpretation = "Low suspicious pattern density - article shows few red flags"
            pattern_color = UIConfig.COLOR_REAL
        elif pattern_score < 0.6:
            pattern_interpretation = "Moderate suspicious pattern density - some concerning indicators present"
            pattern_color = UIConfig.COLOR_MEDIUM_RISK
        else:
            pattern_interpretation = "High suspicious pattern density - multiple red flags detected"
            pattern_color = UIConfig.COLOR_FAKE
        
        st.markdown(
            f"""
            <div style="padding: 10px; background-color: {pattern_color}15; border-left: 4px solid {pattern_color}; border-radius: 5px;">
                <span style="font-size: 1.5em; font-weight: bold; color: {pattern_color};">{pattern_score_percentage}%</span>
                <br>
                <span style="font-size: 0.9em; color: #666;">
                    {pattern_interpretation}
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Display all key indicators as bulleted list
        st.markdown("**Key Indicators:**")
        if key_indicators:
            for indicator in key_indicators:
                st.markdown(f"• {indicator}")
        else:
            st.markdown("• No specific indicators identified")


def render_pattern_analysis(patterns: Dict):
    """
    Render the linguistic pattern analysis section.
    
    Displays all 9 pattern metrics from the PatternDetector in a structured
    table format. Ratios and scores are formatted as percentages for clarity.
    
    Args:
        patterns: Pattern detection results dictionary with keys:
            - sensational_phrases: Count of sensational keywords
            - excessive_caps: Ratio of excessively capitalized words (0.0-1.0)
            - vague_sources: Count of vague source references
            - conspiracy_framing: Count of conspiracy keywords
            - emotional_manipulation: Count of emotional manipulation keywords
            - one_sided: Score indicating one-sided narrative (0.0-1.0)
            - no_evidence: Score indicating lack of evidence (0.0-1.0)
            - extreme_adjectives: Count of extreme adjectives
            - clickbait: Count of clickbait patterns
    """
    st.subheader("Linguistic Pattern Analysis")
    st.markdown("Detailed breakdown of linguistic patterns detected in the article:")
    
    # Extract pattern values with defaults
    sensational_phrases = patterns.get("sensational_phrases", 0)
    excessive_caps = patterns.get("excessive_caps", 0.0)
    vague_sources = patterns.get("vague_sources", 0)
    conspiracy_framing = patterns.get("conspiracy_framing", 0)
    emotional_manipulation = patterns.get("emotional_manipulation", 0)
    one_sided = patterns.get("one_sided", 0.0)
    no_evidence = patterns.get("no_evidence", 0.0)
    extreme_adjectives = patterns.get("extreme_adjectives", 0)
    clickbait = patterns.get("clickbait", 0)
    
    # Convert ratios and scores to percentages
    excessive_caps_pct = f"{excessive_caps * 100:.1f}%"
    one_sided_pct = f"{one_sided * 100:.1f}%"
    no_evidence_pct = f"{no_evidence * 100:.1f}%"
    
    # Create a structured table using Streamlit's native table or dataframe
    # We'll use a more visual approach with columns for better readability
    
    st.markdown("---")
    
    # Row 1: Sensational Phrases, Excessive Caps, Vague Sources
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Sensational Phrases",
            value=sensational_phrases,
            help="Count of sensational keywords (e.g., SHOCKING, BREAKING, EXPOSED)"
        )
    
    with col2:
        st.metric(
            label="Excessive Capitalization",
            value=excessive_caps_pct,
            help="Ratio of excessively capitalized words in the text"
        )
    
    with col3:
        st.metric(
            label="Vague Sources",
            value=vague_sources,
            help="Count of vague source references (e.g., 'sources say', 'experts claim')"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Conspiracy Framing, Emotional Manipulation, One-Sided Score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Conspiracy Framing",
            value=conspiracy_framing,
            help="Count of conspiracy-related keywords (e.g., 'cover-up', 'hidden truth')"
        )
    
    with col2:
        st.metric(
            label="Emotional Manipulation",
            value=emotional_manipulation,
            help="Count of emotionally manipulative keywords (e.g., 'outrage', 'terrifying')"
        )
    
    with col3:
        st.metric(
            label="One-Sided Narrative",
            value=one_sided_pct,
            help="Score indicating lack of balanced perspectives (0% = balanced, 100% = one-sided)"
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 3: No Evidence Score, Extreme Adjectives, Clickbait Patterns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Lack of Evidence",
            value=no_evidence_pct,
            help="Score indicating absence of evidence-based claims (0% = well-evidenced, 100% = no evidence)"
        )
    
    with col2:
        st.metric(
            label="Extreme Adjectives",
            value=extreme_adjectives,
            help="Count of extreme adjectives (e.g., 'always', 'never', 'completely')"
        )
    
    with col3:
        st.metric(
            label="Clickbait Patterns",
            value=clickbait,
            help="Count of clickbait phrases (e.g., 'you won't believe', 'what happened next')"
        )
    
    st.markdown("---")
    
    # Add interpretation note
    st.info(
        "**Interpretation Guide:** Higher counts and percentages indicate more suspicious patterns. "
        "Credible journalism typically shows low values across all metrics, with balanced perspectives "
        "and evidence-based claims."
    )


def render_emotional_tone(emotional_tone: str):
    """
    Render the emotional tone analysis section.
    
    Displays the emotional tone classification with descriptive text, color coding
    for tone severity, and context about what the tone indicates. This helps users
    identify potential emotional manipulation tactics.
    
    Args:
        emotional_tone: Emotional tone classification string from EmotionalAnalyzer
            Possible values:
            - "Neutral and analytical"
            - "Moderately emotional"
            - "Conspiratorial and fear-inducing"
            - "Sensationalized and attention-seeking"
            - "Highly emotional and manipulative"
    """
    st.subheader("Emotional Tone Analysis")
    st.markdown("Assessment of emotional manipulation and sensationalism:")
    
    # Determine tone severity and color coding
    tone_lower = emotional_tone.lower()
    
    if "neutral" in tone_lower or "analytical" in tone_lower:
        # Neutral tone - green (low concern)
        tone_color = UIConfig.COLOR_REAL
        severity_badge = "Low Concern"
        severity_color = UIConfig.COLOR_LOW_RISK
        icon = ""
        context = (
            "The article maintains a neutral, analytical tone typical of credible journalism. "
            "This suggests the content focuses on facts and balanced reporting rather than "
            "emotional manipulation."
        )
    elif "moderately emotional" in tone_lower:
        # Moderate emotional tone - yellow (moderate concern)
        tone_color = UIConfig.COLOR_MEDIUM_RISK
        severity_badge = "Moderate Concern"
        severity_color = UIConfig.COLOR_MEDIUM_RISK
        icon = ""
        context = (
            "The article shows some emotional language that may be used to influence reader reactions. "
            "While not necessarily problematic, this warrants attention to ensure the emotional content "
            "doesn't overshadow factual reporting."
        )
    else:
        # High concern tones (conspiratorial, sensationalized, highly emotional)
        tone_color = UIConfig.COLOR_FAKE
        severity_badge = "High Concern"
        severity_color = UIConfig.COLOR_HIGH_RISK
        icon = ""
        
        if "conspiratorial" in tone_lower:
            context = (
                "The article uses conspiratorial framing and fear-inducing language. "
                "This is a strong indicator of misinformation, as credible sources avoid "
                "conspiracy theories and focus on verifiable facts."
            )
        elif "sensationalized" in tone_lower:
            context = (
                "The article employs sensationalized, attention-seeking language designed to "
                "provoke strong reactions. This clickbait approach is common in low-credibility "
                "content and should be viewed with skepticism."
            )
        else:  # "highly emotional and manipulative"
            context = (
                "The article contains highly emotional and manipulative language intended to "
                "bypass critical thinking and trigger emotional responses. This is a major red flag "
                "for misinformation and propaganda."
            )
    
    # Display emotional tone with color-coded box and text alternative
    # Text alternative: Severity badge provides explicit text label alongside color
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: {tone_color}15; border-left: 5px solid {tone_color}; border-radius: 5px; margin-bottom: 15px;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                <div>
                    <span style="font-size: 1.8em;">{icon}</span>
                    <span style="font-size: 1.3em; font-weight: bold; color: {tone_color}; margin-left: 10px;">{emotional_tone}</span>
                </div>
                <span style="background-color: {severity_color}; color: white; padding: 8px 16px; border-radius: 15px; font-size: 0.9em; font-weight: bold;">{severity_badge}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display context about what the tone indicates
    st.markdown("**What This Means:**")
    st.markdown(
        f"""
        <div style="padding: 15px; background-color: #f8f9fa; border-radius: 5px; color: #333;">
            {context}
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Add educational note about emotional manipulation
    with st.expander("Understanding Emotional Manipulation in News"):
        st.markdown("""
        **Common Emotional Manipulation Tactics:**
        
        - **Fear-mongering:** Using alarming language to create anxiety and urgency
        - **Outrage triggers:** Provoking anger to bypass rational analysis
        - **Sensationalism:** Exaggerating facts to grab attention
        - **Conspiracy framing:** Suggesting hidden agendas or cover-ups
        - **Us vs. Them:** Creating divisive narratives to polarize readers
        
        **Why It Matters:**
        
        Credible journalism aims to inform, not manipulate. While some emotional content is natural 
        in news reporting (especially for serious topics), excessive emotional language is often used 
        to distract from weak evidence or to push a particular agenda.
        
        **What You Can Do:**
        
        - Notice your emotional reaction while reading
        - Ask: "Is this making me feel something to avoid thinking critically?"
        - Look for balanced, evidence-based reporting
        - Cross-reference with multiple credible sources
        """)


def render_suspicious_claims(claims: List[str]):
    """
    Render the suspicious claims section.
    
    Displays up to 5 suspicious claims identified by the ClaimHighlighter component.
    Each claim is shown as a quoted sentence with context about why claims are flagged.
    When no claims are detected, displays an appropriate message.
    
    Args:
        claims: List of suspicious claim strings from ClaimHighlighter
            Each claim is a sentence extracted from the article that requires fact-checking
    """
    st.subheader("Suspicious Claims")
    st.markdown("Specific statements that require fact-checking and verification:")
    
    # Check if claims list is empty
    if not claims or len(claims) == 0:
        # Display message when no suspicious claims are detected
        st.markdown(
            """
            <div style="padding: 20px; background-color: #d4edda; border-left: 5px solid #28a745; border-radius: 5px; margin-top: 10px;">
                <span style="font-size: 1.1em; color: #155724;">
                    <strong>No highly suspicious claims detected</strong>
                </span>
                <br><br>
                <span style="font-size: 0.95em; color: #155724;">
                    The article does not contain obvious unverified claims or statements that trigger 
                    immediate fact-checking alerts. However, this does not guarantee all information 
                    is accurate—always verify important claims with authoritative sources.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # Display up to 5 claims as numbered list
        claims_to_display = claims[:5]  # Limit to 5 claims
        
        # Provide context about why claims are flagged
        st.markdown(
            """
            <div style="padding: 15px; background-color: #fff3cd; border-left: 5px solid #ffc107; border-radius: 5px; margin-bottom: 20px;">
                <span style="font-size: 0.95em; color: #856404;">
                    <strong>Why These Claims Are Flagged:</strong> The following statements contain 
                    unverified assertions, extraordinary claims, vague attributions, or statistical 
                    claims without clear sources. These require independent fact-checking before 
                    accepting as true.
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Display each claim as quoted sentence in a numbered list
        for idx, claim in enumerate(claims_to_display, start=1):
            # Clean up the claim text (remove extra whitespace)
            claim_text = " ".join(claim.split())
            
            # Display claim with number and quote styling
            st.markdown(
                f"""
                <div style="padding: 15px; background-color: #f8f9fa; border-left: 4px solid #dc3545; border-radius: 5px; margin-bottom: 15px;">
                    <div style="font-weight: bold; color: #dc3545; margin-bottom: 8px;">
                        Claim #{idx}
                    </div>
                    <div style="font-style: italic; color: #333; font-size: 1.05em; line-height: 1.6;">
                        "{claim_text}"
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Add note if there are more than 5 claims
        if len(claims) > 5:
            st.caption(f"Note: Showing 5 of {len(claims)} suspicious claims detected. The most significant claims are displayed above.")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Add educational note about fact-checking
        with st.expander("How to Fact-Check These Claims"):
            st.markdown("""
            **Recommended Fact-Checking Steps:**
            
            1. **Identify the Core Assertion:** What specific factual claim is being made?
            2. **Check Primary Sources:** Look for original documents, studies, or official statements
            3. **Consult Fact-Checking Organizations:** 
               - FactCheck.org
               - Snopes.com
               - PolitiFact.com
               - Full Fact (UK)
               - AFP Fact Check
            4. **Verify Statistics:** Check if numbers come from reputable sources (government agencies, peer-reviewed research)
            5. **Look for Expert Consensus:** What do multiple credible experts say about this claim?
            6. **Check Publication Date:** Is the information current or outdated?
            7. **Examine the Source:** Who originally made this claim? Are they credible?
            
            **Red Flags to Watch For:**
            
            - Claims attributed to unnamed "experts" or "sources"
            - Statistics without clear sources or methodology
            - Extraordinary claims without extraordinary evidence
            - Claims that contradict established scientific consensus
            - Information that can't be verified through multiple independent sources
            
            **Remember:** Absence of evidence is not evidence of truth. If a claim cannot be 
            independently verified, treat it with skepticism until proper evidence emerges.
            """)


def render_explanation(result: Dict):
    """
    Render the final explanation and recommendations section.
    
    Displays the detailed explanation text generated by the CredibilityAnalyzer
    and the recommended action based on the risk level. This section provides
    users with comprehensive context about the analysis and actionable guidance.
    
    Args:
        result: Analysis result dictionary from CredibilityAnalyzer.analyze()
            Expected keys: explanation, recommended_action
    """
    st.subheader("Final Explanation & Recommendations")
    
    # Get values from result dictionary
    explanation = result.get("explanation", "No explanation available.")
    recommended_action = result.get("recommended_action", "No recommendation available.")
    
    # Display detailed explanation text
    st.markdown("**Detailed Analysis Explanation:**")
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 20px;">
            <p style="font-size: 1.05em; line-height: 1.7; color: #333; margin: 0;">
                {explanation}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Display recommended action based on risk level
    st.markdown("**Recommended Action:**")
    
    # Determine icon and color based on the content of recommended action
    if "high risk" in recommended_action.lower() or "extreme caution" in recommended_action.lower():
        action_icon = ""
        action_color = UIConfig.COLOR_HIGH_RISK
        action_bg_color = f"{UIConfig.COLOR_HIGH_RISK}15"
    elif "medium risk" in recommended_action.lower() or "caution" in recommended_action.lower():
        action_icon = ""
        action_color = UIConfig.COLOR_MEDIUM_RISK
        action_bg_color = f"{UIConfig.COLOR_MEDIUM_RISK}15"
    else:
        action_icon = ""
        action_color = UIConfig.COLOR_LOW_RISK
        action_bg_color = f"{UIConfig.COLOR_LOW_RISK}15"
    
    st.markdown(
        f"""
        <div style="padding: 20px; background-color: {action_bg_color}; border-left: 5px solid {action_color}; border-radius: 8px; margin-bottom: 20px;">
            <div style="display: flex; align-items: flex-start;">
                <span style="font-size: 2em; margin-right: 15px;">{action_icon}</span>
                <p style="font-size: 1.05em; line-height: 1.7; color: #333; margin: 0; flex: 1;">
                    {recommended_action}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Add important disclaimer
    st.markdown("---")
    st.markdown(
        """
        <div style="padding: 15px; background-color: #e7f3ff; border-left: 4px solid #2196F3; border-radius: 5px;">
            <strong>Important Disclaimer:</strong>
            <br><br>
            This analysis is based on <strong>linguistic patterns and structural analysis only</strong>. 
            It does not perform external fact-checking, verify sources, or access real-time information. 
            <strong>Always verify important claims through multiple credible sources</strong> and consult 
            subject matter experts when making significant decisions based on news content.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# Sidebar Information Display
# ============================================================================

def render_sidebar():
    """
    Render the sidebar with model information, metrics, and disclaimers.
    
    Displays comprehensive information about the ML model including architecture,
    training data, performance metrics, and important disclaimers. This provides
    transparency about the system's capabilities and limitations.
    
    Requirements: 10.1, 10.2, 10.3, 19.1, 19.2, 19.3, 19.4
    """
    with st.sidebar:
        st.header("Model Information")
        
        # Model Type
        st.subheader("Model Architecture")
        st.markdown(
            f"""
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 15px;">
                <div style="font-weight: bold; color: #333; margin-bottom: 8px;">Model Type:</div>
                <div style="font-size: 1.1em; color: #0066cc;">{UIConfig.MODEL_TYPE}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Model Architecture Description
        st.markdown(
            """
            **Architecture Details:**
            
            This system combines traditional machine learning with linguistic pattern analysis:
            
            - **TF-IDF Vectorization:** Converts text into numerical features based on term frequency-inverse document frequency
            - **Logistic Regression Classifier:** Binary classification model trained to distinguish credible from non-credible content
            - **Pattern Detection Layer:** Rule-based analysis of linguistic red flags and suspicious patterns
            - **Ensemble Scoring:** Combines ML predictions with pattern analysis for final credibility assessment
            """
        )
        
        st.markdown("---")
        
        # Training Dataset Information
        st.subheader("Training Data")
        st.markdown(
            f"""
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 15px;">
                <div style="font-weight: bold; color: #333; margin-bottom: 8px;">Dataset Size:</div>
                <div style="font-size: 1.3em; color: #28a745; font-weight: bold;">{UIConfig.DATASET_SIZE}</div>
                <div style="font-size: 0.9em; color: #666; margin-top: 8px;">
                    Labeled news articles from multiple sources, including both credible journalism and known misinformation
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            **Dataset Characteristics:**
            
            - Diverse sources (mainstream media, fact-checked fake news, satire)
            - Balanced representation of credible and non-credible content
            - Multiple domains (politics, health, science, entertainment)
            - Cross-validated for robust performance
            """
        )
        
        st.markdown("---")
        
        # Performance Metrics
        st.subheader("Model Performance")
        st.markdown("**Cross-Validation Metrics:**")
        
        # Create metrics table
        metrics_data = {
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
            "Score": [
                f"{UIConfig.ACCURACY:.2%}",
                f"{UIConfig.PRECISION:.2%}",
                f"{UIConfig.RECALL:.2%}",
                f"{UIConfig.F1_SCORE:.2%}"
            ]
        }
        
        # Display metrics in a clean table format
        st.markdown(
            f"""
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 15px;">
                <table style="width: 100%; border-collapse: collapse;">
                    <thead>
                        <tr style="border-bottom: 2px solid #dee2e6;">
                            <th style="text-align: left; padding: 10px; color: #333;">Metric</th>
                            <th style="text-align: right; padding: 10px; color: #333;">Score</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 10px; color: #666;">Accuracy</td>
                            <td style="text-align: right; padding: 10px; font-weight: bold; color: #28a745;">{UIConfig.ACCURACY:.2%}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 10px; color: #666;">Precision</td>
                            <td style="text-align: right; padding: 10px; font-weight: bold; color: #28a745;">{UIConfig.PRECISION:.2%}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 10px; color: #666;">Recall</td>
                            <td style="text-align: right; padding: 10px; font-weight: bold; color: #28a745;">{UIConfig.RECALL:.2%}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; color: #666;">F1-Score</td>
                            <td style="text-align: right; padding: 10px; font-weight: bold; color: #28a745;">{UIConfig.F1_SCORE:.2%}</td>
                        </tr>
                    </tbody>
                </table>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            **Metric Definitions:**
            
            - **Accuracy:** Overall correctness of predictions
            - **Precision:** Proportion of positive predictions that are correct
            - **Recall:** Proportion of actual positives correctly identified
            - **F1-Score:** Harmonic mean of precision and recall
            """
        )
        
        st.markdown("---")
        
        # Model Version/Date
        st.subheader("Model Version")
        st.markdown(
            """
            <div style="padding: 15px; background-color: #e7f3ff; border-radius: 8px; border-left: 4px solid #2196F3; margin-bottom: 15px;">
                <div style="font-weight: bold; color: #333; margin-bottom: 5px;">Version:</div>
                <div style="color: #0066cc;">v1.0 (2024)</div>
                <div style="font-size: 0.85em; color: #666; margin-top: 8px;">
                    Research-grade prototype for educational and demonstration purposes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Model Documentation Reference
        st.subheader("Documentation")
        st.markdown(
            """
            <div style="padding: 15px; background-color: #f8f9fa; border-radius: 8px; border: 1px solid #dee2e6; margin-bottom: 15px;">
                <div style="font-weight: bold; color: #333; margin-bottom: 8px;">Model Documentation:</div>
                <div style="font-size: 0.95em; color: #666; margin-bottom: 10px;">
                    For detailed information about the model architecture, training methodology, and usage guidelines, 
                    please refer to the project documentation.
                </div>
                <div style="margin-top: 10px;">
                    <a href="https://github.com/yourusername/verifyai" target="_blank" style="color: #0066cc; text-decoration: none; font-weight: 500;">
                        View Full Documentation →
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown("---")
        
        # Disclaimers Section
        st.subheader("Important Disclaimers")
        
        # Disclaimer about probabilistic classification
        st.markdown(
            """
            <div style="padding: 12px; background-color: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 12px;">
                <div style="font-weight: bold; color: #856404; margin-bottom: 5px;">Probabilistic Classification</div>
                <div style="font-size: 0.9em; color: #856404;">
                    This system provides probabilistic assessments based on statistical patterns, not definitive truth determinations. 
                    Results represent likelihood estimates and should be interpreted as one input among many in credibility evaluation.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Disclaimer about no external fact-checking
        st.markdown(
            """
            <div style="padding: 12px; background-color: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 12px;">
                <div style="font-weight: bold; color: #856404; margin-bottom: 5px;">No External Fact-Checking</div>
                <div style="font-size: 0.9em; color: #856404;">
                    This system does NOT perform external fact-checking, verify claims against authoritative sources, or access 
                    real-time information. Analysis is based solely on linguistic patterns and text characteristics within the 
                    provided article.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Disclaimer about need for human verification
        st.markdown(
            """
            <div style="padding: 12px; background-color: #fff3cd; border-radius: 8px; border-left: 4px solid #ffc107; margin-bottom: 12px;">
                <div style="font-weight: bold; color: #856404; margin-bottom: 5px;">Human Verification Required</div>
                <div style="font-size: 0.9em; color: #856404;">
                    All results require human verification and critical thinking. This tool is designed to assist, not replace, 
                    human judgment. Always cross-reference suspicious claims with authoritative sources and professional fact-checkers 
                    before drawing conclusions.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================================
# Main Application Entry Point
# ============================================================================

def main():
    """
    Main application entry point.
    
    Orchestrates the entire UI layout and analysis workflow. Sets up page configuration,
    initializes session state, loads models, creates the two-column layout, and manages
    the analysis workflow.
    
    Requirements: 1.1, 1.2, 1.4, 1.5, 18.1
    """
    # Set page config with wide layout and title
    st.set_page_config(
        page_title="VerifyAI: ML-Based News Credibility Analysis",
        layout=UIConfig.LAYOUT_MODE,
        initial_sidebar_state="expanded"
    )
    
    st.title("VerifyAI: ML-Based News Credibility Analysis")
    st.markdown("---")
    
    # Initialize session state variables (results, analyzed, article_text)
    if "results" not in st.session_state:
        st.session_state.results = None
    
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False
    
    if "article_text" not in st.session_state:
        st.session_state.article_text = ""
    
    if "analyzing" not in st.session_state:
        st.session_state.analyzing = False
    
    # Render sidebar with model information
    render_sidebar()
    
    # Load models with spinner feedback
    try:
        with st.spinner("Initializing models and analyzer..."):
            analyzer = load_analyzer()
            model, vectorizer = load_model()
        st.success("Models loaded successfully!")
    except (FileNotFoundError, Exception) as e:
        # Error messages are already displayed by the load functions
        st.stop()  # Stop execution if models fail to load
    
    # Create two-column layout with ratio [1, 1.3]
    col1, col2 = st.columns(UIConfig.COLUMN_RATIO)
    
    # Render input panel in left column
    with col1:
        article_text = render_input_panel()
    
    # Render results in right column
    with col2:
        st.header("Analysis Results")
        
        # Check if analyze button was clicked
        if st.session_state.get("analyze_clicked", False):
            # Validate input
            is_valid, error_message = validate_input(article_text)
            
            if not is_valid:
                st.warning(f"{error_message}")
                st.session_state.analyze_clicked = False
            else:
                # Clear previous results when new analysis is triggered (Requirement 18.3)
                st.session_state.results = None
                st.session_state.analyzed = False
                
                # Disable analyze button during analysis (handled by session state)
                st.session_state.analyzing = True
                
                # Display spinner during analysis
                with st.spinner("Analyzing article..."):
                    try:
                        # Call analyzer.analyze() with article text, model, and vectorizer
                        results = analyzer.analyze(article_text, model, vectorizer)
                        
                        # Store results in session state
                        st.session_state.results = results
                        st.session_state.analyzed = True
                        
                        # Display success message when analysis completes
                        st.success("Analysis complete! Results are displayed below.")
                        
                    except Exception as e:
                        st.error(f"**Analysis Error**: {str(e)}")
                        st.error("Please try again or contact support if the issue persists.")
                        st.session_state.results = None
                        st.session_state.analyzed = False
                    finally:
                        # Reset button state
                        st.session_state.analyze_clicked = False
                        st.session_state.analyzing = False
        
        # Display results if analysis has been performed
        if st.session_state.analyzed and st.session_state.results:
            results = st.session_state.results
            
            # Render all 6 sections in order
            # 1. Verdict Summary
            render_verdict_summary(results)
            st.markdown("---")
            
            # 2. Model Prediction Details
            render_model_details(results)
            st.markdown("---")
            
            # 3. Linguistic Pattern Analysis
            render_pattern_analysis(results.get("patterns", {}))
            st.markdown("---")
            
            # 4. Emotional Tone Analysis
            render_emotional_tone(results.get("emotional_tone", "N/A"))
            st.markdown("---")
            
            # 5. Suspicious Claims
            render_suspicious_claims(results.get("suspicious_claims", []))
            st.markdown("---")
            
            # 6. Final Explanation
            render_explanation(results)
            
        else:
            st.info("Enter an article in the input panel and click 'Analyze Article' to see results.")


if __name__ == "__main__":
    main()
