"""
Core credibility analyzer for news article assessment.

This module implements the main CredibilityAnalyzer class that orchestrates
the credibility assessment pipeline, combining model predictions with pattern
analysis to generate comprehensive credibility assessments.
"""

from typing import Dict, Any


class CredibilityAnalyzer:
    """
    Main analyzer class that orchestrates credibility assessment.
    
    This class combines machine learning model predictions with linguistic
    pattern analysis to classify news articles and calculate credibility scores.
    """
    
    def __init__(self):
        """Initialize the CredibilityAnalyzer."""
        pass
    
    def calculate_pattern_score(self, patterns: Dict[str, float]) -> float:
        """
        Aggregate pattern detection results into a single pattern score.
        
        This method combines multiple pattern detection metrics into a normalized
        score between 0.0 and 1.0, where higher values indicate more suspicious
        patterns associated with misinformation.
        
        Args:
            patterns: Dictionary containing pattern detection results with keys:
                - sensational_phrases: Count of sensational keywords
                - excessive_caps: Ratio of excessively capitalized words
                - vague_sources: Count of vague source references
                - conspiracy_framing: Count of conspiracy keywords
                - emotional_manipulation: Count of emotional manipulation keywords
                - one_sided: Score indicating one-sided narrative (0.0-1.0)
                - no_evidence: Score indicating lack of evidence (0.0-1.0)
                - extreme_adjectives: Count of extreme adjectives
                - clickbait: Count of clickbait patterns
                
        Returns:
            Normalized pattern score between 0.0 and 1.0
        """
        # Extract pattern values with defaults
        sensational = patterns.get("sensational_phrases", 0)
        excessive_caps = patterns.get("excessive_caps", 0.0)
        vague_sources = patterns.get("vague_sources", 0)
        conspiracy = patterns.get("conspiracy_framing", 0)
        emotional = patterns.get("emotional_manipulation", 0)
        one_sided = patterns.get("one_sided", 0.0)
        no_evidence = patterns.get("no_evidence", 0.0)
        extreme_adj = patterns.get("extreme_adjectives", 0)
        clickbait = patterns.get("clickbait", 0)
        
        # Normalize count-based patterns (cap at reasonable thresholds)
        sensational_norm = min(1.0, sensational / 5.0)
        vague_sources_norm = min(1.0, vague_sources / 3.0)
        conspiracy_norm = min(1.0, conspiracy / 2.0)
        emotional_norm = min(1.0, emotional / 4.0)
        extreme_adj_norm = min(1.0, extreme_adj / 6.0)
        clickbait_norm = min(1.0, clickbait / 2.0)
        
        # Weighted average of all patterns
        # Ratio-based patterns (excessive_caps, one_sided, no_evidence) already normalized
        pattern_score = (
            sensational_norm * 0.15 +
            excessive_caps * 0.10 +
            vague_sources_norm * 0.15 +
            conspiracy_norm * 0.15 +
            emotional_norm * 0.10 +
            one_sided * 0.10 +
            no_evidence * 0.10 +
            extreme_adj_norm * 0.10 +
            clickbait_norm * 0.05
        )
        
        return pattern_score
    
    def classify_credibility(
        self,
        text: str,
        model_prediction: int,
        model_confidence: float,
        detected_patterns: Dict[str, float]
    ) -> str:
        """
        Assign credibility classification based on model prediction and pattern analysis.
        
        This method implements the classification logic from the design document,
        combining model predictions with pattern analysis to determine the
        appropriate credibility label.
        
        Args:
            text: The article text
            model_prediction: Model prediction (0 for fake, 1 for real)
            model_confidence: Model confidence score (0.0-1.0)
            detected_patterns: Dictionary of detected patterns
            
        Returns:
            Classification string: "REAL", "FAKE", "MISLEADING", or "UNVERIFIED"
        """
        # Handle insufficient text case
        if len(text) < 50:
            return "UNVERIFIED"
        
        # Calculate pattern score
        pattern_score = self.calculate_pattern_score(detected_patterns)
        
        # Classification logic based on model prediction and confidence
        if model_prediction == 0 and model_confidence > 0.75:
            # Model strongly predicts fake
            if pattern_score > 0.7:
                return "FAKE"
            else:
                return "MISLEADING"
        
        if model_prediction == 1 and model_confidence > 0.75:
            # Model strongly predicts real
            if pattern_score < 0.3:
                return "REAL"
            else:
                return "MISLEADING"
        
        if model_confidence < 0.5:
            # Low model confidence
            return "UNVERIFIED"
        
        # Medium confidence cases
        if pattern_score > 0.5:
            return "MISLEADING"
        else:
            return "REAL"
    
    def calculate_credibility_score(
        self,
        model_confidence: float,
        model_prediction: int,
        pattern_score: float
    ) -> int:
        """
        Calculate numerical credibility score (0-100).
        
        This method combines model confidence and prediction with pattern analysis
        to generate a credibility score. Higher scores indicate higher credibility.
        
        Args:
            model_confidence: Model confidence score (0.0-1.0)
            model_prediction: Model prediction (0 for fake, 1 for real)
            pattern_score: Aggregated pattern score (0.0-1.0, higher = more suspicious)
            
        Returns:
            Credibility score as integer in range [0, 100]
        """
        # Model-driven component (0-100): reflects ML confidence in its prediction
        if model_prediction == 1:
            model_component = model_confidence * 100
        else:
            # Fake prediction → lower score when model is more confident
            model_component = (1 - model_confidence) * 100

        # Pattern-driven component (0-100): inverted so high suspicion → low score
        # pattern_score of 1.0 (very suspicious) → pattern_component of 0
        # pattern_score of 0.0 (clean)           → pattern_component of 100
        pattern_component = (1 - pattern_score) * 100

        # Weighted blend: ML model carries 70% of the score, patterns 30%.
        # This prevents keyword-based patterns from overriding a high-confidence
        # model prediction (e.g., legitimate political/religious journalism
        # that uses emotional language but is factually credible).
        credibility_score = model_component * 0.70 + pattern_component * 0.30

        # Clamp to [0, 100] range
        return round(max(0, min(100, credibility_score)))

    def determine_risk_level(self, credibility_score: int) -> str:
        """
        Determine risk level based on credibility score thresholds.
        
        This method implements the risk level determination logic from the design
        document, mapping credibility scores to risk categories.
        
        Args:
            credibility_score: Credibility score (0-100)
            
        Returns:
            Risk level string: "Low Risk", "Medium Risk", or "High Risk"
        """
        if credibility_score >= 75:
            return "Low Risk"
        elif credibility_score >= 40:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def calculate_confidence(
        self,
        model_confidence: float,
        pattern_consistency: float
    ) -> int:
        """
        Calculate system confidence in its assessment.
        
        This method combines model confidence and pattern consistency using a
        weighted average to determine the overall confidence in the assessment.
        
        Args:
            model_confidence: Model confidence score (0.0-1.0)
            pattern_consistency: Pattern consistency score (0.0-1.0)
            
        Returns:
            Confidence score as integer in range [0, 100]
        """
        # Combine model confidence and pattern consistency with weighted average
        # Model confidence weighted at 60%, pattern consistency at 40%
        combined_confidence = (model_confidence * 0.6) + (pattern_consistency * 0.4)
        
        # Convert to 0-100 scale
        confidence_score = combined_confidence * 100
        
        return round(confidence_score)

    def extract_key_indicators(self, patterns: Dict[str, float], text: str) -> list:
        """
        Extract the most significant indicators influencing the credibility assessment.
        
        This method analyzes detected patterns and generates human-readable
        descriptions of the key factors that influenced the credibility score.
        At least one indicator is always returned.
        
        Args:
            patterns: Dictionary containing pattern detection results
            text: The article text
            
        Returns:
            List of key indicator description strings
        """
        indicators = []
        
        # Check each pattern against thresholds and add corresponding indicators
        if patterns.get("sensational_phrases", 0) > 3:
            indicators.append("High use of sensational language")
        
        if patterns.get("excessive_caps", 0.0) > 0.1:
            indicators.append("Excessive capitalization detected")
        
        if patterns.get("vague_sources", 0) > 2:
            indicators.append("Multiple vague source references")
        
        if patterns.get("conspiracy_framing", 0) > 0:
            indicators.append("Conspiracy framing language present")
        
        if patterns.get("emotional_manipulation", 0) > 2:
            indicators.append("Emotional manipulation tactics detected")
        
        if patterns.get("one_sided", 0.0) > 0.7:
            indicators.append("One-sided narrative without counterpoints")
        
        if patterns.get("no_evidence", 0.0) > 0.7:
            indicators.append("Lack of verifiable evidence or data")
        
        if patterns.get("extreme_adjectives", 0) > 5:
            indicators.append("Overuse of extreme adjectives")
        
        if patterns.get("clickbait", 0) > 0:
            indicators.append("Clickbait patterns in text")
        
        # Ensure at least one indicator is always returned
        if len(indicators) == 0:
            indicators.append("Balanced language and structure")
            indicators.append("Appropriate use of sources")
        
        return indicators
    
    def generate_analysis_summary(
        self,
        classification: str,
        credibility_score: int,
        key_indicators: list
    ) -> str:
        """
        Generate a concise 2-4 sentence summary of the credibility analysis.
        
        This method creates a human-readable summary that describes the overall
        assessment and references the primary factors that influenced it.
        
        Args:
            classification: The credibility classification (REAL, FAKE, MISLEADING, UNVERIFIED)
            credibility_score: The credibility score (0-100)
            key_indicators: List of key indicator descriptions
            
        Returns:
            Analysis summary string (2-4 sentences)
        """
        # Start with classification and score
        if classification == "REAL":
            summary = f"This article appears credible with a credibility score of {credibility_score}/100. "
        elif classification == "FAKE":
            summary = f"This article shows strong indicators of misinformation with a credibility score of {credibility_score}/100. "
        elif classification == "MISLEADING":
            summary = f"This article contains misleading elements with a credibility score of {credibility_score}/100. "
        else:  # UNVERIFIED
            summary = f"This article cannot be reliably assessed with a credibility score of {credibility_score}/100. "
        
        # Add primary factors
        if len(key_indicators) > 0:
            # Reference top 2-3 indicators
            primary_indicators = key_indicators[:3]
            if len(primary_indicators) == 1:
                summary += f"The primary factor is: {primary_indicators[0].lower()}. "
            elif len(primary_indicators) == 2:
                summary += f"Key factors include: {primary_indicators[0].lower()} and {primary_indicators[1].lower()}. "
            else:
                summary += f"Key factors include: {primary_indicators[0].lower()}, {primary_indicators[1].lower()}, and {primary_indicators[2].lower()}. "
        
        # Add contextual statement based on classification
        if classification == "REAL":
            summary += "The content demonstrates balanced reporting and appropriate sourcing."
        elif classification == "FAKE":
            summary += "Multiple red flags suggest this content should be treated with extreme skepticism."
        elif classification == "MISLEADING":
            summary += "While some elements may be factual, the overall presentation raises concerns."
        else:  # UNVERIFIED
            summary += "Additional information would be needed for a more definitive assessment."
        
        return summary
    
    def generate_recommended_action(self, risk_level: str) -> str:
        """
        Generate recommended action based on the risk level.
        
        This method provides actionable guidance to users based on the
        assessed risk level of the content.
        
        Args:
            risk_level: The risk level (Low Risk, Medium Risk, High Risk)
            
        Returns:
            Recommended action string
        """
        if risk_level == "High Risk":
            return "Exercise extreme caution with this content. Verify claims through multiple independent and reputable sources before accepting or sharing. Consider this content potentially misleading or false."
        elif risk_level == "Medium Risk":
            return "Approach this content with caution. Cross-reference key claims with other credible sources and look for additional evidence before drawing conclusions or sharing."
        else:  # Low Risk
            return "This content appears credible based on linguistic analysis. However, always maintain critical thinking and verify important claims through additional sources when making significant decisions."
    
    def generate_explanation(
        self,
        classification: str,
        credibility_score: int,
        patterns: Dict[str, float],
        indicators: list
    ) -> str:
        """
        Generate a detailed explanation of the credibility assessment.
        
        This method creates a comprehensive explanation that describes the
        reasoning behind the classification and score, referencing specific
        patterns and findings from the analysis.
        
        Args:
            classification: The credibility classification
            credibility_score: The credibility score (0-100)
            patterns: Dictionary of detected patterns
            indicators: List of key indicators
            
        Returns:
            Detailed explanation string
        """
        explanation = f"The article received a classification of '{classification}' with a credibility score of {credibility_score}/100. "
        
        # Explain classification reasoning
        if classification == "REAL":
            explanation += "This classification indicates that the linguistic patterns and content structure align with credible journalism. "
        elif classification == "FAKE":
            explanation += "This classification indicates strong linguistic patterns associated with misinformation and fabricated content. "
        elif classification == "MISLEADING":
            explanation += "This classification indicates a mix of credible and suspicious elements, suggesting partial truth with potential distortion. "
        else:  # UNVERIFIED
            explanation += "This classification indicates insufficient information or ambiguous patterns that prevent a definitive assessment. "
        
        # Reference specific patterns
        pattern_score = self.calculate_pattern_score(patterns)
        explanation += f"The overall pattern analysis score is {pattern_score:.2f} (on a scale where higher values indicate more suspicious patterns). "
        
        # Detail specific findings
        if len(indicators) > 0:
            explanation += "Specific findings include: "
            for i, indicator in enumerate(indicators):
                if i == 0:
                    explanation += indicator.lower()
                elif i == len(indicators) - 1:
                    explanation += f", and {indicator.lower()}"
                else:
                    explanation += f", {indicator.lower()}"
            explanation += ". "
        
        # Add pattern-specific details
        notable_patterns = []
        if patterns.get("sensational_phrases", 0) > 3:
            notable_patterns.append(f"sensational language ({patterns['sensational_phrases']} instances)")
        if patterns.get("vague_sources", 0) > 2:
            notable_patterns.append(f"vague source references ({patterns['vague_sources']} instances)")
        if patterns.get("emotional_manipulation", 0) > 2:
            notable_patterns.append(f"emotional manipulation tactics ({patterns['emotional_manipulation']} instances)")
        if patterns.get("conspiracy_framing", 0) > 0:
            notable_patterns.append(f"conspiracy framing language ({patterns['conspiracy_framing']} instances)")
        
        if notable_patterns:
            explanation += "Notable patterns detected: "
            explanation += ", ".join(notable_patterns) + ". "
        
        # Conclude with assessment basis
        explanation += "This assessment is based exclusively on linguistic and structural analysis of the provided text, without external fact-checking or knowledge injection."
        
        return explanation

    def analyze(self, text: str, model: Any, vectorizer: Any) -> Dict[str, Any]:
        """
        Perform comprehensive credibility analysis on news article text.

        This is the main orchestration method that coordinates all analysis
        components to generate a complete credibility assessment.

        Args:
            text: The news article text to analyze
            model: The trained machine learning model
            vectorizer: The TF-IDF vectorizer

        Returns:
            Dictionary containing complete analysis with keys:
            - classification: str (REAL, FAKE, MISLEADING, UNVERIFIED)
            - credibility_score: int (0-100)
            - risk_level: str (Low Risk, Medium Risk, High Risk)
            - confidence: int (0-100)
            - analysis_summary: str
            - key_indicators: list[str]
            - emotional_tone: str
            - suspicious_claims: list[str]
            - recommended_action: str
            - explanation: str
            - model_prediction: int (0 for fake, 1 for real)
            - pattern_score: float (0.0-1.0)
            - patterns: dict (pattern detection results)
        """
        from pattern_detector import PatternDetector
        from emotional_analyzer import EmotionalAnalyzer
        from claim_highlighter import ClaimHighlighter
        import re

        # Validate input text
        if not text or not isinstance(text, str):
            return {
                "classification": "UNVERIFIED",
                "credibility_score": 0,
                "risk_level": "High Risk",
                "confidence": 0,
                "analysis_summary": "No text provided for analysis.",
                "key_indicators": ["Insufficient input"],
                "emotional_tone": "N/A",
                "suspicious_claims": [],
                "recommended_action": "Please provide article text for analysis.",
                "explanation": "INSUFFICIENT INFORMATION"
            }

        # Handle insufficient text case
        if len(text.strip()) < 50:
            return {
                "classification": "UNVERIFIED",
                "credibility_score": 0,
                "risk_level": "High Risk",
                "confidence": 0,
                "analysis_summary": "The provided text is too short for reliable analysis.",
                "key_indicators": ["Insufficient text length"],
                "emotional_tone": "N/A",
                "suspicious_claims": [],
                "recommended_action": "Please provide more substantial article text for analysis.",
                "explanation": "INSUFFICIENT INFORMATION"
            }

        # Clean text for model prediction (same preprocessing as training)
        def clean_text_for_model(text: str) -> str:
            text = str(text).lower()
            text = re.sub(r"<[^>]+>", " ", text)
            text = re.sub(r"[^a-z\s]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        # Get base prediction from existing model
        cleaned_text = clean_text_for_model(text)
        features = vectorizer.transform([cleaned_text])
        model_prediction = int(model.predict(features)[0])

        # Calculate model confidence
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(features)[0]
            model_confidence = float(max(proba))
        elif hasattr(model, "decision_function"):
            decision = abs(float(model.decision_function(features)[0]))
            model_confidence = min(1.0, 0.5 + decision / 10.0)
        else:
            model_confidence = 0.5

        # Initialize analysis components
        pattern_detector = PatternDetector()
        emotional_analyzer = EmotionalAnalyzer()
        claim_highlighter = ClaimHighlighter()

        # Call pattern detector to get pattern analysis
        detected_patterns = pattern_detector.detect_patterns(text)

        # Calculate pattern score
        pattern_score = self.calculate_pattern_score(detected_patterns)

        # Calculate pattern consistency (how consistent patterns are with each other)
        # Higher consistency when patterns align (all high or all low)
        pattern_values = [
            detected_patterns.get("sensational_phrases", 0) / 5.0,
            detected_patterns.get("excessive_caps", 0.0),
            detected_patterns.get("vague_sources", 0) / 3.0,
            detected_patterns.get("conspiracy_framing", 0) / 2.0,
            detected_patterns.get("emotional_manipulation", 0) / 4.0,
            detected_patterns.get("one_sided", 0.0),
            detected_patterns.get("no_evidence", 0.0),
            detected_patterns.get("extreme_adjectives", 0) / 6.0,
            detected_patterns.get("clickbait", 0) / 2.0
        ]
        # Normalize pattern values to [0, 1]
        normalized_values = [min(1.0, v) for v in pattern_values]
        # Calculate variance - lower variance means higher consistency
        mean_val = sum(normalized_values) / len(normalized_values)
        variance = sum((v - mean_val) ** 2 for v in normalized_values) / len(normalized_values)
        pattern_consistency = 1.0 - min(1.0, variance * 2.0)  # Convert variance to consistency

        # Call classification engine to get classification
        classification = self.classify_credibility(
            text, model_prediction, model_confidence, detected_patterns
        )

        # Calculate credibility score
        credibility_score = self.calculate_credibility_score(
            model_confidence, model_prediction, pattern_score
        )

        # Determine risk level
        risk_level = self.determine_risk_level(credibility_score)

        # Calculate confidence score
        confidence = self.calculate_confidence(model_confidence, pattern_consistency)

        # Extract key indicators
        key_indicators = self.extract_key_indicators(detected_patterns, text)

        # Analyze emotional tone using EmotionalAnalyzer
        emotional_tone = emotional_analyzer.analyze_emotional_tone(detected_patterns, text)

        # Identify suspicious claims using ClaimHighlighter
        suspicious_claims = claim_highlighter.identify_suspicious_claims(text)

        # Generate analysis summary
        analysis_summary = self.generate_analysis_summary(
            classification, credibility_score, key_indicators
        )

        # Generate recommended action
        recommended_action = self.generate_recommended_action(risk_level)

        # Generate detailed explanation
        explanation = self.generate_explanation(
            classification, credibility_score, detected_patterns, key_indicators
        )

        # Return complete analysis dictionary
        return {
            "classification": classification,
            "credibility_score": credibility_score,
            "risk_level": risk_level,
            "confidence": confidence,
            "analysis_summary": analysis_summary,
            "key_indicators": key_indicators,
            "emotional_tone": emotional_tone,
            "suspicious_claims": suspicious_claims,
            "recommended_action": recommended_action,
            "explanation": explanation,
            "model_prediction": model_prediction,
            "pattern_score": pattern_score,
            "patterns": detected_patterns
        }

    def format_json_output(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Format and validate analysis results for JSON output.

        This method ensures all required fields are present with correct data types
        and returns a valid JSON structure conforming to the specified schema.

        Args:
            analysis_result: Dictionary containing analysis results

        Returns:
            Validated JSON-ready dictionary with all required fields

        Raises:
            ValueError: If required fields are missing or have incorrect types
        """
        # Define required fields and their expected types
        required_fields = {
            "classification": str,
            "credibility_score": int,
            "risk_level": str,
            "confidence": int,
            "analysis_summary": str,
            "key_indicators": list,
            "emotional_tone": str,
            "suspicious_claims": list,
            "recommended_action": str,
            "explanation": str
        }

        # Validate all required fields are present
        missing_fields = []
        for field in required_fields.keys():
            if field not in analysis_result:
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Validate data types and create formatted output
        formatted_output = {}

        for field, expected_type in required_fields.items():
            value = analysis_result[field]

            # Type validation
            if not isinstance(value, expected_type):
                raise ValueError(
                    f"Field '{field}' has incorrect type. "
                    f"Expected {expected_type.__name__}, got {type(value).__name__}"
                )

            # Additional validation for specific fields
            if field == "classification":
                valid_classifications = {"REAL", "FAKE", "MISLEADING", "UNVERIFIED"}
                if value not in valid_classifications:
                    raise ValueError(
                        f"Invalid classification value: {value}. "
                        f"Must be one of {valid_classifications}"
                    )

            elif field == "credibility_score":
                if not (0 <= value <= 100):
                    raise ValueError(
                        f"credibility_score must be in range [0, 100], got {value}"
                    )

            elif field == "confidence":
                if not (0 <= value <= 100):
                    raise ValueError(
                        f"confidence must be in range [0, 100], got {value}"
                    )

            elif field == "risk_level":
                valid_risk_levels = {"Low Risk", "Medium Risk", "High Risk"}
                if value not in valid_risk_levels:
                    raise ValueError(
                        f"Invalid risk_level value: {value}. "
                        f"Must be one of {valid_risk_levels}"
                    )

            elif field in ["key_indicators", "suspicious_claims"]:
                # Validate that list items are strings
                if not all(isinstance(item, str) for item in value):
                    raise ValueError(
                        f"All items in '{field}' must be strings"
                    )

            # Add validated field to output
            formatted_output[field] = value

        return formatted_output


