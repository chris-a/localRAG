from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


def anonymize_text(text, entities_to_anonymize=None, **kwargs):
    """
    Anonymizes the given text using Presidio.

    Args:
        text (str): The text to anonymize.
        entities_to_anonymize (list, optional): A list of entities to anonymize.
            If None, all recognized entities will be anonymized.
        **kwargs: Additional keyword arguments passed to the AnonymizerEngine.

    Returns:
        str: The anonymized text.
    """

    # Create an AnalyzerEngine to detect sensitive information
    analyzer = AnalyzerEngine()

    # Analyze the text to detect entities
    results = analyzer.analyze(text=text, language='en')

    # Filter results to include only specified entities
    if entities_to_anonymize:
        results = [result for result in results if result.entity_type in entities_to_anonymize]

    # Create an AnonymizerEngine to anonymize the entities
    anonymizer = AnonymizerEngine()

    # Anonymize the text
    anonymized_text = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        **kwargs
    ).text

    return anonymized_text


# Example usage:
text = """
My name is John Doe, and my email address is john.doe@example.com.
I live in London, and my phone number is 123-456-7890.
"""

# Anonymize all recognized entities
anonymized_text = anonymize_text(text)
print(f"Anonymized text (all entities):\n{anonymized_text}")

# Anonymize only names and email addresses
anonymized_text = anonymize_text(text=text, entities_to_anonymize=["PERSON", "EMAIL_ADDRESS"])
print(f"Anonymized text (names and emails):\n{anonymized_text}")
