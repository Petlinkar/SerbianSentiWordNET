from transformers import pipeline

class SentimentPipeline:
    """
    This class defines a custom sentiment analysis pipeline using Hugging Face's Transformers.
    
    The pipeline uses two separate models for predicting positive/non-positive and 
    negative/non-negative sentiment respectively.

    Inputs:
        Single text string or a list of text strings for sentiment analysis.

    Returns:
        If a single text string is provided, a single dictionary is returned with POS, NEG, and OBJ scores.
        If a list of text strings is provided, a list of dictionaries is returned with each dictionary 
        representing POS, NEG, and OBJ scores for the corresponding text.

    Usage:
        sentiment_pipeline = SentimentPipeline(YOUR_POS_MODEL, YOUR_NEG_MODEL)
        result = sentiment_pipeline("Your glossed text here")
        results = sentiment_pipeline(["Your first glossed text here", "Your second glossed text here"])
    """

    def __init__(self, model_path_positive, model_path_negative):
        """
        Constructor for the SentimentPipeline class.
        Initializes two pipelines using Hugging Face's Transformers, one for positive and one for negative sentiment.
        """
        self.pos_pipeline = pipeline('text-classification', model=model_path_positive)
        self.neg_pipeline = pipeline('text-classification', model=model_path_negative)

    def __call__(self, texts):
        """
        Callable method for the SentimentPipeline class. Processes the given text(s) and returns sentiment scores.
        """
        
        # Check if input is a single string. If it is, convert it into a list.
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            # Run the text through the pipelines
            pos_result = self.pos_pipeline(text)[0]
            neg_result = self.neg_pipeline(text)[0]

            # Calculate probabilities for positive/non-positive and negative/non-negative.
            # If the label is POSITIVE/NEGATIVE, the score for positive/negative is the score returned by the model, 
            # and the score for non-positive/non-negative is 1 - the score returned by the model.
            # If the label is NON-POSITIVE/NON-NEGATIVE, the score for non-positive/non-negative is the score returned by the model,
            # and the score for positive/negative is 1 - the score returned by the model.
            Pt, Pn = (pos_result['score'], 1 - pos_result['score']) if pos_result['label'] == 'POSITIVE' else (1 - pos_result['score'], pos_result['score'])
            Nt, Nn = (neg_result['score'], 1 - neg_result['score']) if neg_result['label'] == 'NEGATIVE' else (1 - neg_result['score'], neg_result['score'])

            # Calculate POS, NEG, OBJ scores using the formulas provided
            POS = Pt * Nn
            NEG = Nt * Pn
            OBJ = 1 - POS - NEG

            # Append the scores to the results
            results.append({"POS": POS, "NEG": NEG, "OBJ": OBJ})

        # If the input was a single string, return a single dictionary. Otherwise, return a list of dictionaries.
        return results if len(results) > 1 else results[0]
