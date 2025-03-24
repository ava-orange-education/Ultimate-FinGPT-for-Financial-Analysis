import re
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords

# Ensure you have the required nltk data files
nltk.download('stopwords')

def clean_text(raw_html):
    """
    Cleans raw HTML content by removing tags, normalizing text, and eliminating noise.
    
    Parameters:
        raw_html (str): The raw HTML content of a financial news article.
        
    Returns:
        str: A cleaned and normalized text string.
    """
    # Remove HTML tags using BeautifulSoup
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(separator=" ")
    
    # Normalize the text: lowercasing and remove non-alphanumeric characters
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    
    # Optionally remove stopwords (if needed for your analysis)
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    cleaned_text = " ".join(tokens)
    return cleaned_text

# Example usage:
raw_example = "<html><body><h1>Market Update!</h1><p>Stocks soar as tech sector rallies.</p></body></html>"
print("Cleaned Text:", clean_text(raw_example))
