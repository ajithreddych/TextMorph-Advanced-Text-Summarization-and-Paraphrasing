import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import string
from textstat import textstat

nltk.download("punkt", quiet=True)

def summarize_text(text, max_sentences=7):
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text

    words = word_tokenize(text.lower())
    words = [w for w in words if w not in string.punctuation]
    freq = Counter(words)

    sentence_scores = {}
    for sent in sentences:
        for word in word_tokenize(sent.lower()):
            if word in freq:
                sentence_scores[sent] = sentence_scores.get(sent, 0) + freq[word]

    top_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:max_sentences]
    return " ".join(top_sentences)

def analyze_readability(text):
    scores = {
        "flesch_kincaid": textstat.flesch_kincaid_grade(text),
        "gunning_fog": textstat.gunning_fog(text),
        "smog_index": textstat.smog_index(text),
        "reading_ease": textstat.flesch_reading_ease(text)
    }

    if scores["reading_ease"] > 70:
        scores["level"] = "Beginner"
    elif scores["reading_ease"] > 40:
        scores["level"] = "Intermediate"
    else:
        scores["level"] = "Advanced"

    return scores
