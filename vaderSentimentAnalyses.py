import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Download NLTK dependencies (only needed once)
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Sentiment Analyzer
analyser = SentimentIntensityAnalyzer()

# Load Dataset
data_file = pd.read_csv('C:/TUS-ML/Steam review data from previous study(in).csv')


# Function for Preprocessing Text
def preprocess_text(text):
    if pd.isna(text):  # Handle NaN values
        return ""

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces

    # Tokenization
    tokens = word_tokenize(text)

    # Stopword Removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# Apply Preprocessing to the Reviews
data_file['cleaned_review'] = data_file['review'].astype(str).apply(preprocess_text)


# Function for Sentiment Analysis
def sentiment_vader(text):
    scores = analyser.polarity_scores(text)
    return scores['compound']  # Returns sentiment score (-1 to 1)

def sentiment_vader_classifier(text):
    scores = analyser.polarity_scores(text)
    if scores['compound'] > 0.05:
        return "positive"
    elif scores['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Apply Sentiment Analysis to Cleaned Text
data_file['sentiment_score'] = data_file['cleaned_review'].apply(sentiment_vader)

# Create column with overall sentiment
data_file['sentiment_vader'] = data_file['cleaned_review'].apply(lambda x: sentiment_vader_classifier(str(x)))

# Define Refined Heuristic Playability Categories
heuristic_keywords = {
    "Engaging Experience (Positive)": ["fun", "addictive", "immersive", "exciting"],
    "Lack of Engagement (Negative)": ["boring", "tedious", "repetitive", "dull"],
    "Challenging Gameplay (Positive)": ["challenging", "rewarding", "strategic"],
    "Frustrating Difficulty (Negative)": ["unfair", "grindy", "unbalanced"],
    "Good Controls & UI (Positive)": ["smooth", "responsive", "intuitive", "user-friendly"],
    "Bad Controls & UI (Negative)": ["laggy", "clunky", "unresponsive"],
    "Accessibility Features (Positive)": ["colorblind", "subtitles", "adjustable"],
    "Lack of Accessibility (Negative)": ["inaccessible", "hard to read", "tiny text"]
}


# Function to Assign Heuristic Categories with Sentiment
def assign_heuristic_category(review, sentiment_score):
    matched_categories = set()

    for category, keywords in heuristic_keywords.items():
        for kw in keywords:
            # Match words & phrases
            if re.search(rf"\b{kw}\b", review, re.IGNORECASE):
                if sentiment_score > 0.05:
                    matched_categories.add(f"{category} (Positive)")
                elif sentiment_score < -0.05:
                    matched_categories.add(f"{category} (Negative)")
                else:
                    matched_categories.add(f"{category} (Neutral)")

    # If sentiment is highly negative and no keyword is matched, classify as "General Negative Feedback"
    if sentiment_score < -0.4 and not matched_categories:
        return "General Negative Feedback"

    return ", ".join(matched_categories) if matched_categories else "Other"



# Assign Categories using Cleaned Reviews & Sentiment Score
data_file['heuristic_category'] = data_file.apply(
    lambda x: assign_heuristic_category(x['cleaned_review'], x['sentiment_score']), axis=1)

# Extract Most Frequent Keywords using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
tfidf_matrix = vectorizer.fit_transform(data_file['cleaned_review'])
keywords = vectorizer.get_feature_names_out()

# Perform Topic Modeling (LDA) for Insight Extraction
lda = LatentDirichletAllocation(n_components=5, random_state=42)
lda.fit(tfidf_matrix)


# Function to Display Topics
def display_topics(model, feature_names, no_top_words):
    topics = {}
    for topic_idx, topic in enumerate(model.components_):
        topics[f"Topic {topic_idx + 1}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topics


# Get Topics
topics = display_topics(lda, keywords, 10)

# Convert to DataFrame
topic_df = pd.DataFrame(topics)

# Save Processed Data
data_file.to_csv('C:/TUS-ML/DATA_FILE_RESULT_ANALYZED.csv', index=False)

# Generate Word Cloud for Negative Reviews
# negative_reviews = " ".join(data_file[data_file['sentiment_score'] < -0.05]['cleaned_review'])
# wordcloud = WordCloud(width=800, height=400, background_color="black").generate(negative_reviews)

# Show the WordCloud
# plt.figure(figsize=(10, 5))
# plt.imshow(wordcloud, interpolation="bilinear")
# plt.axis("off")
# plt.show()
