import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Setup and Imports
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Streamlit App Title
st.title("Product Reviews Analysis")

# Sample Reviews Dataset - (Product: earphones)
data = {
    'review': [
        "Fantastic audio quality with crisp highs and deep bass. Perfect for audiophiles!",
        "Terrible build quality. The earphones broke after just two weeks of light use.",
        "Comfortable fit and amazing sound isolation. Great for long listening sessions.",
        "Not worth the price. The sound is average and the cable tangles too easily.",
        "Surprisingly good for the price. Excellent clarity and decent bass response.",
        "Stopped working after a month. Customer service wasn't helpful in resolving the issue.",
        "Lightweight design and premium finish. They feel great and sound even better!",
        "The sound leaks a lot, which is annoying when used in quiet places like libraries.",
        "Perfect for workouts! Secure fit and sweat-resistant, plus they sound fantastic.",
        "Connection issues with Bluetooth. Constantly disconnects, making it unusable.",
        "Balanced sound profile with no overemphasis on bass. Truly enjoyable experience.",
        "The earbuds are uncomfortable for extended use. My ears hurt after an hour.",
        "Solid performance for the price. They handle all genres of music really well.",
        "Terrible noise cancellation. They barely block out any background sound.",
        "Stylish design and impressive sound quality. The vocals are crystal clear.",
        "Overpriced for what they offer. You can get better sound at half the price.",
        "Love the case and overall build quality. Very portable and durable.",
        "The left earbud stopped working within a week. Very disappointed.",
        "Battery life is great, lasting all day. Perfect for commuting or travel.",
        "The earbuds feel cheap and plasticky. Expected better at this price point."
    ],
    'rating': [5, 1, 5, 2, 4, 1, 5, 2, 5, 1, 5, 2, 4, 1, 5, 2, 5, 1, 5, 2]
}

df = pd.DataFrame(data)

# Preprocessing and Cleaning
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

# Adding Preprocessed Text to DataFrame
df['clean_review'] = df['review'].apply(preprocess_text)

# Sentiment Analysis using Transformers (DistilBERT)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits).item()

# Add Sentiment to DataFrame
df['sentiment'] = df['review'].apply(analyze_sentiment)

# Mapping Sentiment (0 - Negative, 1 - Positive) to More Readable Format
sentiment_map = {0: 'Negative', 1: 'Positive'}
df['sentiment'] = df['sentiment'].map(sentiment_map)

# Streamlit Sidebar Options
st.sidebar.header("Options")
view_data = st.sidebar.checkbox("View Raw Data")

# Display Raw Data if Selected
if view_data:
    st.subheader("Raw Reviews Data")
    st.write(df)

# Topic Modeling (Basic) using NMF
vectorizer = TfidfVectorizer(max_features=1000)
tfidf = vectorizer.fit_transform(df['clean_review'])
nmf = NMF(n_components=2).fit(tfidf)

def display_topics(model, feature_names, no_top_words):
    topic_dict = {}
    for topic_idx, topic in enumerate(model.components_):
        topic_dict[f"Topic {topic_idx}"] = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
    return topic_dict

feature_names = vectorizer.get_feature_names_out()
topics = display_topics(nmf, feature_names, 5)
st.subheader("Topics Identified in Reviews")
st.write(topics)

# WordCloud for Positive and Negative Sentiments
st.subheader("WordCloud for Sentiments")
positive_text = ' '.join(df[df['sentiment'] == 'Positive']['review'])
negative_text = ' '.join(df[df['sentiment'] == 'Negative']['review'])

st.write("### Positive Sentiment WordCloud")
positive_wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(positive_text)
st.image(positive_wordcloud.to_array())

st.write("### Negative Sentiment WordCloud")
negative_wordcloud = WordCloud(width=800, height=800, background_color='white', min_font_size=10).generate(negative_text)
st.image(negative_wordcloud.to_array())

# Sentiment Distribution Visualization
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(x='sentiment', data=df, ax=ax)
st.pyplot(fig)

# Average Rating Distribution Visualization
st.subheader("Average Rating Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x='rating', y='rating', data=df, estimator=lambda x: len(x) / len(df) * 100, ax=ax)
ax.set_ylabel('Percentage of Reviews')
st.pyplot(fig)

# Display Average Rating
st.subheader("Average Rating")
st.write(df['rating'].mean())
