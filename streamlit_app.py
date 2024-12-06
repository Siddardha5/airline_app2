import streamlit as st
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Streamlit App
st.title("Sentiment Analysis and Topic Modeling for Product Reviews")

# Dataset
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

# Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(preprocess_text)

# Sentiment Analysis
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    return torch.argmax(logits).item()

df['sentiment'] = df['review'].apply(analyze_sentiment)
sentiment_map = {0: 'Negative', 1: 'Positive'}
df['sentiment'] = df['sentiment'].map(sentiment_map)

# Topic Modeling
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

# Display Topics
st.subheader("Identified Topics")
for topic, words in topics.items():
    st.write(f"{topic}: {', '.join(words)}")

# WordCloud for Sentiment
st.subheader("Word Clouds")
positive_text = ' '.join(df[df['sentiment'] == 'Positive']['review'])
negative_text = ' '.join(df[df['sentiment'] == 'Negative']['review'])

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Positive Sentiment**")
    wordcloud_pos = WordCloud(width=800, height=800, background_color='white').generate(positive_text)
    st.image(wordcloud_pos.to_array(), use_column_width=True)
with col2:
    st.markdown("**Negative Sentiment**")
    wordcloud_neg = WordCloud(width=800, height=800, background_color='white').generate(negative_text)
    st.image(wordcloud_neg.to_array(), use_column_width=True)

# Sentiment Distribution
st.subheader("Sentiment Distribution")
fig_sentiment = plt.figure(figsize=(8, 6))
sns.countplot(x='sentiment', data=df)
st.pyplot(fig_sentiment)

# Average Rating Distribution
st.subheader("Average Rating Distribution")
fig_rating = plt.figure(figsize=(8, 6))
sns.barplot(x='rating', y='rating', data=df, estimator=lambda x: len(x) / len(df) * 100)
st.pyplot(fig_rating)

# Summary and Takeaways
average_rating = df['rating'].mean()
st.subheader("Overall Summary")
st.write(f"**Average Rating:** {average_rating:.2f}")
if average_rating < 4:
    st.write("**Takeaways:** Consider improving build quality and addressing common customer complaints.")
else:
    st.write("**Takeaways:** Customers are generally satisfied. Focus on maintaining quality and innovation.")
