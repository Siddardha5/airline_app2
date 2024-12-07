# Streamlit: Make sure to run this as a .py file using `streamlit run your_file.py`
import streamlit as st
import openai
import pandas as pd
import numpy as np
from collections import Counter
from transformers import pipeline
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
import os

# Set up DeepAI API key from Streamlit secrets
llm = ChatOpenAI(openai_api_key=st.secrets["MyOpenAIKey"], model_name="gpt-3.5-turbo")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

class ReviewAnalyzer:
    def __init__(self):
        # Initialize the LLM and sentiment analyzer
        self.client = llm

    def analyze_reviews(self, data):
        aligned_data = self.align_reviews_and_ratings(data)
        return self.handle_overall_reviews(aligned_data), self.handle_individual_reviews(aligned_data)

    def align_reviews_and_ratings(self, data):
        checked_data = pd.DataFrame(data)
        checked_data['sentiment'] = checked_data['reviews'].apply(lambda x: sentiment_analyzer(x)[0]['label'])
        checked_data['adjusted_rating'] = checked_data.apply(self.adjust_rating, axis=1)
        return checked_data

    def adjust_rating(self, row):
        sentiment = row['sentiment']
        rating = row['ratings']
        if sentiment == 'POSITIVE' and rating < 4:
            return 3
        elif sentiment == 'NEGATIVE' and rating > 2:
            return 3
        else:
            return rating

    def handle_overall_reviews(self, checked_data):
        avg_rating = np.mean(checked_data['adjusted_rating'])
        all_reviews = " ".join(checked_data['reviews'])
        summary_prompt = ChatPromptTemplate.from_template(
            "Summarize the following product reviews in about 100 words. "
            "If the average rating is less than 4 out of 5, include suggestions to improve. "
            "If the average rating is 4 or higher, identify what to continue keeping.\n\n"
            "Average Rating: {rating}\n"
            "Reviews: {reviews}\n\n"
            "Summary:"
        )
        summary_chain = summary_prompt | self.client
        summary = summary_chain.invoke({"rating": avg_rating, "reviews": all_reviews})
        return {
            "average_rating": avg_rating,
            "summary": summary.content.strip()
        }

    def handle_individual_reviews(self, checked_data):
        positive_reviews = checked_data[checked_data['sentiment'] == 'POSITIVE']['reviews'].tolist()
        negative_reviews = checked_data[checked_data['sentiment'] == 'NEGATIVE']['reviews'].tolist()
        return positive_reviews, negative_reviews

# Sample Reviews Dataset - (Product: earphones)
data = {
    'reviews': [
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
    'ratings': [5, 1, 5, 2, 4, 1, 5, 2, 5, 1, 5, 2, 4, 1, 5, 2, 5, 1, 5, 2]
}

# Streamlit interface
st.title("Review Analyzer")

if st.button("Analyze Reviews"):
    analyzer = ReviewAnalyzer()
    overall_summary, individual_reviews = analyzer.analyze_reviews(data)

    st.subheader("Overall Analysis")
    st.write(f"Overall Average Rating: {overall_summary['average_rating']:.2f}")
    st.write("Overall Summary of Reviews:")
    st.write(overall_summary['summary'])

    st.subheader("Positive Reviews")
    for review in individual_reviews[0]:
        st.write(review)

    st.subheader("Negative Reviews")
    for review in individual_reviews[1]:
        st.write(review)

    # Optional: Visualization of the overall ratings
    visualization_choice = st.selectbox("Would you like to visualize the ratings?", ["Yes", "No"])
    if visualization_choice == "Yes":
        rating_counts = pd.Series(data['ratings']).value_counts().sort_index()

        # Bar chart
        st.subheader("Distribution of Ratings")
        st.bar_chart(rating_counts)

        average_rating = np.mean(data['ratings'])
        st.write("Average Rating Distribution (in %):")
        st.write(rating_counts / len(data['ratings']) * 100)
        st.write("Average Rating:", average_rating)
