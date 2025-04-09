# Customer Sentiment and Product Satisfaction Analysis on Amazon

# Project Overview

This project analyzes customer sentiment in Amazon fine food reviews using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The goal is to classify sentiment, extract aspect-based insights (quality, price, packaging), track sentiment trends over time, and build predictive models for forecasting future customer opinions.

# Key Features

Sentiment Classification: Uses BERT, VADER, Naive Bayes, SVM, Random Forest and LSTM models to classify reviews as positive, neutral, or negative.

Aspect-Based Sentiment Analysis (ABSA): Extracts insights on quality, price, and packaging from customer feedback.

Temporal Sentiment Analysis: Tracks fluctuations in sentiment over time to detect seasonal trends.

Predictive Modeling: Utilizes LSTM model to forecast future sentiment trends.

Data Visualization: Uses Matplotlib & Seaborn to generate insightful graphs for sentiment trends.

# Tech Stack

Programming Language: Python 

Libraries & Frameworks: TensorFlow, Scikit-Learn, Pandas, NLTK, spaCy, Matplotlib, Seaborn

Machine Learning Models: Naïve Bayes, SVM, Random Forest, LSTM,  BERT

NLP Techniques: Tokenization, Lemmatization, Stopword Removal, Dependency Parsing

Predictive Modeling: LSTM

# Results & Impact

Achieved 80% accuracy with BERT, outperforming traditional ML models.

Quality had the highest negative sentiment (70%), while price was the most polarizing factor (55% positive, 45% negative).

Temporal analysis revealed spikes in negative reviews during holidays due to delivery delays.

Predictive modeling helped businesses adjust pricing, packaging, and quality strategies proactively.

# Project Structure

 Amazon-Sentiment-Analysis
 
  • data               # Dataset files
  
  • notebooks          # Jupyter Notebooks for analysis
  
  • models             # Trained ML/DL models
  
  • requirements    # Dependencies
  
  • README.md           # Project documentation
  
  • Amazon Customer Sentiment Analysis.py  # Main script

# Getting Started

1 Install Dependencies

pip install -r requirements

2 Run Sentiment Analysis

python Amazon Customer Sentiment Analysis.py

3 View Results


# Future Improvements

Enhance aspect-based sentiment analysis using more advanced transformers.

Expand dataset to include more product categories.

Deploy as a real-time sentiment analysis dashboard.

# License

This project is open-source and available under the MIT License.
