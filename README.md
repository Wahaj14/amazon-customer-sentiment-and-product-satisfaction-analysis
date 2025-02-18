# Customer Sentiment and Product Satisfaction Analysis on Amazon

# Project Overview

This project analyzes customer sentiment in Amazon fine food reviews using Natural Language Processing (NLP) and Machine Learning (ML) techniques. The goal is to classify sentiment, extract aspect-based insights (quality, price, packaging), track sentiment trends over time, and build predictive models for forecasting future customer opinions.

# Key Features

Sentiment Classification: Uses BERT, LSTM, SVM, Random Forest models to classify reviews as positive, neutral, or negative.

Aspect-Based Sentiment Analysis (ABSA): Extracts insights on quality, price, and packaging from customer feedback.

Temporal Sentiment Analysis: Tracks fluctuations in sentiment over time to detect seasonal trends.

Predictive Modeling: Utilizes LSTM and ARIMA models to forecast future sentiment trends.

Data Visualization: Uses Matplotlib & Seaborn to generate insightful graphs for sentiment trends.

# Tech Stack

Programming Language: Python 

Libraries & Frameworks: TensorFlow, Scikit-Learn, Pandas, NLTK, spaCy, Matplotlib, Seaborn

Machine Learning Models: Naïve Bayes, SVM, Random Forest, LSTM, CNN, BERT

NLP Techniques: Tokenization, Lemmatization, Stopword Removal, Dependency Parsing

Predictive Modeling: LSTM, ARIMA

# Results & Impact

Achieved 91.7% accuracy with BERT, outperforming traditional ML models.

Quality had the highest positive sentiment (85%), while price was the most polarizing factor (55% positive, 45% negative).

Temporal analysis revealed spikes in negative reviews during holidays due to delivery delays.

Predictive modeling helped businesses adjust pricing, packaging, and quality strategies proactively.

# Project Structure

 Amazon-Sentiment-Analysis
 
  • data               # Dataset files
  
  • notebooks          # Jupyter Notebooks for analysis
  
  • models             # Trained ML/DL models
  
  • requirements    # Dependencies
  
  • README.md           # Project documentation
  
  • amazo_customer_sentiment_analysis.py  # Main script

# Getting Started

1 Install Dependencies

pip install -r requirements.txt

2 Run Sentiment Analysis

python amazo_customer_sentiment_analysis.py

3 View Results

Check the output folder for visualizations and sentiment reports.

# Future Improvements

Enhance aspect-based sentiment analysis using more advanced transformers.

Expand dataset to include more product categories.

Deploy as a real-time sentiment analysis dashboard.

# License

This project is open-source and available under the MIT License.
