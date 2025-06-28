# Twitter Tweet Sentiment Analysis

A machine learning project that analyzes the sentiment of tweets using Naive Bayes classification with natural language processing techniques.

## Overview

This project implements sentiment analysis on Twitter data using the Naive Bayes algorithm. The system classifies tweets as positive, negative, or neutral based on their textual content.

## Features

- Tweet preprocessing and cleaning
- Text vectorization using TF-IDF
- Naive Bayes classification
- Model evaluation and performance metrics
- Visualization of results

## Naive Bayes Classification Theory

### What is Naive Bayes?

Naive Bayes is a probabilistic machine learning algorithm based on Bayes' theorem. It's particularly effective for text classification tasks like sentiment analysis.

**Bayes' Theorem:**
```
P(A|B) = P(B|A) * P(A) / P(B)
```

For sentiment analysis:
```
P(sentiment|tweet) = P(tweet|sentiment) * P(sentiment) / P(tweet)
```

### Why "Naive"?

The algorithm assumes that features (words) are conditionally independent given the class label. While this assumption is often violated in real-world text data, Naive Bayes still performs remarkably well for text classification.

### Types of Naive Bayes

1. **Multinomial Naive Bayes**: Best for text classification with word counts
2. **Gaussian Naive Bayes**: For continuous features
3. **Bernoulli Naive Bayes**: For binary features

## Text Preprocessing Pipeline

### 1. Data Cleaning
- Remove URLs, mentions (@username), and hashtags
- Convert text to lowercase
- Remove punctuation and special characters
- Handle contractions (e.g., "don't" → "do not")

### 2. Text Normalization
- **Tokenization**: Split text into individual words
- **Stop Words Removal**: Remove common words (the, and, or, etc.)
- **Stemming/Lemmatization**: Reduce words to their root form
  - Stemming: "running" → "run"
  - Lemmatization: "better" → "good"

### 3. Feature Engineering
- Remove very rare words (appear in <5 documents)
- Remove very common words (appear in >95% of documents)
- Handle negations properly

## Vectorization Techniques

### TF-IDF (Term Frequency-Inverse Document Frequency)

**Term Frequency (TF):**
```
TF(t,d) = (Number of times term t appears in document d) / (Total number of terms in document d)
```

**Inverse Document Frequency (IDF):**
```
IDF(t) = log(Total number of documents / Number of documents containing term t)
```

**TF-IDF Score:**
```
TF-IDF(t,d) = TF(t,d) * IDF(t)
```

### Why TF-IDF?

1. **Reduces impact of common words**: Words appearing in many documents get lower weights
2. **Emphasizes unique words**: Rare words that are informative get higher weights
3. **Normalized representation**: Accounts for document length differences

### Alternative Vectorization Methods

- **Bag of Words (BoW)**: Simple word count representation
- **Word2Vec**: Dense vector representations capturing semantic meaning
- **Doc2Vec**: Document-level embeddings

## Model Training Process

1. **Data Split**: 80% training, 20% testing
2. **Vectorization**: Convert text to numerical features using TF-IDF
3. **Model Training**: Train Multinomial Naive Bayes on training data
4. **Hyperparameter Tuning**: Optimize alpha (smoothing parameter)
5. **Evaluation**: Test on unseen data

## Performance Metrics

- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

## Installation

```bash
# Clone the repository
git clone https://github.com/hassanirfanx21/Twitter-Tweet-Sentiment-Analysis.git

# Navigate to project directory
cd Twitter-Tweet-Sentiment-Analysis

# Install required packages
pip install -r requirements.txt
```

## Usage

1. Open the Jupyter notebook:
```bash
jupyter notebook twitter_sentiment_analysis.ipynb
```

2. Run all cells to:
   - Load and preprocess the dataset
   - Train the Naive Bayes model
   - Evaluate performance
   - Visualize results

## Dependencies

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- nltk
- wordcloud
- jupyter

## Dataset

The project uses a Twitter sentiment dataset containing labeled tweets. The dataset includes:
- Tweet text
- Sentiment labels (positive, negative, neutral)
- User information (optional)

## Results

The Naive Bayes classifier achieves:
- Accuracy: ~85%
- Precision: ~83%
- Recall: ~82%
- F1-Score: ~82%

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Twitter API for data access
- scikit-learn for machine learning tools
- NLTK for natural language processing
- The open-source community for inspiration and tools

## Contact

Hassan Irfan - [GitHub](https://github.com/hassanirfanx21)

Project Link: [https://github.com/hassanirfanx21/Twitter-Tweet-Sentiment-Analysis](https://github.com/hassanirfanx21/Twitter-Tweet-Sentiment-Analysis)
