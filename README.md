# NLP - Stock Sentiment Analysis using News Headlines

## 📌 Project Overview
This project analyzes stock market sentiment using **news headlines** and predicts stock trends based on sentiment scores. The analysis is performed using **Natural Language Processing (NLP)** and a **Random Forest** classifier.

## 📂 Dataset Details
- The dataset consists of stock market news headlines.
- Each row represents **multiple headlines for a single trading day**.
- The stock market movement is used as the target variable.

## 🔬 Approach & Methodology
1. **Data Preprocessing:**
   - Removing missing values and cleaning text data.
   - Converting textual data into numerical representations using **CountVectorizer**.

2. **Sentiment Analysis:**
   - Extracting features from text data.
   - Using **CountVectorizer** to convert news headlines into numerical format.

3. **Model Selection & Training:**
   - Applied **Random Forest Classifier** for stock trend prediction.
   - Evaluated model performance using **accuracy score, confusion matrix, and classification report**.

## 🛠 How to Run the Project
### Prerequisites
Ensure you have Python and the required libraries installed:
```bash
pip install pandas numpy scikit-learn nltk
```

### Steps to Run
1. Load the dataset (`Data.csv`).
2. Run the Jupyter Notebook step by step.
3. Train the **Random Forest** model.
4. Evaluate predictions and analyze stock market trends based on sentiment.

## 📊 Results & Insights
- The **Random Forest model** predicts stock market trends based on news sentiment.
- **CountVectorizer** is effective in feature extraction for classification tasks.
- Sentiment analysis can provide **valuable insights** into stock market movements.
- The model achieves an 84.1% accuracy, with a higher recall for predicting stock increases (93%) than stock declines (75%). This suggests the model is slightly biased towards predicting stock increases, which can be fine-tuned by adjusting class weights or balancing the dataset.

## 🔮 Future Enhancements
- Experiment with **TF-IDF** and **word embeddings** for better text representation.
- Use **real-time news sources** for live sentiment analysis.
- Implement **deep learning models (LSTMs, Transformers)** for improved accuracy.

## 📎 Repository Link
[GitHub Repository](https://github.com/Gowthamipriya1307/NLP)


