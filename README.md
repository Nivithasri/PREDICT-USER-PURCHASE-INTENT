
# Predicting User Purchase Intent  

## About the Project  
This project leverages a dataset from the paper **"Shopper Intent Prediction from Clickstream E-Commerce Data with Minimal Browsing Information"** to develop a system that predicts user purchase intent on a fashion e-commerce platform. Using clickstream data, the project employs feature engineering techniques like k-grams and graph motifs to enhance data representation and builds models including Logistic Regression, Markov Chain, and LSTM for intent prediction.  

---

## Dataset  
The dataset used in this project was obtained from [Shopper Intent Prediction - Public Data Release 1.0.0](https://github.com/coveooss/shopper-intent-prediction-nature-2020).  
It contains over **5.4 million events** grouped into sessions, representing user interactions with an e-commerce website. Key features include session IDs, event types, product actions, and timestamps, providing valuable information for predicting shopper intent.  

### Citation for Dataset  
 
```bibtex
@article{Requena2020,
author = {Requena, Borja and Cassani, Giovanni and Tagliabue, Jacopo and Greco, Ciro and Lacasa, Lucas},
title = {Shopper intent prediction from clickstream e-commerce data with minimal browsing information},
year = {2020},
journal = {Scientific Reports},
pages   = {2045-2322},
volume  = {10},
doi = {10.1038/s41598-020-73622-y}
}
```

---

## Key Features  
1. **Clickstream Data Analysis**:  
   - Analyzes user interactions to predict purchase intent.  

2. **Feature Engineering**:  
   - Techniques like k-grams and graph motifs create meaningful features for modeling.  

3. **Multiple Models**:  
   - Logistic Regression: 88% accuracy.  
   - Markov Chain: 85% accuracy.  
   - LSTM: 91% accuracy.  

4. **Interactive Dashboard**:  
   - Built with **Streamlit** for predictions, visualizations, and model insights.  

---

## How to Run  

### Prerequisites  
1. Install Python (3.8 or higher).  
2. Install required libraries using pip




## Acknowledgments  
This project was made possible using the dataset released by the authors of the paper *"Shopper Intent Prediction from Clickstream E-Commerce Data with Minimal Browsing Information."*  

The dataset is a product of collaboration between industry and academia, involving researchers from **Coveo AI Labs**, **Institut de Ciencies Fotoniques**, **Tilburg University**, and **Queen Mary University of London**. Special thanks to **Coveo AI Labs** for providing the dataset to the research community.  

---


## Technologies Used  
- **Python**  
- **Streamlit**  
- **Machine Learning Libraries**: TensorFlow, scikit-learn, pandas, NumPy  
- **Visualization Tools**: Matplotlib, Seaborn  

---

