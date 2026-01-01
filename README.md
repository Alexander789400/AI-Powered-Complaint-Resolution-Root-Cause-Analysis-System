# AI-Powered-Complaint-Resolution-Root-Cause-Analysis-System

An end-to-end AI-driven NLP system designed to help organizations and enterprises automatically process, prioritize, and resolve large volumes of customer complaints efficiently.

This system acts as an internal AI decision-support tool for customer service and operations teams.

### Project Overview :

Organizations receive thousands of customer complaints daily across banking, finance, and service platforms.
Manual processing is slow, inconsistent, and costly.

This project uses Natural Language Processing (NLP) and Deep Learning to:

Classify complaints automatically

Detect sentiment & urgency

Identify root causes

Generate AI-based resolution replies

Store complaint history for analysis

All through an interactive web interface.

[Kaggle Dataset Link](https://www.kaggle.com/datasets/kennathalexanderroy/ai-powered-complaint-resolution/data)

[Dataset Link](https://www.consumerfinance.gov/data-research/consumer-complaints/#get-the-data)

### Key Features :

✅ Transformer-based Zero-Shot Classification

✅ LSTM Supervised Classification

✅ Transformer–LSTM Ensemble Decision

✅ Confidence Score Calculation

✅ Sentiment Analysis & Priority Detection

✅ Root Cause Identification

✅ LLM-Generated Resolution Replies

✅ SQLite Database for Complaint Storage

✅ Elegant Flask Web Dashboard

### AI & NLP Techniques Used

| Component                | Model / Technique                                 |
| ------------------------ | ------------------------------------------------- |
| Text Cleaning            | Regex-based preprocessing                         |
| Complaint Classification | LSTM (Supervised)                                 |
| Zero-Shot Classification | `facebook/bart-large-mnli`                        |
| Sentiment Analysis       | `distilbert-base-uncased-finetuned-sst-2-english` |
| Reply Generation         | LLM (Text Generation Pipeline)                    |
| Decision Strategy        | Transformer–LSTM Ensemble                         |
| Confidence Scoring       | Probability-based aggregation                     |

### Model Performance (Sample)

| Metric                 | Value       |
| ---------------------- | ----------- |
| Accuracy               | ~86%        |
| Precision              | 0.82        |
| Recall                 | 0.78        |
| Confidence Score Range | 0.70 – 0.95 |

### Web Interface :

Clean and modern UI

Displays:

Final complaint category

Confidence score

Sentiment & priority

Root cause

AI-generated resolution

Complaint history stored and retrievable via database

📦 Complaint-Resolution-AI

 ┣ 📂 models
 
 ┃ ┣ lstm_model.h5
 
 ┃ ┣ tokenizer.pkl
 
 ┃ ┗ label_encoder.pkl
 
 ┣ 📂 templates
 
 ┃ ┗ index.html
 
 ┣ 📂 static
 
 ┃ ┗ style.css
 
 ┣ app.py
 
 ┣ complaints.db
 
 ┗ README.md

### Tech Stack : 

Programming: Python

Framework: Flask

NLP: Transformers, LSTM

Deep Learning: TensorFlow / Keras

Models: Hugging Face

Database: SQLite

Frontend: HTML, CSS

Industry Use Case

<img width="1920" height="1080" alt="Screenshot 2025-12-30 100418" src="https://github.com/user-attachments/assets/698a1bf4-6164-43c2-bfec-7d09f433bc5c" />


### This system is ideal for:

Banks & Financial Institutions

E-commerce platforms

Telecom companies

Insurance providers

Customer support centers

It helps teams:

Reduce manual workload

Improve response time

Maintain complaint history

Make data-driven decisions

### Author :

Alexander Roy
AI & Data Science Enthusiast

[LinkedIn](https://www.linkedin.com/in/alexander-roy-570456191/)
