# AI Document Summarizer and Quiz Generator

An AI-based document summarization and quiz generation tool developed for university students.

## Project Overview

This project provides a Streamlit-based application that helps students understand and revise academic documents more efficiently. Users can upload an academic PDF and generate a concise summary, key revision points, and multiple-choice quiz questions.

## Features

- Upload academic PDF documents
- Extract text from PDF files
- Generate document summaries using BART Large CNN
- Extract key revision points using FLAN-T5
- Generate multiple-choice quiz questions
- View document statistics including pages, words and characters
- Download the generated summary, key points and quiz

## Technologies Used

- Python
- Streamlit
- PyPDF2
- Hugging Face Transformers
- BART Large CNN
- FLAN-T5 Large
- PyTorch

## System Workflow

```text
PDF Upload
    ↓
PDF Text Extraction
    ↓
Text Processing
    ↓
BART Large CNN
    ↓
Summary Generation
    ↓
FLAN-T5
    ↓
Key Points and Quiz Generation
    ↓
Results Displayed in Streamlit
