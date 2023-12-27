# Resume Radar - Powered by Fetch.ai

## Introduction
Resume Radar is an innovative tool developed for the IIT Bombay HACKAI hackathon, sponsored by Fetch.ai. It uses Fetch.ai agents and a pre-trained BERT model from Hugging Face for analyzing resumes and job descriptions, providing a detailed evaluation of candidate skills.

## Key Features
- *Fetch.ai Integration*: Leverages Fetch.ai agents for REST API requests.
- *BERT Model Analysis*: Utilizes a BERT model for text processing and similarity calculations.
- *Skill Categorization*: Classifies skills into Project Management, Backend, Frontend, Data Science, and DevOps.
- *Visual Skill Representation*: Offers a pie chart for visualizing skill distribution.
- *Skill Gap Analysis*: Identifies missing skills in a resume compared to a job description.
- *Similarity Scoring*: Uses BERT model to score the similarity between a resume and job description.

## Technology Stack
- Flask, PyPDF2, NLTK, Transformers, Pandas, Matplotlib.
- Fetch.ai agents for backend operations.
- BERT tokenizer and model for text analysis.

## Requirements
- Python 3.7 or later.

## Installation
Clone the repository and set up the Flask environment:
```bash
$ cd resume-screening-flask
$ python3 -m venv venv
$ source ./venv/bin/activate
$ pip install -r requirements.txt
$ flask run
```

## Usage
- *Upload Files*: Upload resumes and job descriptions in PDF format. Files are stored in the uploads folder.
- *Analysis*: The application processes and analyzes the text from the uploaded files.
- *Visualization*: View the skill distribution in a pie chart.
- *Skill Gap and Recommendations*: Identify missing skills and get recommendations for improvement.

## Security and Privacy
- The uploads folder is hidden and private, ensuring data confidentiality.
- A unique private key is generated for each Fetch.ai agent, enhancing security and integrity.

## Conclusion
Resume Radar demonstrates the effective use of AI and NLP in recruitment, backed by Fetch.ai's robust and scalable technology.

This README provides a comprehensive guide to the Resume Radar application. It includes an introduction, key features, the technology stack used, requirements, installation instructions, usage details, and notes on security and privacy. The Markdown format is used for clarity and organization, ensuring that the document is both informative and easy to navigate.

## Screenshots

### Upload Page
![Output 1](/static/img/output_6.png)

### Skill Wise pie chart
![Output 2](/static/img/output_7.png)

### Similarity with job description
![Output 3](/static/img/output_9.png)

### Recommended skills
![Output 3](/static/img/output_8.png)
