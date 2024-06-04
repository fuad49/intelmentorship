import streamlit as st
import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from fpdf import FPDF
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer, AutoModel
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from summarizer import Summarizer
import os
import re

@st.cache_data
def parse_html_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            html_content = file.read()
            soup = BeautifulSoup(html_content, "html.parser")
            return soup
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

@st.cache_data
def scrape_amazon_product(url):
    global revList
    HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36', 'Accept-Language': 'en-US, en;q=0.5'}
    try:
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            with open("temp.html", 'wb') as file:
                file.write(response.content)
        else:
            print(f"Failed to download HTML. Status code: {response.status_code}")
    except Exception as e:
        print(f"An error occurred: {e}")

    current_directory = os.getcwd()
    file_name = "temp.html"
    file_path = os.path.join(current_directory, file_name)
    global global_file_path
    global_file_path = file_path

    soup = parse_html_file(file_path)

    product_name_element = soup.find('span', {'id': 'productTitle'})
    product_name = product_name_element.text.strip() if product_name_element else None

    categories = soup.find_all('a', {'class': 'a-link-normal a-color-tertiary'})
    category = categories[-1].text.strip() if categories else None

    product_description_element = soup.find('div', {'id': 'productDescription'})
    product_description = product_description_element.text.strip() if product_description_element else None

    ratings_element = soup.find('span', {'class': 'a-icon-alt'})
    ratings = ratings_element.text.strip() if ratings_element else None

    reviews = []
    review_elements = soup.find_all('div', {'class': 'a-section review aok-relative'})
    for review_element in review_elements:
        review_text = review_element.find('span', {'data-hook': 'review-body'}).text.strip()
        
        reviews.append(review_text)  # Add a space after each review

    prodata = {
        'product_name': product_name,
        'Category': category,
        'product_description': product_description,
        'Reviews': reviews,
        'Ratings': ratings
    }
    df = pd.DataFrame(prodata)

    df.to_csv("Pro.csv", index=False)

    return prodata

summarizer = Summarizer()
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")

def chunk_text(text, max_chunk_size=512):
    chunks = []
    words = text.split()
    current_chunk = ""
    for word in words:
        if len(current_chunk) + len(word) <= max_chunk_size:
            current_chunk += word + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = word + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def summarize_single_review(review):
    sentiment_analysis = pipeline("sentiment-analysis", model="bhadresh-savani/distilbert-base-uncased-sentiment-sst2")
    sentiment_labels = [sentiment_analysis(chunk)[0]['label'] for chunk in review]

    if any(label == 'POSITIVE' for label in sentiment_labels):
        concatenated_review = ' '.join(review)
        inputs = tokenizer(concatenated_review, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        summary = summarizer(concatenated_review, min_length=50, max_length=150)
        recommendation = ""
    else:
        concatenated_review = ' '.join(review)
        inputs = tokenizer(concatenated_review, return_tensors="pt", max_length=512, truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state
        summary = summarizer(concatenated_review, min_length=50, max_length=150)
        recommendation = ""

    return summary, recommendation


def parallelize_summarization_async(reviews, num_cores):
    results = []
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for review in reviews:
            review_chunks = chunk_text(review, max_chunk_size=512)
            future = executor.submit(summarize_single_review, review_chunks)
            futures.append(future)
        for future in tqdm(futures, total=len(futures)):
            summary, recommendation = future.result()
            results.append((summary, recommendation))
    return results

@st.cache_data
def CalcReviews(reviews):
    model_name = "bhadresh-savani/distilbert-base-uncased-sentiment-sst2"
    output_file = "mainResult.csv"

    classifier = pipeline("sentiment-analysis", model=model_name)

    positive_reviews = []
    negative_reviews = []

    for review in reviews:
        all_predictions = classifier(review)
        for prediction in all_predictions:
            if prediction['label'] == 'POSITIVE':
                positive_reviews.append(review)
            else:
                negative_reviews.append(review)

    num_positive = len(positive_reviews)
    num_negative = len(negative_reviews)
    ratio = num_positive / num_negative if num_negative != 0 else 0
    summaryPos = parallelize_summarization_async(positive_reviews, 4)
    summaryNeg = parallelize_summarization_async(negative_reviews, 4)

    data = {
        'positive_reviews': [num_positive],
        'negative_reviews': [num_negative],
        'Ratio of Positive to Negative Reviews': [ratio],
        'positive_summary': ['\n'.join(map(str, summaryPos))],
        'negative_summary': ['\n'.join(map(str, summaryNeg))]
    }
    df = pd.DataFrame(data)

    df.to_csv("Rev.csv", index=False)
    return data

@st.cache_data
# Function to generate PDF report
def generate_pdf(product_data, review_data):
    pdf = FPDF()

  # Add a page
    pdf.add_page()
    pdf.set_font("Arial", size=12)


    csv_file1 = "Rev.csv"  # Replace with the path to your CSV file
    df1 = pd.read_csv(csv_file1)


    context = ""
    for column in ['positive_reviews', 'negative_reviews', 'Ratio of Positive to Negative Reviews', 'positive_summary', 'negative_summary']:
      context += f"{column}: {df1.iloc[0][column]}\n"

    csv_file2 = "Pro.csv"
    df2 = pd.read_csv(csv_file2)

    for column in ['product_name', 'Category', 'product_description', 'Reviews', 'Ratings']:
      context += f"{column}: {df2.iloc[0][column]}\n"

    cleaned_string = re.sub(r'[^a-zA-Z0-9\s.:]', '', context)
    pdf.multi_cell(0, 10, cleaned_string)

    pdf_path = "output.pdf"
    pdf.output(pdf_path)
    return pdf_path

# Function to interact with ChatPDF API
def get_answer(question, file_path):
    files = [
        ('file', ('file', open(file_path, 'rb'), 'application/octet-stream'))
    ]
    headers = {
        'x-api-key': "sec_tq3SOgqLfwOlsWcRP8eATcxzGinyICwK",  # Replace with your actual ChatPDF API key
    }

    response1 = requests.post(
        'https://api.chatpdf.com/v1/sources/add-file', headers=headers, files=files)

    if response1.status_code == 200:
        source_id = response1.json()['sourceId']
    else:
        st.error("Failed to upload PDF to ChatPDF.")
        return None

    data = {
        'sourceId': source_id,
        'messages': [
            {
                'role': "user",
                'content': question,
            }
        ]
    }

    response = requests.post(
        'https://api.chatpdf.com/v1/chats/message', headers=headers, json=data)

    if response.status_code == 200:
        return response.json()['content']
    else:
        st.error("Failed to get response from ChatPDF.")
        return None

# Streamlit application
st.title("Amazon Product Insights Dashboard")

# URL input
url = st.text_input("Enter Amazon Product URL:")

if url:
    product_data = scrape_amazon_product(url)

    if product_data:
        st.header(product_data['product_name'])
        st.subheader("Product Description")
        st.write(product_data['product_description'])

        st.subheader("Reviews")
        st.write(product_data['Reviews'])
        review_data = CalcReviews(product_data['Reviews'])

        st.metric("Number of Positive Reviews", ' '.join(map(str,review_data['positive_reviews'])))
        st.metric("Number of Negative Reviews", ' '.join(map(str,review_data['negative_reviews'])))
        st.metric("Positive to Negative Ratio", ' '.join(map(str,review_data['Ratio of Positive to Negative Reviews'])))

        st.subheader("Summary of Positive Reviews")
        st.write(review_data['positive_summary'])

        st.subheader("Summary of Negative Reviews")
        st.write(review_data['negative_summary'])


        # Generate PDF
        pdf_path = generate_pdf(product_data, review_data)

        # Chatbot interaction
        st.subheader("Chat with the Product")
        user_question = st.text_input("Ask a question about the product:")

        if user_question:
            response = get_answer(user_question, pdf_path)
            st.write(response)
