import warnings
warnings.filterwarnings("ignore")
import torch
from PIL import Image, ImageEnhance
import pytesseract
import yolov5
from tkinter import Tk, filedialog
import gdown
import re


# refer to this https://discuss.streamlit.io/t/tesseract-is-not-installed-or-its-not-in-my-path/16039
# change this to tesseract path of streamlit
# if deploying on another instance, do
# apt-get install tesseract-ocr imagemagick libtesseract-dev

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Chatbot Dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')



from py2neo import Graph
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
import numpy as np
from py2neo import Graph
import os
from nltk.tokenize import word_tokenize

# Download NLTK resources
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')


# Initialize Chatbot
class Chatbot_Response:
    def __init__(self):
        try:
            self.g = Graph("neo4j+ssc://8d15dfb6.databases.neo4j.io", user="neo4j", password="VRXTwfUV8JXyZAvQjpsKVux5Ve_L5FWH0_UX7BnRbVA")
        except Exception as e:
            print(f"Error connecting to Graph: {e}")
            raise

        model_size = "small"
        current_directory = os.getcwd()
        model_path = os.path.join(current_directory, f'ChatBotProject/output/output-{model_size}')
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(f'microsoft/DialoGPT-{model_size}')

        #self.model = AutoModelForCausalLM.from_pretrained(f'/Users/iqrabismi/Desktop/Chatbot_project/ChatBotProject/output/output-{model_size}')
    
    def query_graph(self, query):
        return self.g.run(query).data()

    def find_most_similar(self, user_question, operation_names):
        vectorizer = TfidfVectorizer()
        docs = [user_question] + operation_names
        tfidf_matrix = vectorizer.fit_transform(docs)
        similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
        most_similar_index = np.argmax(similarity_matrix)
        return operation_names[most_similar_index]
    
    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Remove symbols
        text = text.replace('\n', ' ').strip()  # Remove escape characters and strip spaces
        stopwords = set([
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll",
            "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's",
            'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
            'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was',
            'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
            'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with',
            'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
            'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once'
        ])
        text = ' '.join([word for word in text.split() if word.lower() not in stopwords])  # Remove stopwords
        return text
    
    def check_keywords_in_graph(self, detected_text):
        cleaned_text = self.clean_text(detected_text)
        keywords = cleaned_text.split() 
        print(keywords)
        for keyword in keywords:
            # Construct Cypher query to search for the keyword in the graph
            query = f"MATCH (n) WHERE toLower(n.name) CONTAINS toLower('{keyword}') RETURN n"
            result = self.g.run(query).data()
            print(result, keyword)
            if result:
                return True  # Return True if keyword found in graph
        return False  # Return False if no keyword found in graph

    def answer_question_with_image(self, user_question, image1):
        detected_text = yolo_and_ocr(image1)
        is_keyword_found = self.check_keywords_in_graph(detected_text)
        if not is_keyword_found:
            print('Uploaded image is not relevant to AI testing tool. Please upload relevant image')
            return
        user_question = f"{user_question} {detected_text}"
        print("Updated Question:", user_question)

        temp =  self.answer_question(user_question)
        print(temp)
        return temp

    def answer_question(self, user_question):
        # Extract all operation names from the KG
        all_operations_query = "MATCH (n:Operation) RETURN n.name as name"
        all_operations_data = self.query_graph(all_operations_query)
        all_operation_names = [item['name'] for item in all_operations_data]

        # Find the most similar operation to the user's question
        most_similar_operation = self.find_most_similar(user_question, all_operation_names)

        # Query the KG to get the method for the most similar operation
        method_query = f"MATCH (n:Operation {{name: '{most_similar_operation}'}}) RETURN n.method as method"
        method_data = self.query_graph(method_query)


        user_question_tokens = word_tokenize(user_question.lower())
        answer_tokens = word_tokenize(method_data[0]['method'][0].lower())

        # Extract keywords from the question (nouns and verbs)
        question_keywords = [word for (word, pos) in nltk.pos_tag(user_question_tokens) if pos in ['NN', 'NNS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']]

        # Calculate keyword overlap using set intersection
        keyword_overlap = set(question_keywords).intersection(set(answer_tokens))

        # Define a relevance threshold (e.g., 0.3)
        relevance_threshold = 0.3

        # Calculate relevance score (simple overlap-based)
        relevance_score = len(keyword_overlap) / len(question_keywords)

        # Check if the answer is relevant based on the threshold
        if relevance_score >= relevance_threshold:
            return method_data[0]['method'][0]

        else:

            # retrieving answer from pre-trained transformer

            #Embedding the Question
            new_user_input_ids = self.tokenizer.encode(user_question + self.tokenizer.eos_token, return_tensors='pt')

            bot_input_ids = torch.cat([new_user_input_ids], dim=-1)

            predicted_answers=[]

            # fetching answer from pre-trained model
            chat_history_ids = self.model.generate(bot_input_ids, max_length=1000,pad_token_id= self.tokenizer.eos_token_id,  no_repeat_ngram_size=3,do_sample=True, top_k=100, top_p=0.7,temperature = 0.8)

            predicted_answers.append(self.tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))

            return predicted_answers[0]


# Initialize YOLO
try:
    model_path = r'best.pt'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'{model_path} not found')
    model = yolov5.load(model_path)
except Exception as e:
    # Download Model if not found 
    url = 'https://drive.google.com/uc?id=1tLJVvg9lo-3otKuTUm_orGxqMTzA8eSz'
    output = model_path
    print("Downloading YOLO model...")
    gdown.download(url, output, quiet=False)

    # Load model after downloading
    try:
        model = yolov5.load(output)
        print("Model loaded successfully after downloading.")
    except Exception as e:
        print(f"Error loading YOLO model after downloading: {e}")

def yolo_and_ocr(image_path):
    im = Image.open(image_path)
    results = model(im)
    text_list = []
    enhancer = ImageEnhance.Contrast(im)
    im = enhancer.enhance(2)
    im = im.convert('L')

    for coordinates in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, coordinates[:4])
        im1 = im.crop((x1, y1, x2, y2))
        rawData = pytesseract.image_to_string(im1)
        text_list.append(rawData)

    print(text_list)

    return ' '.join(text_list)

# Function to open file dialog and return selected file path
def upload_image():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename()
    return file_path


# Main Program
if __name__ == '__main__':
    try:
        chatbot = Chatbot_Response()
    except Exception as e:
        print(f"Error initializing Chatbot: {e}")
        exit(1)

    while True:
        try:
            user_question = input("Please enter your question: ")
            
            if user_question.lower() == "bye":
                print("Chatbot: Goodbye!")
                break

            upload_image_choice = input("Do you want to upload an image? (y/n): ").lower()
            if upload_image_choice == 'y':
                print("Please upload your image.")
                user_image = upload_image()  # Get the uploaded image path

                if not user_image:  # If the user canceled the upload
                    print("No image selected.")
                    continue
                
                detected_text = yolo_and_ocr(user_image)
                is_keyword_found = chatbot.check_keywords_in_graph(detected_text)
                if not is_keyword_found:
                    print('Uploaded image is not relevant to AI testing tool. Please upload relevant image')
                    continue
                user_question = f"{user_question} {detected_text}"
                print("Updated Question:", user_question)
                
            # Replace with the actual function to get the chatbot response
            answer = chatbot.answer_question(user_question)
            print(f"Chatbot: {answer}")

        except Exception as e:
            print(f"An error occurred: {e}")