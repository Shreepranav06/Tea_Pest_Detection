import torch
from flask import Flask, request, jsonify, render_template
from PIL import Image
import torchvision.transforms as transforms
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

app = Flask(__name__)

# Define the device (GPU or CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the TorchScript model
model = torch.jit.load(os.path.join(os.path.dirname(__file__), 'model/final_model.pt'), map_location=device)
class_names = ['healthy', 'gray blight', 'red_spot', 'helopeltis', 'blister blight', 'algal_spot']

# Define transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    img = Image.open(file).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)  # Move input tensor to device

    # Run inference
    with torch.no_grad():
        output = model(img)
        _, predicted = output.max(1)
        predicted_class_idx = predicted.item()
        predicted_class_name = class_names[predicted_class_idx]

    return jsonify({'class': predicted_class_name})

# Chatbot Setup
os.environ["HUGGINGFACEHUB_API_TOKEN"] = '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'
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
pdf_path = r"C:\Users\91978\Downloads\flina.pdf" # Make sure the path to your PDF is correct

def extract_text_from_pdf(pdf_file):
    raw_text = ''
    for page in pdf_file.pages:
        content = page.extract_text()
        if content:
            raw_text += content
    return raw_text

pdf_reader = PdfReader(pdf_path)
raw_text = extract_text_from_pdf(pdf_reader)
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings()
document_search = FAISS.from_texts(texts, embeddings)

template = """You are an AI assistant. Answer the question based on the given documents and conversation history.
{chat_history}
Question: {question}
Answer: Provide only the most relevant and concise answer within 20 words."""
CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(template)

llm = HuggingFaceHub(repo_id='meta-llama/Meta-Llama-3-8B-Instruct', model_kwargs={"max_tokens": 150})

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)#

qa_chain = ConversationalRetrievalChain.from_llm(
    retriever=document_search.as_retriever(search_kwargs={"k": 2}),
    llm=llm,
    memory=memory,
    condense_question_prompt=CUSTOM_QUESTION_PROMPT,
)

def get_response(prompt):
    response = qa_chain.run(question=prompt)
    output = "".join(response)
    return output

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    response = get_response(user_input)
    split_text = response.split("Answer:")

    # Check if the split results in more than one part
    if len(split_text) > 1:
        # The desired part is the second part (index 1)
        response = split_text[1].strip()  # Remove any leading/trailing whitespace
    else:
        response = "Sorry, I couldn't find an answer."
    
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
