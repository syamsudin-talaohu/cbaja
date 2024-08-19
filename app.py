from flask import Flask, render_template, request, jsonify
from gtts import gTTS
import base64
import os
import time
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import speech_recognition as sr
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

app = Flask(__name__)

# Akses kunci API dari variabel lingkungan
gemini_api_key = os.getenv('GOOGLE_API_KEY')

# Load PDF data
loader = PyPDFLoader("unamin22.pdf")
data = loader.load()

# Split data into smaller chunks for faster processing
text_splitter = RecursiveCharacterTextSplitter(chunk_size=600)  # Reduced chunk size
docs = text_splitter.split_documents(data)

# Create vectorstore and retriever
vectorstore = Chroma.from_documents(documents=docs, embedding=GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})  # Reduced k value

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0, max_tokens=300, timeout=3)  # Reduced max_tokens and added timeout

# System prompt
system_prompt = (
    "You are a virtual assistant for the admission process at Universitas Muhammadiyah Sorong (UNAMIN). "
    "Your role is to provide accurate and concise information regarding the admission process, registration steps, "
    "required documents, important dates, and contact details. Respond as if you are the official representative "
    "of the PMB UNAMIN. If you do not know the answer, politely say that you do not have the information."
    "\n\n"
    "{context}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Cache dictionary
cache = {}

def text_to_speech(text):
    tts = gTTS(text, lang='id')
    tts.save("response.mp3")
    with open("response.mp3", "rb") as audio_file:
        audio_bytes = audio_file.read()

    # Create a base64-encoded version of the audio file for HTML use
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
        </audio>
    """
    return audio_html

def speech_to_text():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Silakan bicara sekarang...")
        audio = r.listen(source)
        try:
            query = r.recognize_google(audio, language="id-ID")
            print(f"Anda berkata: {query}")
            return query
        except sr.UnknownValueError:
            print("Maaf, saya tidak mengerti apa yang Anda katakan.")
        except sr.RequestError:
            print("Error pada layanan Speech Recognition.")
    return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("input")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    # Check if response is cached
    if user_input in cache:
        ai_response = cache[user_input]
    else:
        # Generate AI response
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        response = rag_chain.invoke({"input": user_input})
        ai_response = response["answer"]
        # Cache the response
        cache[user_input] = ai_response

    clean_response = ai_response.replace("**", "")

    # Use a thread pool executor to handle TTS processing in parallel
    with ThreadPoolExecutor() as executor:
        audio_html = executor.submit(text_to_speech, clean_response).result()

    return jsonify({"response": ai_response, "audio": audio_html})

@app.route("/speech", methods=["POST"])
def speech():
    query = speech_to_text()
    if query:
        # Check if response is cached
        if query in cache:
            ai_response = cache[query]
        else:
            # Generate AI response
            question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
            response = rag_chain.invoke({"input": query})
            ai_response = response["answer"]
            # Cache the response
            cache[query] = ai_response

        clean_response = ai_response.replace("**", "")

        # Use a thread pool executor to handle TTS processing in parallel
        with ThreadPoolExecutor() as executor:
            audio_html = executor.submit(text_to_speech, clean_response).result()

        return jsonify({"response": ai_response, "audio": audio_html})

    return jsonify({"error": "Speech recognition failed"}), 400

if __name__ == "__main__":
    app.run(debug=True)
