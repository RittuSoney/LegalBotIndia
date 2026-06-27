import os
import pandas as pd
import numpy as np
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from dotenv import load_dotenv
import google.genai as genai 
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import json
import time

#  CONFIGURATION 
# Load the variables from the .env file
load_dotenv()

# Securely fetch the key
api_key = os.getenv("GEMINI_API_KEY")

# Initialize the client securely
client = genai.Client(api_key=api_key)

embedder = SentenceTransformer('all-MiniLM-L6-v2')

#  DATA LOADING 
print(" Loading LARGE dataset...")
current_dir = os.path.dirname(__file__)
csv_path = os.path.join(current_dir, 'Master_Indian_Laws_2026.csv')
embeddings_path = os.path.join(current_dir, 'saved_law_embeddings.npy') # <--- THE EMBEDDED FILE

try:
    df = pd.read_csv(csv_path, encoding='unicode_escape')
    df.columns = df.columns.str.strip()
    df.fillna("Unknown", inplace=True)
    
    df['act_title'] = df['act_title'].astype(str)
    df['section'] = df['section'].astype(str)
    df['law'] = df['law'].astype(str)
    
    df['combined_text'] = df['act_title'] + " " + df['section'] + " " + df['law']
    
    #  HYBRID SEARCH: TF-IDF INITIALIZATION 
    print("⚙️ Building TF-IDF Keyword Matrix...")
    # Combine the title and the law text for maximum keyword visibility
    corpus_for_tfidf = df['act_title'].astype(str) + " " + df['law'].astype(str)

    # Initialize the vectorizer (ignoring common words like 'the', 'and', 'is')
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus_for_tfidf)
    print("✅ TF-IDF Matrix Built Successfully!")
    
    #  THE CACHING LOGIC 
    if os.path.exists(embeddings_path):
        print("⚡ Found saved brain! Loading vectors from disk...")
        law_embeddings = np.load(embeddings_path)
        print("✅ System Online (Loaded in under 1 second).")
    else:
        print("🧠 No saved brain found. Vectorizing 34k laws for the FIRST time...")
        print("👀 Watch the progress bar below. If it freezes, your laptop needs CPR.")
        
        # Added show_progress_bar=True and lowered batch_size
        law_embeddings = embedder.encode(
            df['combined_text'].tolist(), 
            show_progress_bar=True, 
            batch_size=16 
        )
        
        # Save it to disk to NEVER have to do this again
        np.save(embeddings_path, law_embeddings)
        print("💾 Brain saved to disk! System Online.")

except Exception as e:
    print(f"❌ DATA LOAD ERROR: {e}")
    df = None
    law_embeddings = None

#  VIEWS 
def chat_interface(request):
    return render(request, 'chatbot/index.html')

@csrf_exempt
def chat_api(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            user_query = data.get('query', '')
        except:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)

        if not user_query:
            return JsonResponse({'error': 'Empty query'}, status=400)
        # INPUT GUARDRAILS
        query_lower = user_query.lower()
        
        # Add any non-legal buzzwords....
        forbidden_topics = ['recipe', 'weather', 'poem', 'joke', 'movie', 'cricket', 'football', 'sports', 'lyrics', 'code', 'python']
        if any(word in query_lower for word in forbidden_topics):
            print("🛑 Guardrail Triggered: User tried asking some non-legal nonsense.")
            return JsonResponse({
                'response': "Bro, I'm an AI trained strictly on the legal codes of India, not a search engine. Keep it legal.", 
                'sources': []
            })

        # Block the keywords.
        jailbreak_phrases = ['ignore previous', 'ignore all', 'system prompt', 'you are now', 'bypass', 'write code']
        if any(phrase in query_lower for phrase in jailbreak_phrases):
            print("💀 Security Triggered: Nice try, script kiddie.")
            return JsonResponse({
                'response': "Security protocol triggered. I don't break character and I don't override my core legal directives. Nice try though.", 
                'sources': []
            })
        # 
        
        if df is None:
            return JsonResponse({'error': 'Server Error: Data not loaded'}, status=500)
        
        # START THE CLOCK HERE 
        import time
        start_time = time.time()

        # 1. Get Semantic Scores 
        query_embedding = embedder.encode([user_query])
        semantic_scores = cosine_similarity(query_embedding, law_embeddings)[0]

        # 2. Get Keyword Scores 
        query_tfidf = tfidf_vectorizer.transform([user_query])
        keyword_scores = cosine_similarity(query_tfidf, tfidf_matrix)[0]

        # 3. HYBRID MERGE (70% Meaning, 30% Keywords)
        final_hybrid_scores = (semantic_scores * 0.70) + (keyword_scores * 0.30)

        #  CONFIDENCE THRESHOLD (Hallucination Blocker) 
        highest_score = np.max(final_hybrid_scores)
        CONFIDENCE_THRESHOLD = 0.35  # Tweak this if it's too strict or too loose
        
        print(f"📊 Query Confidence Score: {highest_score:.4f}")

        if highest_score < CONFIDENCE_THRESHOLD:
            print("🛡️ Blocked! Score too low. Preventing hallucination.")
            return JsonResponse({
                'response': "I could not find any relevant Indian laws regarding your query. I am an AI trained strictly on the legal codes of India. Please rephrase or ask a legal question.",
                'sources': []
            })
        # --

        # 4. Top 5 indices (highest score first)
        top_indices = np.argsort(final_hybrid_scores)[-5:][::-1]

        #  STOP THE CLOCK HERE 
        end_time = time.time()
        latency = end_time - start_time
        print(f"⏱️ Retrieval Latency: {latency:.4f} seconds")

        # Extracting Data for Gemini 
        context_for_ai = ""
        results = []
        for idx in top_indices:
            row = df.iloc[idx]
            # Build the text block for Gemini to read
            context_for_ai += f"Act: {row['act_title']} | Section: {row['section']} | Law text: {row['law']}\n\n"
            
            # Build the clean list to send to the Frontend UI
            results.append({
                "act_title": row['act_title'],
                "section": row['section'],
                "law": row['law']
            })
        # --
        #  TERMINAL OUTPUT FOR EVALUATING PRECISION@15 
        print("\n=== TOP 15 RETRIEVED LAWS (Backend Output) ===")
        for i, res in enumerate(results):
            print(f"{i+1}. {res['act_title']} | Section: {res['section']}")
        print("================================================\n")
        # --

        # 5. Generate Answer
        try:
            prompt = f"""
            You are a helpful Indian Legal Assistant. 
            User Question: "{user_query}"
            
            Use these retrieved laws to construct your answer:
            {context_for_ai}
            
            Answer in simple English. Explain at least 3 of the retrieved laws.
            When answering questions regarding marriage, divorce, or financial support, you MUST categorize your response by applicable religions and civil codes (e.g., Hindu, Muslim, Christian, Parsi, Civil) using ONLY the provided retrieved laws.
            """
            
            ai_resp = client.models.generate_content(
                model='gemini-2.5-flash', 
                contents=prompt
            )
            final_answer = ai_resp.text
        except Exception as e:
            print(f"Gemini Error: {e}")
            #final_answer = f"I found the laws, but the AI is offline. (Error: {str(e)})"
            fallback_message = f"I am currently offline, but here are the raw legal sections I found for your query:\n\n"
            fallback_message += context_for_ai

            final_answer = fallback_message
        return JsonResponse({
            'response': final_answer,
            'sources': results
        })

    return JsonResponse({'error': 'Method not allowed'}, status=405)