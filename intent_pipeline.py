import json
import os
import time
import pandas as pd
import numpy as np
import google.generativeai as genai
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    print("❌ ERROR: API Key is missing. Check your .env file!")
    exit()

genai.configure(api_key=API_KEY)

CURRENT_INTENT_MAP = """
- Basic Interactions (Greetings, Acknowledgment)
- About Company (Contact, Team)
- About Product (Ingredients, Price, Usage, Effectiveness)
- Logistics (Order Status, Returns, Shipping, Cancellation)
- Recommendation (Skin type queries, Solution seeking)
"""

print("⏳ Loading AI Models... (This happens only once)")
embedder = SentenceTransformer('all-MiniLM-L6-v2') 
llm = genai.GenerativeModel('gemini-2.5-flash')

def run_pipeline():
    
    print("\n1️⃣  Loading Data...")
    
    try:
        with open("inputs_for_assignment.json", "r") as f:
            raw_data = json.load(f)
        
        if "customer_messages" in raw_data:
            df = pd.DataFrame(raw_data["customer_messages"])
            print(f"   ✔ Found 'customer_messages' list.")
            
            if "current_human_message" in df.columns:
                messages = df["current_human_message"].tolist()
                print(f"   ✔ Using column: 'current_human_message'")
            else:
                raise ValueError("Column 'current_human_message' not found!")
        else:
            raise ValueError("Key 'customer_messages' not found in JSON!")

        print(f"   Loaded {len(messages)} messages.")

    except Exception as e:
        print(f"❌ Error Loading Data: {e}")
        exit()

    print("\n2️⃣  Converting text to numbers (Embeddings)...")
    embeddings = embedder.encode(messages)
    
    print("\n3️⃣  Grouping similar messages (Clustering)...")
    # For ~100 messages, 8-10 clusters is a good starting point to find niche intents
    num_clusters = 10 
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster_id'] = kmeans.fit_predict(embeddings)

    print("\n4️⃣  Asking Gemini to analyze each group...")
    
    final_report = []

    for cluster_id in range(num_clusters):
        cluster_df = df[df['cluster_id'] == cluster_id]
        
        if cluster_df.empty:
            continue

        sample_texts = cluster_df["current_human_message"].head(5).tolist()
        
        print(f"\n   🔎 Analyzing Cluster #{cluster_id} (Size: {len(cluster_df)})...")

        prompt = f"""
        Act as a Senior Data Analyst.
        
        Here is a cluster of user messages from a customer support chat:
        {json.dumps(sample_texts)}

        Our CURRENT Intent Map covers:
        {CURRENT_INTENT_MAP}

        TASK:
        1. Summarize what these users are asking.
        2. Does this specific topic fit CLEARLY into the Current Intent Map?
        3. If NO (it is a gap or specific sub-topic), recommend a NEW Intent Name.

        Output strictly in this JSON format:
        {{
            "summary": "Short summary of user need",
            "action": "KEEP_EXISTING" or "CREATE_NEW",
            "proposed_intent": "Category -> Subcategory" (Only if CREATE_NEW),
            "reason": "Why?"
        }}
        """

        try:
            response = llm.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            analysis = json.loads(clean_text)
            
            print(f"      👉 Result: {analysis['action']}")
            if analysis['action'] == "CREATE_NEW":
                print(f"      ✨ PROPOSED: {analysis['proposed_intent']}")
            
            analysis['cluster_id'] = cluster_id
            analysis['examples'] = sample_texts
            final_report.append(analysis)
            
            time.sleep(2) # Rate limit safety

        except Exception as e:
            print(f"      ❌ Error analyzing cluster: {e}")

    with open("final_report.json", "w") as f:
        json.dump(final_report, f, indent=4)
    print("\n✅ Pipeline Finished! Check 'final_report.json'")

if __name__ == "__main__":
    run_pipeline()