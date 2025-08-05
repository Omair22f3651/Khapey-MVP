import streamlit as st
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import io
import base64
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from nomic import embed
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Khapey MVP Milestone", page_icon="🍽️")
st.title("🍽️ Khapey MVP: with Qdrant and Multi-Modal Search")

# Load client prompt
PRODUCTION_PROMPT = """
Khapey-Mod v1.2 — Vision-LLM for a Pakistani food platform.

Return a JSON array with one object per image, using this schema:
{
  "image_id": string,
  "scene": "dish_closeup" | "beverage" | "menu" | "ambiance",
  "visual": {
    "description": string,
    "confidence": float,
    "quality": float,
    "appeal_score": float
  },
  "context": {
    "ambiance": "indoor" | "outdoor" | "mixed" | "none" | "unknown",
    "lighting": "daylight" | "warm" | "lowlight" | "mixed" | "unknown",
    "meal_timing": "breakfast" | "lunch" | "dinner" | "tea_time" | "any" | "unknown"
  },
  "items": [ { "name": string, "confidence": float } ],
  "flag_type": "none" | "low_appeal" | "poor_lighting" | "half_eaten" | "receipt" | "face" | "duplicate" | "inappropriate",
  "penalty_score": int,
  "keywords": [string],
  "ambience_features": [string],
  "thumbnail_rank": float
}

If the image shows no food/restaurant context, return [{"not_food_related": true}].
Use Pakistani cuisine terms (e.g., 'nihari', 'biryani'). Exclude 'embedding' field from JSON output. Output only valid JSON.
"""

class KhapeyMVP:
    def __init__(self):
        print("DEBUG: Initializing KhapeyMVP...")
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize Qdrant collection for 768-dimensional Nomic embeddings
        try:
            self.qdrant_client.recreate_collection(
                collection_name="khapey_reviews",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
            print("DEBUG: Qdrant collection 'khapey_reviews' created with 768-dimensional vectors.")
        except Exception as e:
            st.error(f"Error creating Qdrant collection: {str(e)}")
            print(f"DEBUG: Error creating Qdrant collection: {str(e)}")
    
    def analyze_review(self, images, review_data):
        """Process review matching MongoDB schema, aggregating all images"""
        print("DEBUG: Starting analyze_review...")
        results = {
            "mongodb_data": review_data,
            "ai_analysis": [],
            "quality_score": 0,
            "reach_calculation": {}
        }
        
        total_quality = 0
        individual_analyses = []
        image_embeddings = []
        image_base64_list = []
        
        for idx, image in enumerate(images):
            print(f"DEBUG: Processing image {idx+1}/{len(images)}...")
            ai_result = self._analyze_image(image)
            if ai_result and "not_food_related" not in ai_result:
                individual_analyses.append(ai_result[0])  # Single item from array
                total_quality += ai_result[0]["visual"]["quality"]
                embedding = ai_result[0]["embedding"]
                if embedding:
                    image_embeddings.append(embedding)
                image.seek(0)
                img_base64 = base64.b64encode(image.read()).decode('utf-8')
                image_base64_list.append(img_base64)
                print(f"DEBUG: Image {idx+1} processed successfully.")
            elif ai_result and "not_food_related" in ai_result:
                st.warning(f"Image {idx+1} not food-related, skipping analysis.")
                print(f"DEBUG: Image {idx+1} not food-related, skipped.")
        
        if individual_analyses:
            results["ai_analysis"] = individual_analyses
            results["quality_score"] = total_quality / len(individual_analyses) if individual_analyses else 0
            print(f"DEBUG: Aggregated quality_score: {results['quality_score']}")
            self._store_review_in_qdrant(individual_analyses, review_data, image_embeddings, image_base64_list)
        
        results["reach_calculation"] = self._calculate_reach(results, review_data)
        print("DEBUG: Reach calculation completed.")
        return results
    
    def _analyze_image(self, image_file):
        """Analyze single image with production prompt schema"""
        try:
            image_file.seek(0)
            img = Image.open(image_file)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            image_id = str(uuid4())
            print(f"DEBUG: Analyzing image with image_id: {image_id}")

            content = [
                PRODUCTION_PROMPT + f"\nGenerate analysis for image_id: {image_id}. Ensure Pakistani cuisine terms are used (e.g., 'nihari', 'biryani').",
                {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
            ]

            response = self.model.generate_content(content)
            response_text = response.candidates[0].content.parts[0].text.strip()
            print(f"DEBUG: Raw Gemini response for image_id {image_id}:\n{response_text}")

            # Log raw response
            with open("response_debug.log", "a") as f:
                f.write(f"Raw response for image_id {image_id}:\n{response_text}\n{'='*50}\n")

            # Clean response
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
            print(f"DEBUG: Cleaned response for image_id {image_id}:\n{response_text}")

            # Log cleaned response
            with open("response_debug.log", "a") as f:
                f.write(f"Cleaned response for image_id {image_id}:\n{response_text}\n{'='*50}\n")

            # Validate and parse JSON
            if not response_text.strip():
                print(f"DEBUG: Empty response for image_id {image_id}, returning not_food_related.")
                return [{"not_food_related": True}]
            
            result = json.loads(response_text)
            if not isinstance(result, (list, dict)):
                raise ValueError(f"Response must be a JSON array or dict for image_id {image_id}")
            if isinstance(result, list) and not result:
                raise ValueError(f"Response array cannot be empty for image_id {image_id}")
            if isinstance(result, dict):
                result = [result]
            
            # Generate and assign embedding
            if not any("not_food_related" in item for item in result):
                embedding = self._get_image_embedding(image_file)
                for item in result:
                    item["embedding"] = embedding.tolist() if embedding is not None else []
                print(f"DEBUG: Embedding assigned for image_id {image_id}, length: {len(result[0]['embedding']) if result[0].get('embedding') else 0}")
            
            return result

        except json.JSONDecodeError as e:
            st.error(f"JSON parsing error for image_id {image_id}: {str(e)}")
            print(f"DEBUG: JSON parsing error for image_id {image_id}: {str(e)}")
            with open("response_debug.log", "a") as f:
                f.write(f"JSON error for image_id {image_id}: {str(e)}\n{'='*50}\n")
            return None
        except Exception as e:
            st.error(f"Error analyzing image_id {image_id}: {str(e)}")
            print(f"DEBUG: Error analyzing image_id {image_id}: {str(e)}")
            return None
    
    def _get_image_embedding(self, image_file):
        """Generate 768-d embedding for an image using Nomic"""
        try:
            image_file.seek(0)
            img = Image.open(image_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            print("DEBUG: Generating image embedding with nomic-embed-vision-v1...")
            embeddings = embed.image(images=[img], model='nomic-embed-vision-v1')["embeddings"][0]
            if len(embeddings) != 768:
                st.error(f"Invalid image embedding size: {len(embeddings)}. Expected 768.")
                print(f"DEBUG: Invalid image embedding size: {len(embeddings)}. Expected 768.")
                return None
            print(f"DEBUG: Generated image embedding, length: {len(embeddings)}")
            return np.array(embeddings)
        except Exception as e:
            st.error(f"Error generating image embedding: {str(e)}")
            print(f"DEBUG: Error generating image embedding: {str(e)}")
            return None
    
    def _get_text_embedding(self, text):
        """Generate 768-d embedding for text using Nomic"""
        try:
            print(f"DEBUG: Generating text embedding for query '{text}' with nomic-embed-text-v1...")
            embeddings = embed.text(texts=[text], model='nomic-embed-text-v1')["embeddings"][0]
            if len(embeddings) != 768:
                st.error(f"Invalid text embedding size: {len(embeddings)}. Expected 768.")
                print(f"DEBUG: Invalid text embedding size: {len(embeddings)}. Expected 768.")
                return None
            print(f"DEBUG: Generated text embedding for query '{text}', length: {len(embeddings)}")
            return np.array(embeddings)
        except Exception as e:
            st.error(f"Error generating text embedding: {str(e)}")
            print(f"DEBUG: Error generating text embedding: {str(e)}")
            return None
    
    def _aggregate_analyses(self, analyses):
        """Aggregate individual image analyses into an array"""
        print(f"DEBUG: Aggregating {len(analyses)} analyses.")
        return analyses
    
    def _store_review_in_qdrant(self, analyses, review_data, image_embeddings, image_base64_list):
        """Store review metadata, embeddings, and base64 images in Qdrant"""
        if not analyses or any("not_food_related" in a for a in analyses):
            print("DEBUG: Skipping Qdrant storage due to not_food_related or empty analyses.")
            return
        
        try:
            embedding = np.mean(image_embeddings, axis=0) if image_embeddings else self._get_text_embedding(' '.join(analyses[0]["keywords"]))
            if embedding is None or len(embedding) != 768:
                print(f"DEBUG: No valid embedding for Qdrant storage (length: {len(embedding) if embedding is not None else 0}).")
                return
            payload = {
                "ai_analysis": analyses,
                "restaurantName": review_data.get("restaurantName"),
                "branchName": review_data.get("branchName"),
                "averageRating": review_data.get("averageRating"),
                "location": {"lat": 0.0, "lon": 0.0},
                "images": image_base64_list
            }
            point = PointStruct(
                id=str(uuid4()),
                vector=embedding.tolist(),
                payload=payload
            )
            self.qdrant_client.upsert(
                collection_name="khapey_reviews",
                points=[point]
            )
            print(f"DEBUG: Stored review in Qdrant with point_id: {point.id}")
        except Exception as e:
            st.error(f"Error storing review in Qdrant: {str(e)}")
            print(f"DEBUG: Error storing review in Qdrant: {str(e)}")
    
    def _calculate_reach(self, results, review_data):
        """Calculate content reach with penalties"""
        print("DEBUG: Calculating reach...")
        quality_score = results["quality_score"]
        avg_rating = review_data.get("averageRating", 3.0)
        total_reviews = 50
        engagement_rate = 0.15
        
        reputation_score = (0.4 * avg_rating) + (0.3 * engagement_rate) + (0.2 * min(total_reviews/100, 1.0)) + (0.1 * 0.8)
        reputation_multiplier = 1.25 if reputation_score >= 8.0 else 1.00 if reputation_score >= 6.0 else 0.80 if reputation_score >= 4.0 else 0.60
        penalty_score = max([a["penalty_score"] for a in results["ai_analysis"]], default=0)
        penalty_multiplier = 1 - (penalty_score / 100)
        distance_weight = 1.0
        initial_reach = quality_score * reputation_multiplier * distance_weight * penalty_multiplier
        
        reach_level = (
            "Viral (3.0x multiplier)" if initial_reach >= 80 else
            "High (2.0x multiplier)" if initial_reach >= 65 else
            "Default (1.0x multiplier)" if initial_reach >= 45 else
            "Low (0.5x multiplier)" if initial_reach >= 25 else
            "Search-only (no feed distribution)"
        )
        
        print(f"DEBUG: Reach calculation - quality_score: {quality_score}, reputation_score: {reputation_score}, penalty_multiplier: {penalty_multiplier}, initial_reach: {initial_reach}, reach_level: {reach_level}")
        return {
            "quality_score": quality_score,
            "reputation_score": reputation_score,
            "reputation_multiplier": reputation_multiplier,
            "penalty_multiplier": penalty_multiplier,
            "distance_weight": distance_weight,
            "initial_reach": initial_reach,
            "reach_level": reach_level
        }
    
    def search_reviews(self, query, user_profile, max_results=10, search_type="text"):
        """Search reviews with text-to-text or text-to-image retrieval"""
        print(f"DEBUG: Starting search with query '{query}' and search_type '{search_type}'")
        doc = self.nlp(query.lower())
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        synonyms = self._expand_synonyms(keywords)
        print(f"DEBUG: Keywords: {keywords}, Synonyms: {synonyms}")
        
        query_embedding = self._get_text_embedding(query)
        if query_embedding is None:
            print("DEBUG: Failed to generate query embedding, returning empty results.")
            return []
        
        search_results = self.qdrant_client.search(
            collection_name="khapey_reviews",
            query_vector=query_embedding.tolist(),
            limit=max_results,
            with_payload=True
        )
        print(f"DEBUG: Retrieved {len(search_results)} semantic search results from Qdrant.")
        
        keyword_results = []
        for point in self.qdrant_client.scroll(
            collection_name="khapey_reviews",
            limit=max_results,
            with_payload=True
        )[0]:
            payload = point.payload
            ai_analysis = payload.get("ai_analysis", [])
            review_keywords = []
            for analysis in ai_analysis:
                review_keywords.extend(analysis.get("keywords", []) + [item["name"] for item in analysis.get("items", [])])
            if any(kw in review_keywords for kw in keywords + synonyms):
                keyword_results.append(point)
        print(f"DEBUG: Retrieved {len(keyword_results)} keyword search results.")
        
        combined_results = []
        seen_ids = set()
        
        for result in search_results + keyword_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                combined_results.append(result)
        
        ranked_results = []
        for result in combined_results[:max_results]:
            payload = result.payload
            ai_analysis = payload.get("ai_analysis", [])
            avg_quality = np.mean([a["visual"]["quality"] for a in ai_analysis]) if ai_analysis else 0
            avg_thumbnail_rank = np.mean([a["thumbnail_rank"] for a in ai_analysis]) if ai_analysis else 0
            keyword_score = self._calculate_keyword_score(keywords + synonyms, {"keywords": [item for sublist in [a["keywords"] for a in ai_analysis] for item in sublist]})
            semantic_score = result.score
            quality_score = avg_quality / 10
            personal_score = self._calculate_personal_score(user_profile, ai_analysis)
            geo_score = self._calculate_geo_score(payload.get("location", {}), {"lat": 0.0, "lon": 0.0})
            penalty_score = max([a["penalty_score"] for a in ai_analysis], default=0)
            
            relevance = (
                0.35 * keyword_score +
                0.25 * semantic_score +
                0.20 * quality_score +
                0.15 * personal_score +
                0.05 * geo_score +
                0.05 * avg_thumbnail_rank
            ) * (1 - penalty_score / 100)
            
            ranked_results.append({
                "payload": payload,
                "relevance": relevance,
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
                "quality_score": quality_score,
                "personal_score": personal_score,
                "geo_score": geo_score,
                "thumbnail_rank": avg_thumbnail_rank
            })
            print(f"DEBUG: Ranked result - relevance: {relevance}, keyword_score: {keyword_score}, semantic_score: {semantic_score}, quality_score: {quality_score}, personal_score: {personal_score}, geo_score: {geo_score}, thumbnail_rank: {avg_thumbnail_rank}")
        
        ranked_results.sort(key=lambda x: x["relevance"], reverse=True)
        print(f"DEBUG: Search completed, returning {len(ranked_results)} results.")
        return ranked_results
    
    def _expand_synonyms(self, keywords):
        """Expand query keywords with synonyms"""
        synonyms = {
            "biryani": ["pulao", "rice dish"],
            "karahi": ["curry", "stew"],
            "spicy": ["hot", "zesty"],
            "chicken": ["poultry"],
            "breakfast": ["morning meal"],
            "family": ["group dining", "kids friendly"],
            "nihari": ["stew", "beef curry"]
        }
        expanded = []
        for kw in keywords:
            expanded.extend(synonyms.get(kw, []))
        print(f"DEBUG: Expanded synonyms for keywords {keywords}: {expanded}")
        return list(set(expanded))
    
    def _calculate_keyword_score(self, query_keywords, search_data):
        """Calculate keyword match score"""
        review_keywords = search_data.get("keywords", [])
        matches = sum(1 for kw in query_keywords if kw in review_keywords)
        score = min(matches / max(len(query_keywords), 1), 1.0)
        print(f"DEBUG: Keyword score calculation - matches: {matches}, score: {score}")
        return score
    
    def _calculate_personal_score(self, user_profile, ai_analysis):
        """Calculate personal preference fit"""
        user_cuisines = user_profile.get("cuisineWeights", {})
        quality_expectation = user_profile.get("qualityExpectation", 7.0)
        cuisine_score = 0.0
        quality_score = 0.0
        count = 0
        
        for analysis in ai_analysis:
            items = [item["name"] for item in analysis.get("items", [])]
            cuisine = "Pakistani" if any(item in ["biryani", "nihari", "karahi"] for item in items) else "Mixed"
            cuisine_score += user_cuisines.get(cuisine, 0.0)
            quality_diff = max(0, analysis["visual"]["quality"] - quality_expectation)
            quality_score += min(quality_diff / 3.0, 1.0)
            count += 1
        
        score = (cuisine_score + quality_score) / max(count, 1)
        print(f"DEBUG: Personal score - cuisine_score: {cuisine_score}, quality_score: {quality_score}, final_score: {score}")
        return score
    
    def _calculate_geo_score(self, review_location, user_location):
        """Calculate geographic proximity score (simulated)"""
        distance_km = 5.0
        score = 1.0 if distance_km <= 3 else 0.85 if distance_km <= 10 else 0.60 if distance_km <= 25 else 0.30
        print(f"DEBUG: Geo score - distance_km: {distance_km}, score: {score}")
        return score

# Initialize MVP
@st.experimental_singleton
def get_mvp():
    print("DEBUG: Initializing MVP singleton...")
    return KhapeyMVP()

mvp = get_mvp()

# Streamlit Interface
tab1, tab2, tab3 = st.tabs(["📝 Process Review", "🔍 Search & Discovery", "📊 Algorithm Demo"])

with tab1:
    st.subheader("Process Review (MongoDB Schema Compatible)")
    
    with st.form("review_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            restaurant_name = st.text_input("Restaurant Name", "Karachi Darbar")
            branch_name = st.text_input("Branch Name", "DHA Branch")
            service_type = st.selectbox("Service Type", ["dineIn", "takeAway", "delivery"])
            total_bill = st.number_input("Total Bill (PKR)", min_value=0, value=1500)
        
        with col2:
            food_taste = st.slider("Food Taste", 1, 5, 4)
            ambience = st.slider("Ambience", 1, 5, 4)
            staff = st.slider("Staff", 1, 5, 3)
            will_recommend = st.slider("Will Recommend", 1, 5, 4)
        
        user_thoughts = st.text_area("User Thoughts", "Amazing chicken karahi with perfect spice level!")
        hashtags = st.text_input("Hashtags", "#karahi #spicy #family").split()
        
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)
        
        submitted = st.form_submit_button("🚀 Process Review")
    
    if submitted and uploaded_files:
        review_data = {
            "restaurantName": restaurant_name,
            "branchName": branch_name,
            "serviceType": service_type,
            "totalBill": total_bill,
            "rateUserExperience": {
                "foodTaste": food_taste,
                "ambience": ambience,
                "staff": staff,
                "willRecommend": will_recommend
            },
            "averageRating": (food_taste + ambience + staff + will_recommend) / 4,
            "userThoughts": user_thoughts,
            "hashtags": [tag.replace("#", "") for tag in hashtags if tag.strip()]
        }
        print(f"DEBUG: Review data prepared: {review_data}")
        
        with st.spinner("Processing review..."):
            results = mvp.analyze_review(uploaded_files, review_data)
            
            if results["ai_analysis"]:
                st.success("✅ Review processed successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Quality Metrics")
                    st.metric("Overall Quality Score", f"{results['quality_score']:.0f}/100")
                    
                    reach = results["reach_calculation"]
                    st.metric("Reputation Score", f"{reach['reputation_score']:.2f}")
                    st.metric("Reach Level", reach['reach_level'])
                    st.metric("Penalty Multiplier", f"{reach['penalty_multiplier']:.2f}")
                
                with col2:
                    st.subheader("🎯 AI Analysis Summary")
                    for idx, analysis in enumerate(results["ai_analysis"]):
                        st.write(f"**Image {idx+1}**")
                        st.write(f"• Scene: {analysis['scene']}")
                        st.write(f"• Description: {analysis['visual']['description']}")
                        st.write(f"• Quality: {analysis['visual']['quality']}/10")
                        st.write(f"• Appeal Score: {analysis['visual']['appeal_score']}/10")
                        st.write(f"• Flag Type: {analysis['flag_type']}")
                        st.write(f"• Penalty Score: {analysis['penalty_score']}")
                        st.write(f"• Items: {', '.join([item['name'] for item in analysis['items']])}")
                        st.write(f"• Keywords: {', '.join(analysis['keywords'])}")
                
                with st.expander("🔧 Client Algorithm Output (Per Image)"):
                    st.json(results["ai_analysis"])
                
                with st.expander("📈 Reach Calculation Details"):
                    reach_data = results["reach_calculation"]
                    st.write("**Algorithm Components:**")
                    st.write(f"• Quality Score: {reach_data['quality_score']:.1f}")
                    st.write(f"• Reputation Multiplier: {reach_data['reputation_multiplier']:.2f}")
                    st.write(f"• Penalty Multiplier: {reach_data['penalty_multiplier']:.2f}")
                    st.write(f"• Distance Weight: {reach_data['distance_weight']:.2f}")
                    st.write(f"• **Final Reach: {reach_data['initial_reach']:.1f}**")
                    st.write(f"• **Level: {reach_data['reach_level']}**")

with tab2:
    st.subheader("Search & Discovery")
    
    st.write("**Set Your Preferences**")
    with st.form("user_profile_form"):
        cuisine_weights = {
            "Pakistani": st.slider("Pakistani Cuisine Preference", 0.0, 1.0, 0.5),
            "Italian": st.slider("Italian Cuisine Preference", 0.0, 1.0, 0.3),
            "Chinese": st.slider("Chinese Cuisine Preference", 0.0, 1.0, 0.2)
        }
        quality_expectation = st.slider("Quality Expectation (1-10)", 1.0, 10.0, 7.0)
        profile_submitted = st.form_submit_button("Save Preferences")
    
    if profile_submitted:
        st.session_state.user_profile = {
            "cuisineWeights": cuisine_weights,
            "qualityExpectation": quality_expectation
        }
        print("DEBUG: User preferences saved.")
        st.success("Preferences saved!")
    
    search_type = st.selectbox("Search Type", ["Text-to-Text", "Text-to-Image"])
    query = st.text_input("Search for food or restaurants (e.g., 'spicy chicken karahi')")
    
    if query:
        with st.spinner("Searching reviews..."):
            user_profile = st.session_state.get("user_profile", {
                "cuisineWeights": {"Pakistani": 0.5, "Italian": 0.3, "Chinese": 0.2},
                "qualityExpectation": 7.0
            })
            results = mvp.search_reviews(query, user_profile, search_type=search_type.lower())
            
            if results:
                st.subheader("Search Results")
                for i, result in enumerate(results):
                    payload = result["payload"]
                    ai_analysis = payload.get("ai_analysis", [])
                    st.write(f"**Result {i+1} (Relevance: {result['relevance']:.2f})**")
                    st.write(f"• Restaurant: {payload.get('restaurantName')} ({payload.get('branchName')})")
                    for analysis in ai_analysis:
                        st.write(f"• Scene: {analysis['scene']}")
                        st.write(f"• Items: {', '.join([item['name'] for item in analysis['items']])}")
                    st.write(f"• Quality Score: {result['quality_score']*10:.1f}/10")
                    st.write(f"• Rating: {payload.get('averageRating', 0):.1f}/5")
                    
                    images = payload.get("images", [])
                    if images:
                        st.write("**Images:**")
                        cols = st.columns(min(len(images), 3))
                        for idx, img_base64 in enumerate(images):
                            try:
                                img_data = base64.b64decode(img_base64)
                                img = Image.open(io.BytesIO(img_data))
                                with cols[idx % 3]:
                                    st.image(img, caption=f"Image {idx+1}", width=200)
                            except Exception as e:
                                st.warning(f"Error displaying image {idx+1}: {str(e)}")
                                print(f"DEBUG: Error displaying image {idx+1}: {str(e)}")
                    
                    with st.expander("Details"):
                        st.write(f"• Keyword Score: {result['keyword_score']:.2f}")
                        st.write(f"• Semantic Score: {result['semantic_score']:.2f}")
                        st.write(f"• Personal Fit: {result['personal_score']:.2f}")
                        st.write(f"• Geo Score: {result['geo_score']:.2f}")
                        st.write(f"• Thumbnail Rank: {result['thumbnail_rank']:.2f}")
            else:
                st.info("No results found. Try a different query.")
                print("DEBUG: No search results found.")

with tab3:
    st.subheader("Client Algorithm Demonstration")
    
    st.write("**This demonstrates the key components from the client's algorithm specification:**")
    
    algo_col1, algo_col2 = st.columns(2)
    
    with algo_col1:
        st.write("**✅ Implemented:**")
        st.write("• Multi-modal (text-to-image) search with Nomic embeddings")
        st.write("• New production prompt schema (Khapey-Mod v1.2)")
        st.write("• Penalty-based ranking and content moderation")
        st.write("• Quality and appeal score calculation")
        st.write("• Search & discovery pipeline with image retrieval")
        st.write("• MongoDB schema compatibility")
    
    with algo_col2:
        st.write("**🔄 Next Steps:**")
        st.write("• Implement engagement tracking")
        st.write("• Add personalization engine")
        st.write("• Build content moderation system")
        st.write("• Integrate real geolocation")
        st.write("• Add advertising system")
    
    st.subheader("📊 Algorithm Flow")
    st.write("""
    Algorithm Flow:
    1. User uploads review with multiple images.
    2. AI analysis is performed using Gemini per the production prompt.
    3. Results are returned as a JSON array of per-image analyses.
    4. Quality score is averaged across images.
    5. Penalty multiplier is applied based on highest penalty_score.
    6. Review and base64-encoded images are stored in Qdrant with 512-d embeddings.
    7. Search queries are processed with text-to-text or text-to-image matching.
    8. Results are ranked by relevance, incorporating thumbnail_rank and penalties.
    """)
    
    st.subheader("📈 Sample Algorithm Results")
    
    sample_data = {
        "High Quality Biryani": {"quality": 8.5, "reputation": 8.5, "reach": "Viral", "penalty": 0},
        "Low Appeal Karahi": {"quality": 6.5, "reputation": 6.2, "reach": "High", "penalty": 30},
        "Half-Eaten Nihari": {"quality": 7.0, "reputation": 4.5, "reach": "Low", "penalty": 45},
        "Poor Lighting Restaurant": {"quality": 3.0, "reputation": 3.0, "reach": "Search-only", "penalty": 35}
    }
    
    for restaurant, data in sample_data.items():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Quality", f"{data['quality']}/10")
        with col2:
            st.metric("Reputation", f"{data['reputation']}/10")
        with col3:
            st.metric("Reach Level", data['reach'])
        with col4:
            st.metric("Penalty", data['penalty'])