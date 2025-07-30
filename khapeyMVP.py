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
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.set_page_config(page_title="Khapey MVP Milestone", page_icon="🍽️")
st.title("🍽️ Khapey MVP: with Qdrant integration")

# Load client prompt from external file
try:
    with open("prompt.json", "r") as f:
        prompt_data = json.load(f)
        CLIENT_PROMPT = prompt_data["prompt"]
except FileNotFoundError:
    st.error("Prompt file 'prompt.json' not found!")
    CLIENT_PROMPT = ""

class KhapeyMVP:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        self.nlp = spacy.load("en_core_web_sm")
        self.text_embedder = SentenceTransformer('all-mpnet-base-v2')
        
        # Initialize Qdrant collection
        try:
            self.qdrant_client.recreate_collection(
                collection_name="khapey_reviews",
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        except Exception as e:
            st.error(f"Error creating Qdrant collection: {str(e)}")
    
    def analyze_review(self, images, review_data):
        """Process review matching MongoDB schema + client algorithm, aggregating all images"""
        
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
        
        for image in images:
            # AI Analysis for each image
            ai_result = self._analyze_image(image)
            if ai_result:
                individual_analyses.append(ai_result)
                total_quality += ai_result.get("quality", {}).get("score", 0)
                # Generate embedding for the image
                embedding = self._get_image_embedding(image)
                if embedding:
                    image_embeddings.append(embedding)
                # Convert image to base64
                image.seek(0)
                img_base64 = base64.b64encode(image.read()).decode('utf-8')
                image_base64_list.append(img_base64)
        
        # Aggregate analysis results
        if individual_analyses:
            aggregated_analysis = self._aggregate_analyses(individual_analyses)
            results["ai_analysis"] = aggregated_analysis
            results["quality_score"] = total_quality / len(individual_analyses)
            
            # Store in Qdrant with base64 images
            self._store_review_in_qdrant(aggregated_analysis, review_data, image_embeddings, image_base64_list)
        
        # Calculate reach based on client algorithm
        results["reach_calculation"] = self._calculate_reach(results, review_data)
        
        return results
    
    def _analyze_image(self, image_file):
        """Analyze single image with client's exact JSON structure"""
        try:
            # Prepare image
            image_file.seek(0)
            img = Image.open(image_file)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')

            # Dynamically update the prompt to include image-specific instructions
            dynamic_prompt = CLIENT_PROMPT + "\nAnalyze the provided image in detail and avoid using default sample outputs."

            # Call Gemini
            content = [
                dynamic_prompt,
                {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
            ]

            response = self.model.generate_content(content)
            response_text = response.candidates[0].content.parts[0].text.strip()

            # Clean response
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()

            result = json.loads(response_text)
            result["meta"]["processingTime"] = datetime.utcnow().isoformat() + "Z"

            return result

        except Exception as e:
            st.error(f"Error analyzing image: {str(e)}")
            return None
    
    def _get_image_embedding(self, image_file):
        """Generate embedding for an image using Nomic"""
        try:
            image_file.seek(0)
            img = Image.open(image_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            embeddings = embed.image(images=[img], model='nomic-embed-vision-v1')["embeddings"][0]
            return embeddings
        except Exception as e:
            st.error(f"Error generating image embedding: {str(e)}")
            return None
    
    def _get_text_embedding(self, text):
        """Generate embedding for text query using sentence-transformers"""
        return self.text_embedder.encode(text, convert_to_numpy=True)
    
    def _aggregate_analyses(self, analyses):
        """Aggregate individual image analyses into a single JSON output"""
        if not analyses:
            return None
        
        # Initialize aggregated structure
        aggregated = {
            "meta": {
                "type": "Food" if any(a["food"] for a in analyses) else "Ambience",
                "processingTime": datetime.utcnow().isoformat() + "Z"
            },
            "quality": {
                "informative": 0.0,
                "visualAppeal": 0.0,
                "score": 0.0
            },
            "food": None,
            "ambience": None,
            "search": {
                "keywords": [],
                "semantic": [],
                "ingredients": []
            },
            "moderation": {
                "safety": "safe",
                "flags": []
            }
        }
        
        # Aggregate quality scores
        informative_sum = sum(a["quality"]["informative"] for a in analyses if a)
        visual_appeal_sum = sum(a["quality"]["visualAppeal"] for a in analyses if a)
        count = len(analyses)
        aggregated["quality"]["informative"] = informative_sum / count if count > 0 else 0.0
        aggregated["quality"]["visualAppeal"] = visual_appeal_sum / count if count > 0 else 0.0
        aggregated["quality"]["score"] = (aggregated["quality"]["informative"] * 0.4 + aggregated["quality"]["visualAppeal"] * 0.6) * 10
        
        # Aggregate food or ambience data
        if aggregated["meta"]["type"] == "Food":
            dishes = []
            cuisines = set()
            ingredients = set()
            meal_times = set()
            
            for analysis in analyses:
                if analysis["food"]:
                    dishes.append(analysis["food"]["dish"])
                    cuisines.add(analysis["food"]["cuisine"])
                    ingredients.update(analysis["food"]["ingredients"])
                    meal_times.add(analysis["food"]["mealTime"])
            
            aggregated["food"] = {
                "dishes": dishes,
                "cuisine": list(cuisines)[0] if len(cuisines) == 1 else "Mixed",
                "ingredients": list(ingredients),
                "mealTime": list(meal_times)[0] if len(meal_times) == 1 else "Mixed"
            }
        else:
            atmospheres = set()
            lightings = set()
            crowds = set()
            cleanlinesses = set()
            
            for analysis in analyses:
                if analysis["ambience"]:
                    atmospheres.add(analysis["ambience"]["atmosphere"])
                    lightings.add(analysis["ambience"]["lighting"])
                    crowds.add(analysis["ambience"]["crowd"])
                    cleanlinesses.add(analysis["ambience"]["cleanliness"])
            
            aggregated["ambience"] = {
                "atmosphere": list(atmospheres)[0] if len(atmospheres) == 1 else "Mixed",
                "lighting": list(lightings)[0] if len(lightings) == 1 else "Mixed",
                "crowd": list(crowds)[0] if len(crowds) == 1 else "Mixed",
                "cleanliness": list(cleanlinesses)[0] if len(cleanlinesses) == 1 else "Mixed"
            }
        
        # Aggregate search data
        for analysis in analyses:
            if analysis["search"]:
                aggregated["search"]["keywords"].extend(analysis["search"]["keywords"])
                aggregated["search"]["semantic"].extend(analysis["search"]["semantic"])
                aggregated["search"]["ingredients"].extend(analysis["search"]["ingredients"])
        
        aggregated["search"]["keywords"] = list(set(aggregated["search"]["keywords"]))
        aggregated["search"]["semantic"] = list(set(aggregated["search"]["semantic"]))
        aggregated["search"]["ingredients"] = list(set(aggregated["search"]["ingredients"]))
        
        # Aggregate moderation data
        safety_levels = ["unsafe", "review", "safe"]  # Order of restrictiveness
        flags = set()
        for analysis in analyses:
            if analysis["moderation"]:
                current_safety = analysis["moderation"]["safety"]
                if safety_levels.index(current_safety) < safety_levels.index(aggregated["moderation"]["safety"]):
                    aggregated["moderation"]["safety"] = current_safety
                flags.update(analysis["moderation"]["flags"])
        
        aggregated["moderation"]["flags"] = list(flags)
        
        return aggregated
    
    def _store_review_in_qdrant(self, analysis, review_data, image_embeddings, image_base64_list):
        """Store review metadata, embedding, and base64 images in Qdrant"""
        try:
            # Use the first embedding or average if multiple
            embedding = np.mean(image_embeddings, axis=0) if image_embeddings else self._get_text_embedding(' '.join(analysis["search"]["semantic"]))
            
            # Create payload with review data and images
            payload = {
                "ai_analysis": analysis,
                "restaurantName": review_data.get("restaurantName"),
                "branchName": review_data.get("branchName"),
                "averageRating": review_data.get("averageRating"),
                "location": {"lat": 0.0, "lon": 0.0},  # Simulated, replace with real geolocation
                "images": image_base64_list  # Store base64-encoded images
            }
            
            # Store in Qdrant
            point = PointStruct(
                id=str(uuid4()),  # Generate a valid UUID for the point ID
                vector=embedding.tolist(),
                payload=payload
            )
            self.qdrant_client.upsert(
                collection_name="khapey_reviews",
                points=[point]
            )
        except Exception as e:
            st.error(f"Error storing review in Qdrant: {str(e)}")
    
    def _calculate_reach(self, results, review_data):
        """Calculate content reach based on client algorithm"""
        
        # Quality Score (from aggregated AI analysis)
        quality_score = results["quality_score"]
        
        # Restaurant Reputation (simulated - would come from MongoDB)
        avg_rating = review_data.get("averageRating", 3.0)
        total_reviews = 50  # Simulated
        engagement_rate = 0.15  # Simulated
        
        reputation_score = (0.4 * avg_rating) + (0.3 * engagement_rate) + (0.2 * min(total_reviews/100, 1.0)) + (0.1 * 0.8)
        
        # Reputation multiplier
        if reputation_score >= 8.0:
            reputation_multiplier = 1.25
        elif reputation_score >= 6.0:
            reputation_multiplier = 1.00
        elif reputation_score >= 4.0:
            reputation_multiplier = 0.80
        else:
            reputation_multiplier = 0.60
        
        # Geographic weighting (simulated)
        distance_weight = 1.0  # Assuming local
        
        # Final reach calculation
        initial_reach = quality_score * reputation_multiplier * distance_weight
        
        # Reach levels
        if initial_reach >= 80:
            reach_level = "Viral (3.0x multiplier)"
        elif initial_reach >= 65:
            reach_level = "High (2.0x multiplier)"
        elif initial_reach >= 45:
            reach_level = "Default (1.0x multiplier)"
        elif initial_reach >= 25:
            reach_level = "Low (0.5x multiplier)"
        else:
            reach_level = "Search-only (no feed distribution)"
        
        return {
            "quality_score": quality_score,
            "reputation_score": reputation_score,
            "reputation_multiplier": reputation_multiplier,
            "distance_weight": distance_weight,
            "initial_reach": initial_reach,
            "reach_level": reach_level
        }
    
    def search_reviews(self, query, user_profile, max_results=10):
        """Search reviews using keyword and semantic matching, returning images"""
        # Query Analysis
        doc = self.nlp(query.lower())
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        synonyms = self._expand_synonyms(keywords)
        
        # Generate query embedding
        query_embedding = self._get_text_embedding(query)
        
        # Semantic search in Qdrant
        semantic_results = self.qdrant_client.search(
            collection_name="khapey_reviews",
            query_vector=query_embedding.tolist(),
            limit=max_results,
            with_payload=True
        )
        
        # Keyword search (filter by keywords and synonyms)
        keyword_results = []
        for point in self.qdrant_client.scroll(
            collection_name="khapey_reviews",
            limit=max_results,
            with_payload=True
        )[0]:
            payload = point.payload
            ai_analysis = payload.get("ai_analysis", {})
            search_data = ai_analysis.get("search", {})
            review_keywords = search_data.get("keywords", []) + search_data.get("ingredients", [])
            if any(kw in review_keywords for kw in keywords + synonyms):
                keyword_results.append(point)
        
        # Combine results
        combined_results = []
        seen_ids = set()
        
        for result in semantic_results + keyword_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                combined_results.append(result)
        
        # Rank results
        ranked_results = []
        for result in combined_results[:max_results]:
            payload = result.payload
            ai_analysis = payload.get("ai_analysis", {})
            
            # Calculate ranking components
            keyword_score = self._calculate_keyword_score(keywords + synonyms, ai_analysis.get("search", {}))
            semantic_score = result.score  # Qdrant cosine similarity
            quality_score = ai_analysis.get("quality", {}).get("score", 0) / 100
            personal_score = self._calculate_personal_score(user_profile, ai_analysis)
            geo_score = self._calculate_geo_score(payload.get("location", {}), {"lat": 0.0, "lon": 0.0})  # Simulated user location
            
            # Search relevance formula
            relevance = (
                0.35 * keyword_score +
                0.25 * semantic_score +
                0.20 * quality_score +
                0.15 * personal_score +
                0.05 * geo_score
            )
            
            ranked_results.append({
                "payload": payload,
                "relevance": relevance,
                "keyword_score": keyword_score,
                "semantic_score": semantic_score,
                "quality_score": quality_score,
                "personal_score": personal_score,
                "geo_score": geo_score
            })
        
        # Sort by relevance
        ranked_results.sort(key=lambda x: x["relevance"], reverse=True)
        return ranked_results
    
    def _expand_synonyms(self, keywords):
        """Expand query keywords with synonyms"""
        synonyms = {
            "biryani": ["pulao", "rice dish"],
            "karahi": ["curry", "stew"],
            "spicy": ["hot", "zesty"],
            "chicken": ["poultry"],
            "breakfast": ["morning meal"],
            "family": ["group dining", "kids friendly"]
        }
        expanded = []
        for kw in keywords:
            expanded.extend(synonyms.get(kw, []))
        return list(set(expanded))
    
    def _calculate_keyword_score(self, query_keywords, search_data):
        """Calculate keyword match score"""
        review_keywords = search_data.get("keywords", []) + search_data.get("ingredients", [])
        matches = sum(1 for kw in query_keywords if kw in review_keywords)
        return min(matches / max(len(query_keywords), 1), 1.0)
    
    def _calculate_personal_score(self, user_profile, ai_analysis):
        """Calculate personal preference fit"""
        user_cuisines = user_profile.get("cuisineWeights", {})
        review_cuisine = ai_analysis.get("food", {}).get("cuisine", "Mixed")
        quality_expectation = user_profile.get("qualityExpectation", 7.0)
        review_quality = ai_analysis.get("quality", {}).get("score", 0) / 10
        
        cuisine_score = user_cuisines.get(review_cuisine, 0.0)
        quality_diff = max(0, review_quality - quality_expectation)
        quality_score = min(quality_diff / 3.0, 1.0)  # Normalize to 0-1
        return (cuisine_score + quality_score) / 2
    
    def _calculate_geo_score(self, review_location, user_location):
        """Calculate geographic proximity score (simulated)"""
        # Simulated distance calculation (replace with real geolocation)
        distance_km = 5.0  # Assume 5km for demo
        if distance_km <= 3:
            return 1.0
        elif distance_km <= 10:
            return 0.85
        elif distance_km <= 25:
            return 0.60
        else:
            return 0.30

# Initialize MVP
@st.cache_resource
def get_mvp():
    return KhapeyMVP()

mvp = get_mvp()

# Streamlit Interface
tab1, tab2, tab3 = st.tabs(["📝 Process Review", "🔍 Search & Discovery", "📊 Algorithm Demo"])

with tab1:
    st.subheader("Process Review (MongoDB Schema Compatible)")
    
    with st.form("review_form"):
        # Match MongoDB schema fields
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
        
        # Images
        uploaded_files = st.file_uploader("Upload Images", type=["jpg", "png"], accept_multiple_files=True)
        
        submitted = st.form_submit_button("🚀 Process Review")
    
    if submitted and uploaded_files:
        # Create MongoDB-compatible data structure
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
        
        with st.spinner("Processing review..."):
            # Process with MVP
            results = mvp.analyze_review(uploaded_files, review_data)
            
            if results["ai_analysis"]:
                st.success("✅ Review processed successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Quality Metrics")
                    st.metric("Overall Quality Score", f"{results['quality_score']:.0f}/100")
                    
                    reach = results["reach_calculation"]
                    st.metric("Reputation Score", f"{reach['reputation_score']:.2f}")
                    st.metric("Reach Level", reach['reach_level'])
                
                with col2:
                    st.subheader("🎯 AI Analysis Summary")
                    analysis = results["ai_analysis"]
                    if analysis.get("food"):
                        food = analysis["food"]
                        st.write(f"• Dishes: {', '.join(food['dishes'])}")
                        st.write(f"• Cuisine: {food['cuisine']}")
                        st.write(f"• Ingredients: {', '.join(food['ingredients'])}")
                        st.write(f"• Meal Time: {food['mealTime']}")
                    elif analysis.get("ambience"):
                        ambience = analysis["ambience"]
                        st.write(f"• Atmosphere: {ambience['atmosphere']}")
                        st.write(f"• Lighting: {ambience['lighting']}")
                        st.write(f"• Crowd: {ambience['crowd']}")
                        st.write(f"• Cleanliness: {ambience['cleanliness']}")
                    
                    quality = analysis["quality"]
                    st.write(f"• Quality: {quality['score']}/100")
                
                # Show aggregated client-spec JSON
                with st.expander("🔧 Client Algorithm Output (Aggregated)"):
                    st.json(results["ai_analysis"])
                
                # Show reach calculation
                with st.expander("📈 Reach Calculation Details"):
                    reach_data = results["reach_calculation"]
                    st.write("**Algorithm Components:**")
                    st.write(f"• Quality Score: {reach_data['quality_score']:.1f}")
                    st.write(f"• Reputation Multiplier: {reach_data['reputation_multiplier']:.2f}")
                    st.write(f"• Distance Weight: {reach_data['distance_weight']:.2f}")
                    st.write(f"• **Final Reach: {reach_data['initial_reach']:.1f}**")
                    st.write(f"• **Level: {reach_data['reach_level']}**")

with tab2:
    st.subheader("Search & Discovery")
    
    # User profile setup
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
        st.success("Preferences saved!")
    
    # Search interface
    query = st.text_input("Search for food or restaurants (e.g., 'spicy chicken karahi')")
    if query:
        with st.spinner("Searching reviews..."):
            user_profile = st.session_state.get("user_profile", {
                "cuisineWeights": {"Pakistani": 0.5, "Italian": 0.3, "Chinese": 0.2},
                "qualityExpectation": 7.0
            })
            results = mvp.search_reviews(query, user_profile)
            
            if results:
                st.subheader("Search Results")
                for i, result in enumerate(results):
                    payload = result["payload"]
                    ai_analysis = payload.get("ai_analysis", {})
                    st.write(f"**Result {i+1} (Relevance: {result['relevance']:.2f})**")
                    st.write(f"• Restaurant: {payload.get('restaurantName')} ({payload.get('branchName')})")
                    if ai_analysis.get("food"):
                        food = ai_analysis["food"]
                        st.write(f"• Dishes: {', '.join(food['dishes'])}")
                        st.write(f"• Cuisine: {food['cuisine']}")
                    elif ai_analysis.get("ambience"):
                        st.write(f"• Atmosphere: {ai_analysis['ambience']['atmosphere']}")
                    st.write(f"• Quality Score: {ai_analysis['quality']['score']}/100")
                    st.write(f"• Rating: {payload.get('averageRating', 0):.1f}/5")
                    
                    # Display images
                    images = payload.get("images", [])
                    if images:
                        st.write("**Images:**")
                        cols = st.columns(min(len(images), 3))  # Display up to 3 images per row
                        for idx, img_base64 in enumerate(images):
                            try:
                                img_data = base64.b64decode(img_base64)
                                img = Image.open(io.BytesIO(img_data))
                                with cols[idx % 3]:
                                    st.image(img, caption=f"Image {idx+1}", width=200)
                            except Exception as e:
                                st.warning(f"Error displaying image {idx+1}: {str(e)}")
                    
                    with st.expander("Details"):
                        st.write(f"• Keyword Score: {result['keyword_score']:.2f}")
                        st.write(f"• Semantic Score: {result['semantic_score']:.2f}")
                        st.write(f"• Personal Fit: {result['personal_score']:.2f}")
                        st.write(f"• Geo Score: {result['geo_score']:.2f}")
            else:
                st.info("No results found. Try a different query.")

with tab3:
    st.subheader("Client Algorithm Demonstration")
    
    st.write("**This demonstrates the key components from the client's algorithm specification:**")
    
    # Algorithm components
    algo_col1, algo_col2 = st.columns(2)
    
    with algo_col1:
        st.write("**✅ Implemented:**")
        st.write("• Unified image analysis output (aggregated)")
        st.write("• Quality score calculation")
        st.write("• Content reach algorithm")
        st.write("• Search & discovery pipeline with image retrieval")
        st.write("• MongoDB schema compatibility")
        st.write("• Client's JSON structure")
    
    with algo_col2:
        st.write("**🔄 Next Steps:**")
        st.write("• Implement engagement tracking")
        st.write("• Add personalization engine")
        st.write("• Build content moderation system")
        st.write("• Integrate real geolocation")
        st.write("• Add advertising system")
    
    # Show algorithm flow
    st.subheader("📊 Algorithm Flow")
    st.write("""
    Algorithm Flow:
    1. User uploads review with multiple images.
    2. AI analysis is performed using Gemini for each image.
    3. Results are aggregated into a single JSON output.
    4. Quality score is calculated as average across images.
    5. Reputation multiplier is applied.
    6. Geographic weighting is considered.
    7. Review and base64-encoded images are stored in Qdrant Cloud with embeddings.
    8. Search queries are processed with keyword and semantic matching.
    9. Results are ranked by relevance and images are displayed.
    """)
    
    # Sample data for demonstration
    st.subheader("📈 Sample Algorithm Results")
    
    sample_data = {
        "High Quality Restaurant": {"quality": 85, "reputation": 8.5, "reach": "Viral"},
        "Medium Quality Restaurant": {"quality": 65, "reputation": 6.2, "reach": "High"},
        "New Restaurant": {"quality": 70, "reputation": 4.5, "reach": "Low"},
        "Poor Quality Content": {"quality": 30, "reputation": 3.0, "reach": "Search-only"}
    }
    
    for restaurant, data in sample_data.items():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Quality", f"{data['quality']}/100")
        with col2:
            st.metric("Reputation", f"{data['reputation']}/10")
        with col3:
            st.metric("Reach Level", data['reach'])