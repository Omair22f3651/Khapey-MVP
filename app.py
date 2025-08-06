from flask import Flask, request, render_template, jsonify
import json
import os
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
from PIL import Image, UnidentifiedImageError
import io
import base64
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from nomic import embed
import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from uuid import uuid4
import logging

# Configure logging
logging.basicConfig(filename='response_debug.log', level=logging.DEBUG)

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
print("INFO: Environment variables loaded and Gemini API configured.")

app = Flask(__name__)

# Load client prompt from prompt.json
with open(os.path.join(os.path.dirname(__file__), "prompt.json"), "r", encoding="utf-8") as f:
    _prompt_data = json.load(f)
PRODUCTION_PROMPT = _prompt_data["PRODUCTION_PROMPT"]
print("INFO: Loaded production prompt from prompt.json.")

class KhapeyMVP:
    RELEVANCY_THRESHOLD = 0.2  # Minimum relevance score to include a result

    def __init__(self):
        logging.debug("Initializing KhapeyMVP...")
        print("INFO: Initializing KhapeyMVP class...")
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        print("INFO: Gemini model initialized.")
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY")
        )
        print("INFO: Qdrant client initialized.")
        self.nlp = spacy.load("en_core_web_sm")
        print("INFO: spaCy NLP model loaded.")
        
        # Initialize Qdrant collection for 768-dimensional Nomic embeddings
        try:
            existing = self.qdrant_client.get_collections().collections
            print(f"INFO: Existing Qdrant collections: {[c.name for c in existing]}")
            if "khapey_reviews" not in [c.name for c in existing]:
                self.qdrant_client.create_collection(
                    collection_name="khapey_reviews",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                logging.debug("Qdrant collection 'khapey_reviews' created with 768-dimensional vectors.")
                print("INFO: Created Qdrant collection 'khapey_reviews' with 768-dimensional vectors.")
            else:
                logging.debug("Qdrant collection 'khapey_reviews' already exists. Skipping creation.")
                print("INFO: Qdrant collection 'khapey_reviews' already exists. Skipping creation.")
        except Exception as e:
            logging.error(f"Error checking/creating Qdrant collection: {str(e)}")
            print(f"ERROR: Failed to check/create Qdrant collection: {str(e)}")
            raise

    def analyze_review(self, images, review_data):
        logging.debug("Starting analyze_review...")
        print(f"INFO: Starting review analysis for {len(images)} images.")
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
            logging.debug(f"Processing image {idx+1}/{len(images)}...")
            print(f"INFO: Processing image {idx+1}/{len(images)}...")
            ai_result = self._analyze_image(image)
            if ai_result and "not_food_related" not in ai_result:
                individual_analyses.append(ai_result[0])
                total_quality += ai_result[0]["visual"]["quality"]
                embedding = ai_result[0]["embedding"]
                if embedding:
                    image_embeddings.append(embedding)
                image.seek(0)
                img_base64 = base64.b64encode(image.read()).decode('utf-8')
                image_base64_list.append(img_base64)
                logging.debug(f"Image {idx+1} processed successfully.")
                print(f"INFO: Image {idx+1} processed successfully.")
            elif ai_result and "not_food_related" in ai_result:
                logging.debug(f"Image {idx+1} not food-related, skipped.")
                print(f"INFO: Image {idx+1} not food-related, skipped.")
        
        if individual_analyses:
            results["ai_analysis"] = individual_analyses
            results["quality_score"] = total_quality / len(individual_analyses)
            logging.debug(f"Aggregated quality_score: {results['quality_score']}")
            print(f"INFO: Aggregated quality score: {results['quality_score']:.2f}")
            self._store_review_in_qdrant(individual_analyses, review_data, image_embeddings, image_base64_list)
        
        results["reach_calculation"] = self._calculate_reach(results, review_data)
        logging.debug("Reach calculation completed.")
        print("INFO: Reach calculation completed.")
        return results
    
    def _analyze_image(self, image_file):
        image_id = str(uuid4())  # Define image_id before try block
        logging.debug(f"Assigned image_id: {image_id}")
        print(f"INFO: Assigned image_id: {image_id}")
        try:
            image_file.seek(0)
            logging.debug(f"Opening image for image_id: {image_id}")
            print(f"INFO: Opening image for image_id: {image_id}")
            img = Image.open(image_file)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG')
            logging.debug(f"Image converted to JPEG for image_id: {image_id}")
            print(f"INFO: Image converted to JPEG for image_id: {image_id}")

            content = [
                PRODUCTION_PROMPT + f"\nGenerate analysis for image_id: {image_id}. Ensure Pakistani cuisine terms are used (e.g., 'nihari', 'biryani').",
                {"mime_type": "image/jpeg", "data": img_byte_arr.getvalue()}
            ]

            logging.debug(f"Sending image to Gemini model for image_id: {image_id}")
            print(f"INFO: Sending image to Gemini model for image_id: {image_id}")
            response = self.model.generate_content(content)
            response_text = response.candidates[0].content.parts[0].text.strip()
            logging.debug(f"Raw Gemini response for image_id {image_id}:\n{response_text}")
            print(f"INFO: Raw Gemini response for image_id {image_id} received.")

            with open("response_debug.log", "a") as f:
                f.write(f"Raw response for image_id {image_id}:\n{response_text}\n{'='*50}\n")

            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
            logging.debug(f"Cleaned response for image_id {image_id}:\n{response_text}")
            print(f"INFO: Cleaned Gemini response for image_id {image_id}.")

            with open("response_debug.log", "a") as f:
                f.write(f"Cleaned response for image_id {image_id}:\n{response_text}\n{'='*50}\n")

            if not response_text.strip():
                logging.debug(f"Empty response for image_id {image_id}, returning not_food_related.")
                print(f"INFO: Empty response for image_id {image_id}, marking as not food-related.")
                return [{"not_food_related": True}]
            
            result = json.loads(response_text)
            if not isinstance(result, (list, dict)):
                raise ValueError(f"Response must be a JSON array or dict for image_id {image_id}")
            if isinstance(result, list) and not result:
                raise ValueError(f"Response array cannot be empty for image_id {image_id}")
            if isinstance(result, dict):
                result = [result]
            
            if not any("not_food_related" in item for item in result):
                embedding = self._get_image_embedding(image_file)
                for item in result:
                    item["embedding"] = embedding.tolist() if embedding is not None else []
                logging.debug(f"Embedding assigned for image_id {image_id}, length: {len(result[0]['embedding']) if result[0].get('embedding') else 0}")
                print(f"INFO: Embedding assigned for image_id {image_id}, length: {len(result[0]['embedding']) if result[0].get('embedding') else 0}")
            
            return result

        except UnidentifiedImageError as e:
            logging.error(f"Invalid image file for image_id {image_id}: {str(e)}")
            print(f"ERROR: Invalid image file for image_id {image_id}: {str(e)}")
            with open("response_debug.log", "a") as f:
                f.write(f"Invalid image error for image_id {image_id}: {str(e)}\n{'='*50}\n")
            return None
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error for image_id {image_id}: {str(e)}")
            print(f"ERROR: JSON parsing error for image_id {image_id}: {str(e)}")
            with open("response_debug.log", "a") as f:
                f.write(f"JSON error for image_id {image_id}: {str(e)}\n{'='*50}\n")
            return None
        except Exception as e:
            logging.error(f"Error analyzing image_id {image_id}: {str(e)}")
            print(f"ERROR: Error analyzing image_id {image_id}: {str(e)}")
            return None
    
    def _get_image_embedding(self, image_file):
        try:
            image_file.seek(0)
            img = Image.open(image_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            logging.debug("Generating image embedding with nomic-embed-vision-v1...")
            print("INFO: Generating image embedding with nomic-embed-vision-v1...")
            embeddings = embed.image(images=[img], model='nomic-embed-vision-v1')["embeddings"][0]
            if len(embeddings) != 768:
                logging.error(f"Invalid image embedding size: {len(embeddings)}. Expected 768.")
                print(f"ERROR: Invalid image embedding size: {len(embeddings)}. Expected 768.")
                return None
            logging.debug(f"Generated image embedding, length: {len(embeddings)}")
            print(f"INFO: Generated image embedding, length: {len(embeddings)}")
            return np.array(embeddings)
        except Exception as e:
            logging.error(f"Error generating image embedding: {str(e)}")
            print(f"ERROR: Error generating image embedding: {str(e)}")
            return None
    
    def _get_text_embedding(self, text):
        try:
            logging.debug(f"Generating text embedding for query '{text}' with nomic-embed-text-v1...")
            print(f"INFO: Generating text embedding for query '{text}'...")
            embeddings = embed.text(texts=[text], model='nomic-embed-text-v1')["embeddings"][0]
            if len(embeddings) != 768:
                logging.error(f"Invalid text embedding size: {len(embeddings)}. Expected 768.")
                print(f"ERROR: Invalid text embedding size: {len(embeddings)}. Expected 768.")
                return None
            logging.debug(f"Generated text embedding for query '{text}', length: {len(embeddings)}")
            print(f"INFO: Generated text embedding for query '{text}', length: {len(embeddings)}")
            return np.array(embeddings)
        except Exception as e:
            logging.error(f"Error generating text embedding: {str(e)}")
            print(f"ERROR: Error generating text embedding: {str(e)}")
            return None
    
    def _store_review_in_qdrant(self, analyses, review_data, image_embeddings, image_base64_list):
        if not analyses or any("not_food_related" in a for a in analyses):
            logging.debug("Skipping Qdrant storage due to not_food_related or empty analyses.")
            print("INFO: Skipping Qdrant storage due to not_food_related or empty analyses.")
            return
        
        try:
            embedding = np.mean(image_embeddings, axis=0) if image_embeddings else self._get_text_embedding(' '.join(analyses[0]["keywords"]))
            if embedding is None or len(embedding) != 768:
                logging.debug(f"No valid embedding for Qdrant storage (length: {len(embedding) if embedding is not None else 0}).")
                print(f"INFO: No valid embedding for Qdrant storage (length: {len(embedding) if embedding is not None else 0}).")
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
            logging.debug(f"Stored review in Qdrant with point_id: {point.id}")
            print(f"INFO: Stored review in Qdrant with point_id: {point.id}")
        except Exception as e:
            logging.error(f"Error storing review in Qdrant: {str(e)}")
            print(f"ERROR: Error storing review in Qdrant: {str(e)}")
    
    def _calculate_reach(self, results, review_data):
        logging.debug("Calculating reach...")
        print("INFO: Calculating reach...")
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
        
        logging.debug(f"Reach calculation - quality_score: {quality_score}, reputation_score: {reputation_score}, penalty_multiplier: {penalty_multiplier}, initial_reach: {initial_reach}, reach_level: {reach_level}")
        print(f"INFO: Reach calculation completed - quality_score: {quality_score:.2f}, reputation_score: {reputation_score:.2f}, penalty_multiplier: {penalty_multiplier:.2f}, initial_reach: {initial_reach:.2f}, reach_level: {reach_level}")
        return {
            "quality_score": quality_score,
            "reputation_score": reputation_score,
            "reputation_multiplier": reputation_multiplier,
            "penalty_multiplier": penalty_multiplier,
            "distance_weight": distance_weight,
            "initial_reach": initial_reach,
            "reach_level": reach_level
        }
    
    def search_reviews(self, query, user_profile, max_results=10):
        logging.debug(f"Starting unified search with query '{query}'")
        print(f"INFO: Starting unified search with query '{query}'")
        doc = self.nlp(query.lower())
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop]
        synonyms = self._expand_synonyms(keywords)
        logging.debug(f"Keywords: {keywords}, Synonyms: {synonyms}")
        print(f"INFO: Extracted keywords: {keywords}, Synonyms: {synonyms}")
        
        query_embedding = self._get_text_embedding(query)
        if query_embedding is None:
            logging.debug("Failed to generate query embedding, returning empty results.")
            print("INFO: Failed to generate query embedding, returning empty results.")
            return []
        
        # Text-to-image search using image embeddings
        logging.debug("Performing text-to-image search using image embeddings.")
        print("INFO: Performing text-to-image search using image embeddings.")
        image_search_results = self.qdrant_client.search(
            collection_name="khapey_reviews",
            query_vector=query_embedding.tolist(),
            limit=max_results,
            with_payload=True
        )
        logging.debug(f"Retrieved {len(image_search_results)} text-to-image search results from Qdrant.")
        print(f"INFO: Retrieved {len(image_search_results)} text-to-image search results from Qdrant.")
        
        # Text-to-text search with semantic and keyword matching
        logging.debug("Performing text-to-text search.")
        print("INFO: Performing text-to-text search.")
        text_search_results = self.qdrant_client.search(
            collection_name="khapey_reviews",
            query_vector=query_embedding.tolist(),
            limit=max_results,
            with_payload=True
        )
        logging.debug(f"Retrieved {len(text_search_results)} semantic search results from Qdrant.")
        print(f"INFO: Retrieved {len(text_search_results)} semantic search results from Qdrant.")
        
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
        logging.debug(f"Retrieved {len(keyword_results)} keyword search results.")
        print(f"INFO: Retrieved {len(keyword_results)} keyword search results.")
        
        # Combine all results (text-to-image, semantic, and keyword)
        combined_results = []
        seen_ids = set()
        for result in image_search_results + text_search_results + keyword_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                combined_results.append(result)
        logging.debug(f"Combined {len(combined_results)} unique results from text-to-image and text-to-text searches.")
        print(f"INFO: Combined {len(combined_results)} unique results from text-to-image and text-to-text searches.")

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
            
            if relevance >= self.RELEVANCY_THRESHOLD:
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
                logging.debug(f"Ranked result - relevance: {relevance}, keyword_score: {keyword_score}, semantic_score: {semantic_score}, quality_score: {quality_score}, personal_score: {personal_score}, geo_score: {geo_score}, thumbnail_rank: {avg_thumbnail_rank}")
                print(f"INFO: Ranked result - relevance: {relevance:.2f}, keyword_score: {keyword_score:.2f}, semantic_score: {semantic_score:.2f}, quality_score: {quality_score:.2f}, personal_score: {personal_score:.2f}, geo_score: {geo_score:.2f}, thumbnail_rank: {avg_thumbnail_rank:.2f}")
            else:
                logging.debug(f"Filtered out result with low relevance: {relevance} (threshold: {self.RELEVANCY_THRESHOLD})")
                print(f"INFO: Filtered out result with low relevance: {relevance:.2f} (threshold: {self.RELEVANCY_THRESHOLD})")
        
        ranked_results.sort(key=lambda x: x["relevance"], reverse=True)
        logging.debug(f"Search completed, returning {len(ranked_results)} results after relevancy filtering.")
        print(f"INFO: Search completed, returning {len(ranked_results)} results after relevancy filtering.")
        return ranked_results[:max_results]
    
    def _expand_synonyms(self, keywords):
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
        logging.debug(f"Expanded synonyms for keywords {keywords}: {expanded}")
        print(f"INFO: Expanded synonyms for keywords {keywords}: {expanded}")
        return list(set(expanded))
    
    def _calculate_keyword_score(self, query_keywords, search_data):
        review_keywords = search_data.get("keywords", [])
        matches = sum(1 for kw in query_keywords if kw in review_keywords)
        score = min(matches / max(len(query_keywords), 1), 1.0)
        logging.debug(f"Keyword score calculation - matches: {matches}, score: {score}")
        print(f"INFO: Keyword score calculation - matches: {matches}, score: {score:.2f}")
        return score
    
    def _calculate_personal_score(self, user_profile, ai_analysis):
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
        logging.debug(f"Personal score - cuisine_score: {cuisine_score}, quality_score: {quality_score}, final_score: {score}")
        print(f"INFO: Personal score - cuisine_score: {cuisine_score:.2f}, quality_score: {quality_score:.2f}, final_score: {score:.2f}")
        return score
    
    def _calculate_geo_score(self, review_location, user_location):
        distance_km = 5.0
        score = 1.0 if distance_km <= 3 else 0.85 if distance_km <= 10 else 0.60 if distance_km <= 25 else 0.30
        logging.debug(f"Geo score - distance_km: {distance_km}, score: {score}")
        print(f"INFO: Geo score - distance_km: {distance_km:.2f}, score: {score:.2f}")
        return score

# Initialize MVP
mvp = KhapeyMVP()
print("INFO: KhapeyMVP instance created.")

@app.route('/')
def index():
    print("INFO: Rendering index page.")
    return render_template('index.html')

@app.route('/process_review', methods=['POST'])
def process_review():
    try:
        restaurant_name = request.form.get('restaurant_name', 'Karachi Darbar')
        branch_name = request.form.get('branch_name', 'DHA Branch')
        service_type = request.form.get('service_type', 'dineIn')
        total_bill = float(request.form.get('total_bill', 1500))
        food_taste = int(request.form.get('food_taste', 4))
        ambience = int(request.form.get('ambience', 4))
        staff = int(request.form.get('staff', 3))
        will_recommend = int(request.form.get('will_recommend', 4))
        user_thoughts = request.form.get('user_thoughts', 'Amazing chicken karahi with perfect spice level!')
        hashtags = request.form.get('hashtags', '#karahi #spicy #family').split()
        images = request.files.getlist('images')

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
        logging.debug(f"Review data prepared: {review_data}")
        print(f"INFO: Review data prepared: {review_data}")

        results = mvp.analyze_review(images, review_data)
        print("INFO: Review processing completed successfully.")
        return jsonify({
            "success": True,
            "results": results,
            "message": "Review processed successfully"
        })
    except Exception as e:
        logging.error(f"Error processing review: {str(e)}")
        print(f"ERROR: Error processing review: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/search', methods=['POST'])
def search():
    try:
        query = request.form.get('query', '')
        user_profile = {
            "cuisineWeights": {
                "Pakistani": float(request.form.get('pakistani_cuisine', 0.5)),
                "Italian": float(request.form.get('italian_cuisine', 0.3)),
                "Chinese": float(request.form.get('chinese_cuisine', 0.2))
            },
            "qualityExpectation": float(request.form.get('quality_expectation', 7.0))
        }
        logging.debug(f"Search initiated with query: {query}, user_profile: {user_profile}")
        print(f"INFO: Search initiated with query: {query}, user_profile: {user_profile}")

        if not query:
            print("ERROR: Search query is required.")
            return jsonify({"success": False, "error": "Query is required"}), 400

        results = mvp.search_reviews(query, user_profile)
        print("INFO: Search processing completed successfully.")
        return jsonify({
            "success": True,
            "results": results,
            "message": "Search completed successfully"
        })
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        print(f"ERROR: Error during search: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print("INFO: Starting Flask application...")
    app.run(debug=True)