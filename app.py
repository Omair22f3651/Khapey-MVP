
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
import cv2
import tempfile

# Configure logging
logging.basicConfig(filename='response_debug.log', level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
logging.info("Environment variables loaded and Gemini API configured.")
print("INFO: Environment variables loaded and Gemini API configured.")

app = Flask(__name__)

# Load client prompt from prompt.json
try:
    with open(os.path.join(os.path.dirname(__file__), "prompt.json"), "r", encoding="utf-8") as f:
        _prompt_data = json.load(f)
    PRODUCTION_PROMPT = _prompt_data["PRODUCTION_PROMPT"]
    logging.info("Loaded production prompt from prompt.json.")
    print("INFO: Loaded production prompt from prompt.json.")
except Exception as e:
    logging.error(f"Failed to load prompt.json: {str(e)}")
    print(f"ERROR: Failed to load prompt.json: {str(e)}")
    raise

class KhapeyMVP:
    RELEVANCY_THRESHOLD = 0.25  # Minimum relevance score to include a result
    MAX_FILE_SIZE_MB = 20  # Maximum file size in MB
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.mov', '.avi'}

    def __init__(self):
        logging.debug("Initializing KhapeyMVP...")
        print("INFO: Initializing KhapeyMVP class...")
        try:
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
            logging.info("Gemini model initialized.")
            print("INFO: Gemini model initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini model: {str(e)}")
            print(f"ERROR: Failed to initialize Gemini model: {str(e)}")
            raise

        try:
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL"),
                api_key=os.getenv("QDRANT_API_KEY")
            )
            logging.info("Qdrant client initialized.")
            print("INFO: Qdrant client initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize Qdrant client: {str(e)}")
            print(f"ERROR: Failed to initialize Qdrant client: {str(e)}")
            raise

        try:
            self.nlp = spacy.load("en_core_web_sm")
            logging.info("spaCy NLP model loaded.")
            print("INFO: spaCy NLP model loaded.")
        except Exception as e:
            logging.error(f"Failed to load spaCy NLP model: {str(e)}")
            print(f"ERROR: Failed to load spaCy NLP model: {str(e)}")
            raise
        
        # Initialize Qdrant collection for 768-dimensional Nomic embeddings
        try:
            existing = self.qdrant_client.get_collections().collections
            logging.info(f"Existing Qdrant collections: {[c.name for c in existing]}")
            print(f"INFO: Existing Qdrant collections: {[c.name for c in existing]}")
            if "khapey_reviews" not in [c.name for c in existing]:
                self.qdrant_client.create_collection(
                    collection_name="khapey_reviews",
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE)
                )
                logging.info("Created Qdrant collection 'khapey_reviews' with 768-dimensional vectors.")
                print("INFO: Created Qdrant collection 'khapey_reviews' with 768-dimensional vectors.")
            else:
                logging.info("Qdrant collection 'khapey_reviews' already exists. Skipping creation.")
                print("INFO: Qdrant collection 'khapey_reviews' already exists. Skipping creation.")
        except Exception as e:
            logging.error(f"Error checking/creating Qdrant collection: {str(e)}")
            print(f"ERROR: Failed to check/create Qdrant collection: {str(e)}")
            raise

    def analyze_review(self, media_files, review_data):
        logging.debug(f"Starting analyze_review for {len(media_files)} media files.")
        print(f"INFO: Starting review analysis for {len(media_files)} media files.")
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
        
        for idx, media_file in enumerate(media_files):
            logging.debug(f"Processing media {idx+1}/{len(media_files)}...")
            print(f"INFO: Processing media {idx+1}/{len(media_files)}...")
            filename = media_file.filename.lower()
            content_type = media_file.content_type or ''
            is_video = filename.endswith(tuple(self.SUPPORTED_VIDEO_FORMATS)) or content_type.startswith('video/')
            logging.debug(f"Media {idx+1} detected as {'video' if is_video else 'image'}, filename: {filename}, content_type: {content_type}")
            
            # Check file size
            media_file.seek(0, os.SEEK_END)
            file_size_mb = media_file.tell() / (1024 * 1024)
            media_file.seek(0)
            if file_size_mb > self.MAX_FILE_SIZE_MB:
                logging.error(f"Media {idx+1} exceeds size limit: {file_size_mb:.2f}MB (max: {self.MAX_FILE_SIZE_MB}MB)")
                print(f"ERROR: Media {idx+1} exceeds size limit: {file_size_mb:.2f}MB")
                continue

            ai_result = self._analyze_media(media_file, is_video, review_data)
            if ai_result and "not_food_related" not in ai_result:
                individual_analyses.append(ai_result[0])
                total_quality += ai_result[0]["visual"]["quality"]
                embedding = self._get_media_embedding(media_file, is_video)
                if embedding is not None:
                    image_embeddings.append(embedding)
                media_file.seek(0)
                if is_video:
                    img_base64 = self._get_video_keyframe_base64(media_file)
                else:
                    img_base64 = base64.b64encode(media_file.read()).decode('utf-8')
                image_base64_list.append(img_base64)
                logging.debug(f"Media {idx+1} processed successfully, quality: {ai_result[0]['visual']['quality']}")
                print(f"INFO: Media {idx+1} processed successfully.")
            elif ai_result and "not_food_related" in ai_result:
                logging.debug(f"Media {idx+1} not food-related, skipped.")
                print(f"INFO: Media {idx+1} not food-related, skipped.")
            else:
                logging.warning(f"Media {idx+1} analysis failed, skipped.")
                print(f"WARN: Media {idx+1} analysis failed, skipped.")
        
        if individual_analyses:
            results["ai_analysis"] = individual_analyses
            results["quality_score"] = total_quality / len(individual_analyses)
            logging.info(f"Aggregated quality_score: {results['quality_score']:.2f}")
            print(f"INFO: Aggregated quality score: {results['quality_score']:.2f}")
            self._store_review_in_qdrant(individual_analyses, review_data, image_embeddings, image_base64_list)
        else:
            logging.warning("No valid analyses produced for any media files.")
            print("WARN: No valid analyses produced for any media files.")
        
        results["reach_calculation"] = self._calculate_reach(results, review_data)
        logging.info("Reach calculation completed.")
        print("INFO: Reach calculation completed.")
        return results
    
    def _analyze_media(self, media_file, is_video=False, review_data=None):
        image_id = str(uuid4())
        logging.debug(f"Assigned image_id: {image_id}")
        print(f"INFO: Assigned image_id: {image_id}")
        try:
            media_file.seek(0)
            logging.debug(f"Opening media for image_id: {image_id}")
            print(f"INFO: Opening media for image_id: {image_id}")
            if not is_video:
                img = Image.open(media_file)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG')
                mime_type = "image/jpeg"
                data = img_byte_arr.getvalue()
                logging.debug(f"Image converted to JPEG for image_id: {image_id}")
                print(f"INFO: Image converted to JPEG for image_id: {image_id}")
            else:
                data = media_file.read()
                mime_type = "video/mp4"
                logging.debug(f"Video data prepared for image_id: {image_id}, size: {len(data)/(1024*1024):.2f}MB")
                print(f"INFO: Video data prepared for image_id: {image_id}")

            # Append user input to prompt
            user_context = ""
            if review_data:
                user_thoughts = review_data.get("userThoughts", "")
                hashtags = review_data.get("hashtags", [])
                if user_thoughts:
                    user_context += f"User description: {user_thoughts}\n"
                if hashtags:
                    user_context += f"Hashtags: {' '.join(f'#{tag}' for tag in hashtags)}\n"
            prompt = (
                PRODUCTION_PROMPT +
                f"\nGenerate analysis for image_id: {image_id}. Ensure Pakistani cuisine terms are used (e.g., 'nihari', 'biryani'). "
                f"Use the following user-provided context to enhance analysis:\n{user_context}Return response in JSON format."
            )
            logging.debug(f"Prompt for image_id {image_id}:\n{prompt}")
            print(f"INFO: Prepared prompt for image_id {image_id} with user context.")

            content = [
                prompt,
                {"mime_type": mime_type, "data": data}
            ]

            logging.debug(f"Sending media to Gemini model for image_id: {image_id}, mime_type: {mime_type}")
            print(f"INFO: Sending media to Gemini model for image_id: {image_id}")
            response = self.model.generate_content(content)
            logging.debug(f"Received response for image_id: {image_id}")
            print(f"INFO: Received response for image_id {image_id} from Gemini model: {response}")
            response_text = response.candidates[0].content.parts[0].text.strip()
            logging.debug(f"Raw Gemini response for image_id {image_id}:\n{response_text}")
            print(f"INFO: Raw Gemini response for image_id {image_id} received.")

            with open("response_debug.log", "a", encoding="utf-8") as f:
                f.write(f"Raw response for image_id {image_id}:\n{response_text}\n{'='*50}\n")

            if not response_text:
                logging.warning(f"Empty response from Gemini for image_id {image_id}, marking as not food-related.")
                print(f"WARN: Empty response for image_id {image_id}, marking as not food-related.")
                return [{"not_food_related": True}]

            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
            logging.debug(f"Cleaned response for image_id {image_id}:\n{response_text}")
            print(f"INFO: Cleaned Gemini response for image_id {image_id}.")

            with open("response_debug.log", "a", encoding="utf-8") as f:
                f.write(f"Cleaned response for image_id {image_id}:\n{response_text}\n{'='*50}\n")

            try:
                result = json.loads(response_text)
            except json.JSONDecodeError as e:
                logging.warning(f"JSON parsing error for image_id {image_id}: {str(e)}. Falling back to text-based analysis.")
                print(f"WARN: JSON parsing error for image_id {image_id}: {str(e)}. Using fallback analysis.")
                with open("response_debug.log", "a", encoding="utf-8") as f:
                    f.write(f"JSON error for image_id {image_id}: {str(e)}\nResponse: {response_text}\n{'='*50}\n")
                
                # Fallback JSON with user input
                combined_text = response_text.lower()
                if review_data:
                    combined_text += f" {review_data.get('userThoughts', '').lower()} {' '.join(review_data.get('hashtags', []))}"
                doc = self.nlp(combined_text)
                keywords = [token.text for token in doc if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "ADJ"]]
                items = [{"name": kw, "confidence": 0.8} for kw in keywords if kw in ["burger", "fries", "karahi", "biryani", "nihari", "pizza", "curry"]]
                result = [{
                    "image_id": image_id,
                    "visual": {"quality": 5.0, "description": "Food scene", "confidence": 0.8, "appeal_score": 0.7},
                    "scene": "dish_closeup",
                    "context": {"ambiance": "unknown", "lighting": "unknown", "meal_timing": "unknown"},
                    "keywords": keywords[:7],
                    "items": items,
                    "flag_type": "none",
                    "penalty_score": 0,
                    "thumbnail_rank": 0.5,
                    "is_video": is_video
                }]
                logging.debug(f"Fallback JSON created for image_id {image_id}: {json.dumps(result, indent=2)}")
                print(f"INFO: Fallback JSON created for image_id {image_id}.")

            if not isinstance(result, (list, dict)):
                logging.error(f"Invalid response format for image_id {image_id}: Expected JSON array or dict, got {type(result)}")
                print(f"ERROR: Invalid response format for image_id {image_id}: Expected JSON array or dict")
                return None
            if isinstance(result, list) and not result:
                logging.error(f"Empty response array for image_id {image_id}")
                print(f"ERROR: Empty response array for image_id {image_id}")
                return None
            if isinstance(result, dict):
                result = [result]
            
            if not any("not_food_related" in item for item in result):
                embedding = self._get_media_embedding(media_file, is_video)
                for item in result:
                    item["embedding"] = embedding.tolist() if embedding is not None else []
                    item["is_video"] = is_video
                logging.debug(f"Embedding assigned for image_id {image_id}, length: {len(result[0]['embedding']) if result[0].get('embedding') else 0}, is_video: {is_video}")
                print(f"INFO: Embedding assigned for image_id {image_id}, length: {len(result[0]['embedding']) if result[0].get('embedding') else 0}")
            
            return result

        except UnidentifiedImageError as e:
            logging.error(f"Invalid image file for image_id {image_id}: {str(e)}")
            print(f"ERROR: Invalid image file for image_id {image_id}: {str(e)}")
            with open("response_debug.log", "a", encoding="utf-8") as f:
                f.write(f"Invalid image error for image_id {image_id}: {str(e)}\n{'='*50}\n")
            return None
        except Exception as e:
            logging.error(f"Error analyzing media for image_id {image_id}: {str(e)}")
            print(f"ERROR: Error analyzing media for image_id {image_id}: {str(e)}")
            with open("response_debug.log", "a", encoding="utf-8") as f:
                f.write(f"Analysis error for image_id {image_id}: {str(e)}\n{'='*50}\n")
            return None
    
    def _get_media_embedding(self, media_file, is_video=False):
        try:
            media_file.seek(0)
            if not is_video:
                img = Image.open(media_file)
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
            else:
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                    temp_file.write(media_file.read())
                    temp_file_path = temp_file.name
                logging.debug(f"Created temporary video file: {temp_file_path}")
                cap = cv2.VideoCapture(temp_file_path)
                if not cap.isOpened():
                    os.unlink(temp_file_path)
                    logging.error(f"Failed to open video for embedding: {temp_file_path}")
                    print(f"ERROR: Failed to open video for embedding.")
                    return None
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                duration = frame_count / fps if fps > 0 else 0
                logging.debug(f"Video metadata - frames: {frame_count}, fps: {fps:.2f}, duration: {duration:.2f}s")
                print(f"INFO: Video metadata - frames: {frame_count}, fps: {fps:.2f}, duration: {duration:.2f}s")
                num_frames = min(5, max(1, int(duration / 2)))
                if frame_count == 0:
                    os.unlink(temp_file_path)
                    logging.error(f"No frames found in video: {temp_file_path}")
                    print(f"ERROR: No frames found in video.")
                    return None
                step = frame_count // num_frames
                embeddings_list = []
                for i in range(num_frames):
                    pos = i * step
                    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                    ret, frame = cap.read()
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img = Image.fromarray(frame_rgb)
                        logging.debug(f"Extracted frame {i+1}/{num_frames} at position {pos}")
                        emb = embed.image(images=[img], model='nomic-embed-vision-v1')["embeddings"][0]
                        if len(emb) == 768:
                            embeddings_list.append(emb)
                            logging.debug(f"Generated embedding for frame {i+1}, length: {len(emb)}")
                        else:
                            logging.warning(f"Invalid embedding size for frame {i+1}: {len(emb)}. Expected 768.")
                    else:
                        logging.warning(f"Failed to read frame {i+1} at position {pos}")
                cap.release()
                os.unlink(temp_file_path)
                logging.debug(f"Deleted temporary video file: {temp_file_path}")
                if embeddings_list:
                    mean_embedding = np.mean(embeddings_list, axis=0)
                    logging.info(f"Generated mean video embedding from {len(embeddings_list)} frames, length: {len(mean_embedding)}")
                    print(f"INFO: Generated mean video embedding from {len(embeddings_list)} frames, length: {len(mean_embedding)}")
                    return mean_embedding
                logging.warning("No valid embeddings generated from video frames.")
                print(f"WARN: No valid embeddings generated from video frames.")
                return None
        except Exception as e:
            logging.error(f"Error generating media embedding: {str(e)}")
            print(f"ERROR: Error generating media embedding: {str(e)}")
            return None
    
    def _get_video_keyframe_base64(self, video_file):
        try:
            video_file.seek(0)
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(video_file.read())
                temp_file_path = temp_file.name
            logging.debug(f"Created temporary video file for keyframe: {temp_file_path}")
            cap = cv2.VideoCapture(temp_file_path)
            if not cap.isOpened():
                os.unlink(temp_file_path)
                logging.error(f"Failed to open video for keyframe extraction: {temp_file_path}")
                print(f"ERROR: Failed to open video for keyframe extraction.")
                return ""
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.debug(f"Video has {frame_count} frames")
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count // 2)
            ret, frame = cap.read()
            cap.release()
            os.unlink(temp_file_path)
            logging.debug(f"Deleted temporary video file: {temp_file_path}")
            if not ret:
                logging.warning("Failed to extract keyframe from video.")
                print(f"WARN: Failed to extract keyframe from video.")
                return ""
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            byte_arr = io.BytesIO()
            img.save(byte_arr, format='JPEG')
            logging.debug("Generated JPEG keyframe for video.")
            return base64.b64encode(byte_arr.getvalue()).decode('utf-8')
        except Exception as e:
            logging.error(f"Error extracting video keyframe: {str(e)}")
            print(f"ERROR: Error extracting video keyframe: {str(e)}")
            return ""
    
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
                "userThoughts": review_data.get("userThoughts"),
                "hashtags": review_data.get("hashtags"),
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
            logging.info(f"Stored review in Qdrant with point_id: {point.id}, is_video: {any(a.get('is_video', False) for a in analyses)}")
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
        
        logging.info(f"Reach calculation - quality_score: {quality_score:.2f}, reputation_score: {reputation_score:.2f}, penalty_multiplier: {penalty_multiplier:.2f}, initial_reach: {initial_reach:.2f}, reach_level: {reach_level}")
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
        keywords = [token.text for token in doc if token.is_alpha and not token.is_stop and token.pos_ in ["NOUN", "ADJ"]]
        synonyms = self._expand_synonyms(keywords)
        logging.debug(f"Keywords: {keywords}, Synonyms: {synonyms}")
        print(f"INFO: Extracted keywords: {keywords}, Synonyms: {synonyms}")
        
        query_embedding = self._get_text_embedding(query)
        if query_embedding is None:
            logging.warning("Failed to generate query embedding, returning empty results.")
            print("INFO: Failed to generate query embedding, returning empty results.")
            return []
        
        logging.debug("Performing text-to-image/video search using media embeddings.")
        print("INFO: Performing text-to-image/video search using media embeddings.")
        try:
            image_search_results = self.qdrant_client.search(
                collection_name="khapey_reviews",
                query_vector=query_embedding.tolist(),
                limit=max_results * 2,
                with_payload=True
            )
            logging.info(f"Retrieved {len(image_search_results)} text-to-image/video search results from Qdrant.")
            print(f"INFO: Retrieved {len(image_search_results)} text-to-image/video search results from Qdrant.")
            for result in image_search_results:
                is_video = any(a.get("is_video", False) for a in result.payload.get("ai_analysis", []))
                logging.debug(f"Text-to-image/video result - point_id: {result.id}, score: {result.score:.2f}, is_video: {is_video}")
        except Exception as e:
            logging.error(f"Error in text-to-image/video search: {str(e)}")
            print(f"ERROR: Error in text-to-image/video search: {str(e)}")
            image_search_results = []
        
        logging.debug("Performing text-to-text search.")
        print("INFO: Performing text-to-text search.")
        try:
            text_search_results = self.qdrant_client.search(
                collection_name="khapey_reviews",
                query_vector=query_embedding.tolist(),
                limit=max_results * 2,
                with_payload=True
            )
            logging.info(f"Retrieved {len(text_search_results)} semantic search results from Qdrant.")
            print(f"INFO: Retrieved {len(text_search_results)} semantic search results from Qdrant.")
            for result in text_search_results:
                is_video = any(a.get("is_video", False) for a in result.payload.get("ai_analysis", []))
                logging.debug(f"Text-to-text result - point_id: {result.id}, score: {result.score:.2f}, is_video: {is_video}")
        except Exception as e:
            logging.error(f"Error in text-to-text search: {str(e)}")
            print(f"ERROR: Error in text-to-text search: {str(e)}")
            text_search_results = []
        
        keyword_results = []
        try:
            for point in self.qdrant_client.scroll(
                collection_name="khapey_reviews",
                limit=max_results * 2,
                with_payload=True
            )[0]:
                payload = point.payload
                ai_analysis = payload.get("ai_analysis", [])
                review_keywords = []
                for analysis in ai_analysis:
                    review_keywords.extend(analysis.get("keywords", []) + [item["name"] for item in analysis.get("items", [])])
                review_keywords.extend(payload.get("userThoughts", "").lower().split())
                review_keywords.extend(payload.get("hashtags", []))
                if any(kw in review_keywords for kw in keywords + synonyms):
                    keyword_results.append(point)
                    is_video = any(a.get("is_video", False) for a in ai_analysis)
                    logging.debug(f"Keyword result - point_id: {point.id}, matched_keywords: {[kw for kw in keywords + synonyms if kw in review_keywords]}, is_video: {is_video}")
            logging.info(f"Retrieved {len(keyword_results)} keyword search results.")
            print(f"INFO: Retrieved {len(keyword_results)} keyword search results.")
        except Exception as e:
            logging.error(f"Error in keyword search: {str(e)}")
            print(f"ERROR: Error in keyword search: {str(e)}")
            keyword_results = []
        
        combined_results = []
        seen_ids = set()
        for result in image_search_results + text_search_results + keyword_results:
            if result.id not in seen_ids:
                seen_ids.add(result.id)
                combined_results.append(result)
        logging.info(f"Combined {len(combined_results)} unique results from text-to-image, text-to-text, and keyword searches.")
        print(f"INFO: Combined {len(combined_results)} unique results from text-to-image, text-to-text, and keyword searches.")

        ranked_results = []
        for result in combined_results[:max_results]:
            payload = result.payload
            ai_analysis = payload.get("ai_analysis", [])
            is_video = any(a.get("is_video", False) for a in ai_analysis)
            avg_quality = np.mean([a["visual"]["quality"] for a in ai_analysis]) if ai_analysis else 0
            avg_thumbnail_rank = np.mean([a["thumbnail_rank"] for a in ai_analysis]) if ai_analysis else 0
            keyword_score = self._calculate_keyword_score(keywords + synonyms, {
                "keywords": [item for sublist in [a["keywords"] for a in ai_analysis] for item in sublist] +
                            payload.get("userThoughts", "").lower().split() +
                            payload.get("hashtags", [])
            })
            semantic_score = result.score
            quality_score = avg_quality / 10
            personal_score = self._calculate_personal_score(user_profile, ai_analysis)
            geo_score = self._calculate_geo_score(payload.get("location", {}), {"lat": 0.0, "lon": 0.0})
            penalty_score = max([a["penalty_score"] for a in ai_analysis], default=0)
            
            video_boost = 1.2 if is_video else 1.0
            relevance = (
                0.35 * keyword_score +
                0.25 * semantic_score +
                0.25 * quality_score * video_boost +
                0.10 * personal_score +
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
                    "thumbnail_rank": avg_thumbnail_rank,
                    "is_video": is_video
                })
                logging.debug(f"Ranked result - point_id: {result.id}, relevance: {relevance:.2f}, keyword_score: {keyword_score:.2f}, semantic_score: {semantic_score:.2f}, quality_score: {quality_score:.2f}, personal_score: {personal_score:.2f}, geo_score: {geo_score:.2f}, thumbnail_rank: {avg_thumbnail_rank:.2f}, is_video: {is_video}")
                print(f"INFO: Ranked result - relevance: {relevance:.2f}, keyword_score: {keyword_score:.2f}, is_video: {is_video}")
            else:
                logging.debug(f"Filtered out result with low relevance: {relevance:.2f} (threshold: {self.RELEVANCY_THRESHOLD}), is_video: {is_video}")
                print(f"INFO: Filtered out result with low relevance: {relevance:.2f} (threshold: {self.RELEVANCY_THRESHOLD})")
        
        ranked_results.sort(key=lambda x: x["relevance"], reverse=True)
        logging.info(f"Search completed, returning {len(ranked_results)} results after relevancy filtering (videos: {sum(1 for r in ranked_results if r['is_video'])}).")
        print(f"INFO: Search completed, returning {len(ranked_results)} results with {sum(1 for r in ranked_results if r['is_video'])} videos.")
        return ranked_results[:max_results]
    
    def _expand_synonyms(self, keywords):
        synonyms = {
            "biryani": ["pulao", "rice dish"],
            "pizza": ["flatbread", "italian food"],
            "burger": ["sandwich", "fast food"],
            "karahi": ["wok", "curry dish"],
            "icecream": ["frozen dessert", "sweet treat"],
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
        logging.debug(f"Keyword score calculation - matches: {matches}, score: {score:.2f}")
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
        logging.debug(f"Personal score - cuisine_score: {cuisine_score:.2f}, quality_score: {quality_score:.2f}, final_score: {score:.2f}")
        print(f"INFO: Personal score - cuisine_score: {cuisine_score:.2f}, quality_score: {quality_score:.2f}, final_score: {score:.2f}")
        return score
    
    def _calculate_geo_score(self, review_location, user_location):
        distance_km = 5.0
        score = 1.0 if distance_km <= 3 else 0.85 if distance_km <= 10 else 0.60 if distance_km <= 25 else 0.30
        logging.debug(f"Geo score - distance_km: {distance_km:.2f}, score: {score:.2f}")
        print(f"INFO: Geo score - distance_km: {distance_km:.2f}, score: {score:.2f}")
        return score

try:
    mvp = KhapeyMVP()
    logging.info("KhapeyMVP instance created.")
    print("INFO: KhapeyMVP instance created.")
except Exception as e:
    logging.error(f"Failed to initialize KhapeyMVP: {str(e)}")
    print(f"ERROR: Failed to initialize KhapeyMVP: {str(e)}")
    raise

@app.route('/')
def index():
    logging.info("Rendering index page.")
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
        media_files = request.files.getlist('media')

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
        logging.debug(f"Review data prepared: {json.dumps(review_data, indent=2)}")
        print(f"INFO: Review data prepared: {review_data}")

        if not media_files:
            logging.warning("No media files uploaded for review.")
            print("WARN: No media files uploaded for review.")
            return jsonify({"success": False, "error": "At least one media file (image or video) is required"}), 400

        results = mvp.analyze_review(media_files, review_data)
        logging.info("Review processing completed successfully.")
        print("INFO: Review processing completed successfully.")
        return jsonify({
            "success": True,
            "results": results,
            "message": "Review processed successfully"
        })
    except Exception as e:
        logging.error(f"Error processing review: {str(e)}", exc_info=True)
        print(f"ERROR: Error processing review: {str(e)}")
        return jsonify({"success": False, "error": f"Failed to process review: {str(e)}"}), 500

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
        logging.debug(f"Search initiated with query: {query}, user_profile: {json.dumps(user_profile, indent=2)}")
        print(f"INFO: Search initiated with query: {query}, user_profile: {user_profile}")

        if not query:
            logging.warning("Search query is empty.")
            print("ERROR: Search query is required.")
            return jsonify({"success": False, "error": "Query is required"}), 400

        results = mvp.search_reviews(query, user_profile)
        logging.info("Search processing completed successfully.")
        print("INFO: Search processing completed successfully.")
        return jsonify({
            "success": True,
            "results": results,
            "message": "Search completed successfully"
        })
    except Exception as e:
        logging.error(f"Error during search: {str(e)}", exc_info=True)
        print(f"ERROR: Error during search: {str(e)}")
        return jsonify({"success": False, "error": f"Failed to process search: {str(e)}"}), 500

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    print("INFO: Starting Flask application...")
    app.run(debug=True)