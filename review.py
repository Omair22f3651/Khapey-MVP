import streamlit as st
import json
import os
import base64
from dotenv import load_dotenv
from datetime import datetime
import google.generativeai as genai
from PIL import Image
import io
import pandas as pd
import nomic
from nomic import embed

load_dotenv()  # Load environment variables from .env file
# Initialize Streamlit app
st.title("PakFoodie: Food Review App")
st.write("Upload food images, write a review, and rate your experience for restaurants in Pakistan!")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available models for debugging
st.subheader("Available Models")
try:
    models = genai.list_models()
    model_list = [model.name for model in models if 'generateContent' in model.supported_generation_methods]
    st.write("Supported Gemini models for generateContent:", model_list)
except Exception as e:
    st.error(f"Error listing Gemini models: {str(e)}")

# Use the correct Gemini model ID
model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')

# Define the Gemini API prompt
GEMINI_PROMPT = """
You are an advanced image and text analysis model tasked with extracting detailed information from food review images and text for a Pakistani food app called PakFoodie. The app focuses on both local (desi) cuisine (e.g., biryani, nihari, karahi) and international cuisines (e.g., Chinese, Italian), with cultural sensitivity to the Pakistani/Muslim community. Follow the STRICT ANALYSIS PROTOCOLS and DATA EXTRACTION MATRIX provided below to analyze the images and text. Extract as much relevant information as possible, ensuring accuracy and cultural context.

**STRICT ANALYSIS PROTOCOLS:**
1. VISUAL EVIDENCE ONLY: Extract data solely from what is clearly visible in THIS image.
2. QUALITY OVER QUANTITY: Report fewer accurate fields rather than many uncertain ones.
3. CONTEXT AWARENESS: Different image types provide different data sets.
4. CULTURAL SENSITIVITY: Respect Pakistani/Muslim community context (e.g., identify halal status, avoid non-halal assumptions, consider cultural dining norms like family seating).
5. MULTI-IMAGE AWARENESS: This is image {imageNumber} of {totalImages} in the review.

**DATA EXTRACTION MATRIX:**
- **Food Close-up**: dishName, ingredients, spiceLevel (mild, medium, spicy), presentation, halalStatus ✓ | ambience, facilities ✗
- **Ambience Shot**: atmosphere, crowd, lighting, facilities, halalStatus ✓ | specific dishes ✗
- **Delivery Package**: packaging, condition, temperature cues, halalStatus ✓ | restaurant ambience ✗
- **Feature Focus**: facilities, accessibility, cleanliness, halalStatus ✓ | food details ✗
- **Experience**: staff interactions, events, celebrations, halalStatus ✓ | food specifics ✗

**Image Analysis Requirements:**
For each image, extract:
- **Cuisine Type**: Identify as Pakistani (e.g., biryani, nihari), Indian, Chinese, Italian, etc. Normalize to standard terms.
- **Lighting Quality Score**: Rate from 1 (poor) to 5 (excellent) based on clarity and visibility.
- **Face or Food Detection**: Specify if the image contains faces, food, or both.
- **Ambience**: Describe the dining environment (e.g., cozy, modern, crowded) if applicable.
- **Image Quality**: Rate from 1 (low) to 5 (high) based on sharpness and clarity.
- **Resolution**: Provided by the app (width x height in pixels).
- **Item Description**: Detailed description of visible items (e.g., dish name, ingredients, packaging).
- **Halal Status**: Infer if the food or restaurant appears halal (e.g., no pork, halal symbols) or "Unknown" if not determinable.

**Text Analysis Requirements:**
From the text review, extract:
- **Sentiment**: Positive, negative, or neutral.
- **Key Topics**: Identify main topics (e.g., food quality, staff behavior, ambience, service speed).
- **Specific Mentions**: Extract mentions of dishes, staff, or restaurant features (e.g., "biryani", "waiter", "clean tables").
- **Halal Mentions**: Note any explicit mentions of halal status or related terms.

**Input Details:**
- **Images**: {totalImages} images provided.
- **Text Review**: A free-text review of the dining experience.
- **Ratings**: Food Quality (1-5), Food Quantity (1-5), Staff (1-5), Dining Experience (1-5), Ambience (1-5), Overall (1-5).

**Output Format:**
Provide the output in JSON format with the following structure:
{
  "image_analysis": [
    {
      "image_number": int,
      "cuisine_type": str,
      "lighting_quality_score": int,
      "face_or_food_detection": str,
      "ambience": str,
      "image_quality": int,
      "resolution": str,
      "item_description": str,
      "halal_status": str
    }
  ],
  "text_analysis": {
    "sentiment": str,
    "key_topics": [str],
    "specific_mentions": [str],
    "halal_mentions": str
  },
  "ratings": {
    "food_quality": int,
    "food_quantity": int,
    "staff": int,
    "dining_experience": int,
    "ambience": int,
    "overall": int
  },
  "timestamp": str
}

**Instructions:**
1. Analyze each image according to the protocols and matrix.
2. Analyze the text review for sentiment, key topics, specific mentions, and halal references.
3. Include the provided ratings.
4. Use the current UTC timestamp in ISO format (e.g., "2025-07-07T14:45:00Z").
5. Ensure cultural sensitivity (e.g., prioritize halal status, avoid non-halal assumptions).
6. If information cannot be determined, use "Not visible", "Not applicable", or "Unknown" as appropriate.
7. Return the response as a JSON string, without markdown code fences or extra formatting.

Provide the JSON output based on the provided images, text, and ratings.
"""

# Function to encode image and get resolution
def encode_image(image_file):
    img = Image.open(image_file)
    resolution = f"{img.width}x{img.height}"
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format=img.format)
    return img_byte_arr.getvalue(), resolution

# Function to generate image embeddings using Nomic Embed Vision
def get_image_embeddings(image_file):
    try:
        img = Image.open(image_file)
        # Ensure image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Generate embeddings
        embeddings = embed.image(images=[img], model='nomic-embed-vision-v1')["embeddings"][0]
        return embeddings
    except Exception as e:
        st.error(f"Error generating image embeddings: {str(e)}")
        return None

# Function to call Gemini API
def call_gemini_api(images, text, ratings, total_images):
    try:
        # Prepare content for Gemini API
        content = [GEMINI_PROMPT.replace("{totalImages}", str(total_images))]
        image_resolutions = []
        for i, img in enumerate(images):
            img_data, resolution = encode_image(img)
            content.append({
                "mime_type": img.type,
                "data": img_data
            })
            image_resolutions.append(resolution)
        
        content.append({"text": text})
        content.append({"text": json.dumps(ratings)})
        
        # Call Gemini API
        response = model.generate_content(content)
        
        # Parse response
        response_text = response.candidates[0].content.parts[0].text
        # Strip markdown code fences if present
        response_text = response_text.strip()
        if response_text.startswith("```json") and response_text.endswith("```"):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith("```") and response_text.endswith("```"):
            response_text = response_text[3:-3].strip()
        
        # Parse JSON
        try:
            result = json.loads(response_text)
            # Validate and enrich result
            result["timestamp"] = datetime.utcnow().isoformat() + "Z"  # Ensure current timestamp
            for i, analysis in enumerate(result.get("image_analysis", [])):
                analysis["resolution"] = image_resolutions[i] if i < len(image_resolutions) else "Unknown"
                # Normalize cuisine type
                analysis["cuisine_type"] = analysis.get("cuisine_type", "Unknown").capitalize()
                # Ensure halal_status is present
                analysis["halal_status"] = analysis.get("halal_status", "Unknown")
                # Add image embeddings
                img.seek(0)  # Reset file pointer
                embeddings = get_image_embeddings(img)
                analysis["embeddings"] = embeddings if embeddings else []
            # Ensure halal_mentions in text_analysis
            result["text_analysis"]["halal_mentions"] = result["text_analysis"].get("halal_mentions", "None")
            return result
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse Gemini response as JSON: {str(e)}")
            st.write("Raw response:", response_text)
            return None
    except Exception as e:
        st.error(f"Error calling Gemini API: {str(e)}")
        return None

# Streamlit form for user input
with st.form("review_form"):
    st.subheader("Upload Images (1-3)")
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "png"], accept_multiple_files=True)
    
    st.subheader("Write Your Review")
    review_text = st.text_area("Enter your review here")
    
    st.subheader("Rate Your Experience (1-5 Stars)")
    food_quality = st.slider("Food Quality", 1, 5, 3)
    food_quantity = st.slider("Food Quantity", 1, 5, 3)
    staff = st.slider("Staff", 1, 5, 3)
    dining_experience = st.slider("Dining Experience", 1, 5, 3)
    ambience = st.slider("Ambience", 1, 5, 3)
    overall = st.slider("Overall", 1, 5, 3)
    
    submitted = st.form_submit_button("Submit Review")

# Process submission
if submitted:
    if not uploaded_files or len(uploaded_files) > 3:
        st.error("Please upload 1 to 3 images.")
    elif not review_text:
        st.error("Please enter a review text.")
    else:
        ratings = {
            "food_quality": food_quality,
            "food_quantity": food_quantity,
            "staff": staff,
            "dining_experience": dining_experience,
            "ambience": ambience,
            "overall": overall
        }
        # Call Gemini API
        result = call_gemini_api(uploaded_files, review_text, ratings, len(uploaded_files))
        
        if result:
            # Display results in UI
            st.subheader("Review Analysis")
            
            # Image Analysis Table
            st.subheader("Image Analysis")
            if result.get("image_analysis"):
                # Exclude embeddings from table for readability
                image_data = [
                    {k: v for k, v in item.items() if k != "embeddings"}
                    for item in result["image_analysis"]
                ]
                image_df = pd.DataFrame(image_data)
                st.table(image_df)
            
            # Text Analysis
            st.subheader("Text Analysis")
            text_analysis = result.get("text_analysis", {})
            st.write(f"**Sentiment**: {text_analysis.get('sentiment', 'Unknown')}")
            st.write(f"**Key Topics**: {', '.join(text_analysis.get('key_topics', []))}")
            st.write(f"**Specific Mentions**: {', '.join(text_analysis.get('specific_mentions', []))}")
            st.write(f"**Halal Mentions**: {text_analysis.get('halal_mentions', 'None')}")
            
            # Ratings Summary
            st.subheader("Ratings")
            ratings_df = pd.DataFrame([result["ratings"]])
            st.table(ratings_df)
            
            # Timestamp
            st.subheader("Timestamp")
            st.write(result.get("timestamp", "Unknown"))
            
            # Raw JSON
            st.subheader("Raw JSON Output")
            st.json(result)
            
            # Display images
            st.subheader("Uploaded Images")
            for i, img in enumerate(uploaded_files):
                img.seek(0)  # Reset file pointer
                st.image(img, caption=f"Image {i+1}")
            
            # Provide downloadable JSON
            json_str = json.dumps(result, indent=2)
            st.download_button(
                label="Download JSON",
                data=json_str,
                file_name=f"review_{datetime.utcnow().isoformat()}.json",
                mime="application/json"
            )