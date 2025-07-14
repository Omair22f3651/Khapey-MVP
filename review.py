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
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import matplotlib.pyplot as plt
import tempfile

load_dotenv()

st.title("Khapey: Enhanced Food Review App")
st.write("Upload food images, write a review, and get comprehensive AI analysis!")

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

nomic_token = os.getenv("NOMIC_API_TOKEN")
if not nomic_token:
    st.error("Nomic API token not found. Please set NOMIC_API_TOKEN in your .env file.")

# Explicitly configure the Nomic API token
nomic.login(nomic_token)

# Enhanced Gemini prompt with more detailed extraction
ENHANCED_GEMINI_PROMPT = """
You are an expert food and restaurant analyst for Khapey, a Pakistani food review platform. Analyze the provided images and text with extreme detail and cultural sensitivity.

**ENHANCED ANALYSIS REQUIREMENTS:**

**Image Analysis (for each image):**
1. **Food Identification:**
   - Exact dish name (e.g., "Chicken Biryani", "Seekh Kebab", "Margherita Pizza")
   - Cuisine type and sub-category (e.g., "Pakistani - Punjabi", "Italian - Neapolitan")
   - Cooking method (grilled, fried, baked, steamed, etc.)
   - Serving style (family-style, individual portion, buffet, etc.)

2. **Visual Quality Assessment:**
   - Food presentation score (1-10)
   - Color vibrancy and appeal (1-10)
   - Portion size assessment (small/medium/large/extra-large)
   - Plating and garnish quality (1-10)
   - Food freshness indicators (steam, texture, color)

3. **Ingredients & Composition:**
   - Visible ingredients list
   - Estimated spice level (1-10 scale)
   - Sauce/gravy consistency and color
   - Protein type and cooking level
   - Vegetable freshness and variety
   - Rice/bread type and quality

4. **Cultural & Dietary Analysis:**
   - Halal status confidence (high/medium/low/unknown)
   - Traditional vs modern preparation style
   - Regional Pakistani cuisine indicators
   - Dietary restrictions compatibility (vegetarian, vegan, gluten-free)
   - Festival or special occasion food indicators

5. **Technical Image Quality:**
   - Resolution quality (1-10)
   - Lighting conditions (natural/artificial/mixed)
   - Camera angle effectiveness (top-down/side/angled)
   - Background cleanliness and appeal
   - Focus and clarity assessment

6. **Restaurant Environment (if visible):**
   - Dining atmosphere (casual/fine-dining/fast-food/street)
   - Cleanliness indicators
   - Table setting quality
   - Staff uniform/appearance (if visible)
   - Customer demographics (families/young adults/mixed)
   - Interior design style (traditional/modern/fusion)

**Text Analysis Enhancement:**
1. **Sentiment Analysis:**
   - Overall sentiment (positive/negative/neutral/mixed)
   - Sentiment intensity (1-10)
   - Specific aspect sentiments (food/service/ambience/value)

2. **Detailed Topic Extraction:**
   - Food quality mentions (taste, texture, temperature, freshness)
   - Service quality (speed, friendliness, accuracy, professionalism)
   - Ambience factors (noise level, lighting, music, decor)
   - Value for money assessment
   - Hygiene and cleanliness mentions
   - Wait time and efficiency comments

3. **Cultural Context:**
   - Halal/dietary requirement mentions
   - Family-friendly indicators
   - Traditional vs modern preference indicators
   - Local vs international cuisine preferences
   - Price sensitivity indicators

4. **Actionable Insights:**
   - Specific improvement suggestions mentioned
   - Recommendation likelihood (would recommend/wouldn't recommend)
   - Return visit intention
   - Comparison with other restaurants

**Output Format:**
{
  "image_analysis": [
    {
      "image_number": int,
      "food_identification": {
        "dish_name": str,
        "cuisine_type": str,
        "cuisine_subcategory": str,
        "cooking_method": str,
        "serving_style": str
      },
      "visual_quality": {
        "presentation_score": int,
        "color_vibrancy": int,
        "portion_size": str,
        "plating_quality": int,
        "freshness_indicators": [str]
      },
      "ingredients_composition": {
        "visible_ingredients": [str],
        "spice_level": int,
        "sauce_consistency": str,
        "protein_details": str,
        "vegetable_quality": str,
        "carb_type": str
      },
      "cultural_dietary": {
        "halal_confidence": str,
        "preparation_style": str,
        "regional_indicators": [str],
        "dietary_compatibility": [str],
        "special_occasion": str
      },
      "technical_quality": {
        "resolution_score": int,
        "lighting_type": str,
        "camera_angle": str,
        "background_quality": int,
        "focus_clarity": int
      },
      "environment": {
        "dining_atmosphere": str,
        "cleanliness_indicators": [str],
        "table_setting": str,
        "interior_style": str,
        "customer_demographics": str
      }
    }
  ],
  "text_analysis": {
    "sentiment": {
      "overall": str,
      "intensity": int,
      "aspect_sentiments": {
        "food": str,
        "service": str,
        "ambience": str,
        "value": str
      }
    },
    "detailed_topics": {
      "food_quality": [str],
      "service_quality": [str],
      "ambience_factors": [str],
      "value_assessment": [str],
      "hygiene_mentions": [str],
      "efficiency_comments": [str]
    },
    "cultural_context": {
      "halal_mentions": [str],
      "family_friendly": str,
      "tradition_preference": str,
      "cuisine_preference": str,
      "price_sensitivity": str
    },
    "actionable_insights": {
      "improvement_suggestions": [str],
      "recommendation_likelihood": str,
      "return_intention": str,
      "competitor_comparisons": [str]
    }
  },
  "ratings": {
    "food_quality": int,
    "food_quantity": int,
    "staff": int,
    "dining_experience": int,
    "ambience": int,
    "overall": int
  },
  "metadata": {
    "analysis_timestamp": str,
    "total_images": int,
    "analysis_confidence": float
  }
}

Analyze with maximum detail and cultural sensitivity. If information cannot be determined, use "unknown" or "not_visible".
"""

class EnhancedKhapeyAnalyzer:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
        self.embeddings_cache = {}
    
    def encode_image(self, image_file):
        """Encode image and get resolution"""
        img = Image.open(image_file)
        resolution = f"{img.width}x{img.height}"
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format)
        return img_byte_arr.getvalue(), resolution
    
    def get_image_embeddings(self, image_file, image_name):
        """Generate embeddings with caching"""
        if image_name in self.embeddings_cache:
            return self.embeddings_cache[image_name]
        
        try:
            img = Image.open(image_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            embeddings = embed.image(images=[img], model='nomic-embed-vision-v1')["embeddings"][0]
            self.embeddings_cache[image_name] = embeddings
            return embeddings
        except Exception as e:
            st.error(f"Error generating embeddings for {image_name}: {str(e)}")
            return None
    
    def calculate_image_similarities(self, uploaded_files):
        """Calculate similarities between uploaded images"""
        if len(uploaded_files) < 2:
            return None
        
        embeddings_list = []
        image_names = []
        
        for i, img_file in enumerate(uploaded_files):
            img_file.seek(0)
            embedding = self.get_image_embeddings(img_file, f"image_{i+1}")
            if embedding is not None:
                embeddings_list.append(embedding)
                image_names.append(f"Image {i+1}")
        
        if len(embeddings_list) < 2:
            return None
        
        embeddings_array = np.array(embeddings_list)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        return {
            "similarity_matrix": similarity_matrix.tolist(),
            "image_names": image_names,
            "average_similarity": float(np.mean(similarity_matrix[np.triu_indices_from(similarity_matrix, k=1)]))
        }
    
    def enhanced_gemini_analysis(self, images, text, ratings):
        """Enhanced Gemini analysis with detailed prompting"""
        try:
            content = [ENHANCED_GEMINI_PROMPT.replace("{totalImages}", str(len(images)))]
            
            # Add images
            for i, img in enumerate(images):
                img_data, resolution = self.encode_image(img)
                content.append({
                    "mime_type": img.type,
                    "data": img_data
                })
            
            # Add text and ratings
            content.append({"text": f"Review Text: {text}"})
            content.append({"text": f"Ratings: {json.dumps(ratings)}"})
            
            response = self.model.generate_content(content)
            response_text = response.candidates[0].content.parts[0].text.strip()
            
            # Clean response
            if response_text.startswith("```json"):
                response_text = response_text[7:-3].strip()
            elif response_text.startswith("```"):
                response_text = response_text[3:-3].strip()
            
            result = json.loads(response_text)
            
            # Add metadata
            result["metadata"] = {
                "analysis_timestamp": datetime.utcnow().isoformat() + "Z",
                "total_images": len(images),
                "analysis_confidence": 0.85  # Could be calculated based on response quality
            }
            
            return result
            
        except Exception as e:
            st.error(f"Error in enhanced Gemini analysis: {str(e)}")
            return None

# PDF report class
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Khapey Review Evaluation Report', 0, 1, 'C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_pdf_report(result, similarity_data, output_path):
    pdf = PDFReport()
    pdf.add_page()

    # Add summary section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Summary', 0, 1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, f"Overall Rating: {result['ratings']['overall']}/5")

    if similarity_data:
        pdf.multi_cell(0, 10, f"Average Image Similarity: {similarity_data['average_similarity']:.3f}")

    # Add image analysis section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Image Analysis', 0, 1)
    pdf.set_font('Arial', '', 12)
    for i, img_analysis in enumerate(result.get('image_analysis', [])):
        pdf.cell(0, 10, f'Image {i+1}', 0, 1)
        food_id = img_analysis.get('food_identification', {})
        pdf.multi_cell(0, 10, f"Dish: {food_id.get('dish_name', 'Unknown')}")
        pdf.multi_cell(0, 10, f"Cuisine: {food_id.get('cuisine_type', 'Unknown')}")

    # Add text analysis section
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Text Analysis', 0, 1)
    pdf.set_font('Arial', '', 12)
    sentiment = result.get('text_analysis', {}).get('sentiment', {})
    pdf.multi_cell(0, 10, f"Overall Sentiment: {sentiment.get('overall', 'Unknown')}")

    # Add a chart for ratings
    ratings = result['ratings']
    categories = list(ratings.keys())
    scores = list(ratings.values())

    plt.figure(figsize=(6, 4))
    plt.bar(categories, scores, color='skyblue')
    plt.title('Ratings Overview')
    plt.xlabel('Categories')
    plt.ylabel('Scores')

    # Save chart to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        chart_path = tmpfile.name
        plt.savefig(chart_path)
        plt.close()

    # Add chart to PDF
    pdf.add_page()
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Ratings Chart', 0, 1)
    pdf.image(chart_path, x=10, y=30, w=180)

    # Save PDF
    pdf.output(output_path)

# Initialize analyzer
analyzer = EnhancedKhapeyAnalyzer()

# Streamlit interface
with st.form("enhanced_review_form"):
    st.subheader("📸 Upload Images (1-3)")
    uploaded_files = st.file_uploader("Choose images", type=["jpg", "png"], accept_multiple_files=True)
    
    st.subheader("✍️ Write Your Review")
    review_text = st.text_area("Enter your detailed review here", height=150)
    
    st.subheader("⭐ Rate Your Experience (1-5 Stars)")
    col1, col2, col3 = st.columns(3)
    with col1:
        food_quality = st.slider("Food Quality", 1, 5, 3)
        food_quantity = st.slider("Food Quantity", 1, 5, 3)
    with col2:
        staff = st.slider("Staff", 1, 5, 3)
        dining_experience = st.slider("Dining Experience", 1, 5, 3)
    with col3:
        ambience = st.slider("Ambience", 1, 5, 3)
        overall = st.slider("Overall", 1, 5, 3)
    
    submitted = st.form_submit_button("🚀 Submit Enhanced Review", use_container_width=True)

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

        with st.spinner("🔍 Performing enhanced analysis..."):
            # Enhanced Gemini analysis
            result = analyzer.enhanced_gemini_analysis(uploaded_files, review_text, ratings)

            # Image similarity analysis
            similarity_data = analyzer.calculate_image_similarities(uploaded_files)

            if result:
                st.success("✅ Analysis completed successfully!")
                
                # Generate PDF report
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmpfile:
                    pdf_path = tmpfile.name
                    generate_pdf_report(result, similarity_data, pdf_path)
                    
                    # Provide download button for PDF
                    with open(pdf_path, "rb") as pdf_file:
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_file,
                            file_name="review_evaluation_report.pdf",
                            mime="application/pdf"
                        )
                
                # Display enhanced results
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Summary", "🖼️ Image Analysis", "📝 Text Analysis", 
                    "🔗 Similarities", "📋 Raw Data"
                ])
                
                with tab1:
                    st.subheader("Analysis Summary")
                    
                    # Key metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    if result.get("image_analysis"):
                        avg_presentation = np.mean([
                            img.get("visual_quality", {}).get("presentation_score", 0) 
                            for img in result["image_analysis"]
                            if isinstance(img.get("visual_quality", {}).get("presentation_score", 0), (int, float))
                        ])
                        with col1:
                            st.metric("Avg Presentation", f"{avg_presentation:.1f}/10")
                    
                    if result.get("text_analysis", {}).get("sentiment"):
                        sentiment_intensity = result["text_analysis"]["sentiment"].get("intensity", 0)
                        with col2:
                            st.metric("Sentiment Intensity", f"{sentiment_intensity}/10")
                    
                    with col3:
                        st.metric("Overall Rating", f"{overall}/5")
                    
                    if similarity_data:
                        with col4:
                            st.metric("Image Similarity", f"{similarity_data['average_similarity']:.3f}")
                
                with tab2:
                    st.subheader("Detailed Image Analysis")
                    
                    for i, img_analysis in enumerate(result.get("image_analysis", [])):
                        with st.expander(f"Image {i+1} Analysis"):
                            
                            # Display image
                            uploaded_files[i].seek(0)
                            st.image(uploaded_files[i], caption=f"Image {i+1}", width=300)
                            
                            # Food identification
                            food_id = img_analysis.get("food_identification", {})
                            st.write("**Food Identification:**")
                            st.write(f"- Dish: {food_id.get('dish_name', 'Unknown')}")
                            st.write(f"- Cuisine: {food_id.get('cuisine_type', 'Unknown')}")
                            st.write(f"- Cooking Method: {food_id.get('cooking_method', 'Unknown')}")
                            
                            # Visual quality
                            visual = img_analysis.get("visual_quality", {})
                            st.write("**Visual Quality:**")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Presentation", f"{visual.get('presentation_score', 0)}/10")
                            with col2:
                                st.metric("Color Vibrancy", f"{visual.get('color_vibrancy', 0)}/10")
                            with col3:
                                st.metric("Plating Quality", f"{visual.get('plating_quality', 0)}/10")
                            
                            # Cultural analysis
                            cultural = img_analysis.get("cultural_dietary", {})
                            st.write("**Cultural & Dietary:**")
                            st.write(f"- Halal Confidence: {cultural.get('halal_confidence', 'Unknown')}")
                            st.write(f"- Preparation Style: {cultural.get('preparation_style', 'Unknown')}")
                
                with tab3:
                    st.subheader("Enhanced Text Analysis")
                    
                    text_analysis = result.get("text_analysis", {})
                    
                    # Sentiment analysis
                    sentiment = text_analysis.get("sentiment", {})
                    st.write("**Sentiment Analysis:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"Overall Sentiment: **{sentiment.get('overall', 'Unknown')}**")
                        st.write(f"Intensity: **{sentiment.get('intensity', 0)}/10**")
                    
                    with col2:
                        aspect_sentiments = sentiment.get("aspect_sentiments", {})
                        st.write("**Aspect Sentiments:**")
                        for aspect, sent in aspect_sentiments.items():
                            st.write(f"- {aspect.title()}: {sent}")
                    
                    # Detailed topics
                    topics = text_analysis.get("detailed_topics", {})
                    st.write("**Key Topics Mentioned:**")
                    for topic, mentions in topics.items():
                        if mentions:
                            st.write(f"**{topic.replace('_', ' ').title()}:** {', '.join(mentions)}")
                    
                    # Actionable insights
                    insights = text_analysis.get("actionable_insights", {})
                    st.write("**Actionable Insights:**")
                    if insights.get("improvement_suggestions"):
                        st.write("**Suggestions:** " + ", ".join(insights["improvement_suggestions"]))
                    st.write(f"**Recommendation Likelihood:** {insights.get('recommendation_likelihood', 'Unknown')}")
                
                with tab4:
                    st.subheader("Image Similarity Analysis")
                    
                    if similarity_data and len(uploaded_files) > 1:
                        st.write(f"**Average Similarity:** {similarity_data['average_similarity']:.3f}")
                        
                        # Create similarity matrix display
                        similarity_matrix = np.array(similarity_data['similarity_matrix'])
                        similarity_df = pd.DataFrame(
                            similarity_matrix,
                            index=similarity_data['image_names'],
                            columns=similarity_data['image_names']
                        )
                        
                        st.write("**Similarity Matrix:**")
                        st.dataframe(similarity_df.style.format("{:.3f}"))
                        
                        # Interpretation
                        st.write("**Interpretation:**")
                        if similarity_data['average_similarity'] > 0.8:
                            st.success("🎯 High similarity - Images show very similar content")
                        elif similarity_data['average_similarity'] > 0.6:
                            st.info("📊 Medium similarity - Images share some common features")
                        else:
                            st.warning("🔍 Low similarity - Images show different content")
                    else:
                        st.info("Upload multiple images to see similarity analysis")
                
                with tab5:
                    st.subheader("Complete Analysis Data")
                    st.json(result)
                    
                    # Download options
                    col1, col2 = st.columns(2)
                    with col1:
                        json_str = json.dumps(result, indent=2)
                        st.download_button(
                            label="📥 Download Analysis JSON",
                            data=json_str,
                            file_name=f"khapey_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                    
                    with col2:
                        if similarity_data:
                            similarity_json = json.dumps(similarity_data, indent=2)
                            st.download_button(
                                label="📥 Download Similarity Data",
                                data=similarity_json,
                                file_name=f"similarity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )

# Sidebar with tips
st.sidebar.header("💡 Tips for Better Analysis")
st.sidebar.write("""
**For Best Results:**
- Upload clear, well-lit images
- Include different angles of the food
- Write detailed reviews mentioning:
  - Taste and texture
  - Service quality
  - Ambience details
  - Value for money
  - Cultural preferences

**Image Types:**
- Food close-ups
- Ambience shots
- Table settings
- Restaurant exterior
""")
