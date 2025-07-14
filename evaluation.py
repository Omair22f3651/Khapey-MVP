import streamlit as st
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import google.generativeai as genai
from nomic import embed
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

st.title("Khapey: Model Evaluation Dashboard")
st.write("Evaluate Gemini image analysis and Nomic embeddings performance")

class EmbeddingEvaluator:
    def __init__(self):
        self.embeddings_cache = {}
        self.similarity_matrix = None
        
    def get_image_embedding(self, image_file, image_name):
        """Generate embedding for a single image"""
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
            st.error(f"Error generating embedding for {image_name}: {str(e)}")
            return None
    
    def calculate_similarity_matrix(self, image_files, image_names):
        """Calculate cosine similarity matrix for all image pairs"""
        embeddings_list = []
        valid_names = []
        
        for i, (img_file, name) in enumerate(zip(image_files, image_names)):
            embedding = self.get_image_embedding(img_file, name)
            if embedding is not None:
                embeddings_list.append(embedding)
                valid_names.append(name)
        
        if len(embeddings_list) < 2:
            st.error("Need at least 2 valid embeddings to calculate similarity")
            return None, None
        
        # Convert to numpy array and calculate cosine similarity
        embeddings_array = np.array(embeddings_list)
        similarity_matrix = cosine_similarity(embeddings_array)
        
        return similarity_matrix, valid_names
    
    def display_similarity_heatmap(self, similarity_matrix, image_names):
        """Display similarity matrix as heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(similarity_matrix, 
                   annot=True, 
                   fmt='.3f', 
                   xticklabels=image_names, 
                   yticklabels=image_names,
                   cmap='viridis',
                   ax=ax)
        ax.set_title('Image Similarity Matrix (Cosine Similarity)')
        st.pyplot(fig)
        
    def get_top_similar_pairs(self, similarity_matrix, image_names, top_k=5):
        """Get top K most similar image pairs"""
        pairs = []
        n = len(image_names)
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = similarity_matrix[i][j]
                pairs.append({
                    'Image 1': image_names[i],
                    'Image 2': image_names[j],
                    'Similarity': similarity
                })
        
        # Sort by similarity descending
        pairs.sort(key=lambda x: x['Similarity'], reverse=True)
        return pairs[:top_k]

class GeminiEvaluator:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash-lite-preview-06-17')
    
    def analyze_image_batch(self, images, image_names):
        """Analyze multiple images and compare results"""
        results = {}
        
        for img_file, name in zip(images, image_names):
            try:
                img_data, resolution = self.encode_image(img_file)
                
                prompt = """
                Analyze this food image and extract the following information in JSON format:
                {
                  "cuisine_type": "Pakistani/Indian/Chinese/Italian/etc",
                  "dish_name": "specific dish name if identifiable",
                  "spice_level": "mild/medium/spicy/unknown",
                  "main_ingredients": ["ingredient1", "ingredient2"],
                  "presentation_quality": 1-5,
                  "lighting_quality": 1-5,
                  "image_clarity": 1-5,
                  "halal_status": "halal/non-halal/unknown",
                  "dining_context": "restaurant/home/delivery/street",
                  "color_palette": ["dominant_color1", "dominant_color2"],
                  "food_category": "main_course/appetizer/dessert/beverage/snack"
                }
                
                Be specific and accurate. If unsure, use "unknown".
                """
                
                content = [
                    prompt,
                    {
                        "mime_type": img_file.type,
                        "data": img_data
                    }
                ]
                
                response = self.model.generate_content(content)
                response_text = response.candidates[0].content.parts[0].text.strip()
                
                # Clean response
                if response_text.startswith("```json"):
                    response_text = response_text[7:-3].strip()
                elif response_text.startswith("```"):
                    response_text = response_text[3:-3].strip()
                
                result = json.loads(response_text)
                results[name] = result
                
            except Exception as e:
                st.error(f"Error analyzing {name}: {str(e)}")
                results[name] = {"error": str(e)}
        
        return results
    
    def encode_image(self, image_file):
        """Encode image for Gemini API"""
        img = Image.open(image_file)
        resolution = f"{img.width}x{img.height}"
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format=img.format)
        return img_byte_arr.getvalue(), resolution
    
    def compare_analysis_results(self, results):
        """Compare and evaluate Gemini analysis results"""
        comparison_data = []
        
        for name, result in results.items():
            if "error" not in result:
                comparison_data.append({
                    'Image': name,
                    'Cuisine': result.get('cuisine_type', 'Unknown'),
                    'Dish': result.get('dish_name', 'Unknown'),
                    'Spice Level': result.get('spice_level', 'Unknown'),
                    'Presentation': result.get('presentation_quality', 0),
                    'Lighting': result.get('lighting_quality', 0),
                    'Clarity': result.get('image_clarity', 0),
                    'Halal Status': result.get('halal_status', 'Unknown'),
                    'Context': result.get('dining_context', 'Unknown')
                })
        
        return pd.DataFrame(comparison_data)

# Initialize evaluators
embedding_evaluator = EmbeddingEvaluator()
gemini_evaluator = GeminiEvaluator()

# Streamlit interface
st.sidebar.header("Evaluation Options")
evaluation_type = st.sidebar.selectbox(
    "Choose Evaluation Type",
    ["Embedding Similarity Analysis", "Gemini Analysis Evaluation", "Combined Evaluation"]
)

# File upload
st.subheader("Upload Test Images")
uploaded_files = st.file_uploader(
    "Choose images for evaluation", 
    type=["jpg", "png"], 
    accept_multiple_files=True,
    help="Upload 2-10 images to compare their embeddings and analysis"
)

if uploaded_files and len(uploaded_files) >= 2:
    # Get image names
    image_names = [f.name for f in uploaded_files]
    
    # Display uploaded images
    st.subheader("Uploaded Images")
    cols = st.columns(min(len(uploaded_files), 4))
    for i, (img_file, name) in enumerate(zip(uploaded_files, image_names)):
        with cols[i % 4]:
            img_file.seek(0)
            st.image(img_file, caption=name, use_column_width=True)
    
    if evaluation_type in ["Embedding Similarity Analysis", "Combined Evaluation"]:
        st.subheader("🔍 Embedding Similarity Analysis")
        
        if st.button("Calculate Embeddings & Similarities"):
            with st.spinner("Generating embeddings and calculating similarities..."):
                # Reset file pointers
                for f in uploaded_files:
                    f.seek(0)
                
                similarity_matrix, valid_names = embedding_evaluator.calculate_similarity_matrix(
                    uploaded_files, image_names
                )
                
                if similarity_matrix is not None:
                    # Display similarity heatmap
                    st.subheader("Similarity Heatmap")
                    embedding_evaluator.display_similarity_heatmap(similarity_matrix, valid_names)
                    
                    # Show top similar pairs
                    st.subheader("Most Similar Image Pairs")
                    top_pairs = embedding_evaluator.get_top_similar_pairs(
                        similarity_matrix, valid_names
                    )
                    
                    pairs_df = pd.DataFrame(top_pairs)
                    st.dataframe(pairs_df, use_container_width=True)
                    
                    # Similarity statistics
                    st.subheader("Similarity Statistics")
                    similarities = []
                    for i in range(len(valid_names)):
                        for j in range(i+1, len(valid_names)):
                            similarities.append(similarity_matrix[i][j])
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Similarity", f"{np.mean(similarities):.3f}")
                    with col2:
                        st.metric("Max Similarity", f"{np.max(similarities):.3f}")
                    with col3:
                        st.metric("Min Similarity", f"{np.min(similarities):.3f}")
                    with col4:
                        st.metric("Std Deviation", f"{np.std(similarities):.3f}")
                    
                    # Interpretation
                    st.subheader("🎯 Eye Test Interpretation")
                    st.write("**How to interpret these results:**")
                    st.write("- **High similarity (>0.8)**: Images should look very similar to human eyes")
                    st.write("- **Medium similarity (0.5-0.8)**: Images share some visual features")
                    st.write("- **Low similarity (<0.5)**: Images are visually different")
                    st.write("- **Very low similarity (<0.3)**: Images are completely different")
                    
                    # Manual validation prompts
                    st.write("**Manual Validation Questions:**")
                    for pair in top_pairs[:3]:
                        st.write(f"- Do '{pair['Image 1']}' and '{pair['Image 2']}' look similar? (Similarity: {pair['Similarity']:.3f})")
    
    if evaluation_type in ["Gemini Analysis Evaluation", "Combined Evaluation"]:
        st.subheader("🤖 Gemini Analysis Evaluation")
        
        if st.button("Analyze Images with Gemini"):
            with st.spinner("Analyzing images with Gemini..."):
                # Reset file pointers
                for f in uploaded_files:
                    f.seek(0)
                
                analysis_results = gemini_evaluator.analyze_image_batch(
                    uploaded_files, image_names
                )
                
                # Display results table
                st.subheader("Analysis Results Comparison")
                comparison_df = gemini_evaluator.compare_analysis_results(analysis_results)
                
                if not comparison_df.empty:
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Analysis insights
                    st.subheader("📊 Analysis Insights")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Cuisine Distribution:**")
                        cuisine_counts = comparison_df['Cuisine'].value_counts()
                        st.bar_chart(cuisine_counts)
                    
                    with col2:
                        st.write("**Quality Scores:**")
                        quality_cols = ['Presentation', 'Lighting', 'Clarity']
                        quality_data = comparison_df[quality_cols].mean()
                        st.bar_chart(quality_data)
                    
                    # Detailed results
                    st.subheader("Detailed Analysis Results")
                    for name, result in analysis_results.items():
                        if "error" not in result:
                            with st.expander(f"Analysis for {name}"):
                                st.json(result)
                
                # Evaluation metrics
                st.subheader("🎯 Gemini Evaluation Metrics")
                
                successful_analyses = len([r for r in analysis_results.values() if "error" not in r])
                total_analyses = len(analysis_results)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Success Rate", f"{successful_analyses}/{total_analyses}")
                with col2:
                    avg_confidence = np.mean([
                        (r.get('presentation_quality', 0) + r.get('lighting_quality', 0) + r.get('image_clarity', 0)) / 3
                        for r in analysis_results.values() if "error" not in r
                    ]) if successful_analyses > 0 else 0
                    st.metric("Avg Quality Score", f"{avg_confidence:.2f}/5")
                with col3:
                    halal_identified = len([
                        r for r in analysis_results.values() 
                        if "error" not in r and r.get('halal_status') != 'unknown'
                    ])
                    st.metric("Halal Status Identified", f"{halal_identified}/{successful_analyses}")

else:
    st.info("Please upload at least 2 images to start the evaluation.")

# Evaluation guidelines
st.sidebar.subheader("📋 Evaluation Guidelines")
st.sidebar.write("""
**For Embeddings:**
- Similar food types should have high similarity (>0.7)
- Different cuisines should have lower similarity (<0.5)
- Same dish from different angles should be very similar (>0.8)

**For Gemini Analysis:**
- Check accuracy of cuisine identification
- Verify spice level detection
- Validate halal status inference
- Assess image quality scoring
""")

# Export functionality
if st.sidebar.button("Export Evaluation Results"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create export data
    export_data = {
        "timestamp": timestamp,
        "evaluation_type": evaluation_type,
        "total_images": len(uploaded_files) if uploaded_files else 0,
        "image_names": image_names if uploaded_files else []
    }
    
    # Add similarity data if available
    if 'similarity_matrix' in locals() and similarity_matrix is not None:
        export_data["similarity_analysis"] = {
            "matrix": similarity_matrix.tolist(),
            "image_names": valid_names,
            "top_pairs": top_pairs
        }
    
    # Add Gemini analysis if available
    if 'analysis_results' in locals():
        export_data["gemini_analysis"] = analysis_results
    
    json_str = json.dumps(export_data, indent=2)
    st.sidebar.download_button(
        label="Download Evaluation Report",
        data=json_str,
        file_name=f"khapey_evaluation_{timestamp}.json",
        mime="application/json"
    )
