Khapey MVP: Food Review and Search Platform
Overview
Khapey is a Minimum Viable Product (MVP) for a food review and discovery platform, designed to process restaurant reviews with images, analyze them using AI, and enable search and discovery of relevant reviews. The platform aligns with a client-specified algorithm for analyzing review content, calculating quality and reach, and implementing a search pipeline. This MVP integrates Google Gemini for image analysis, Qdrant Cloud for vector storage and semantic search, and Streamlit for a user-friendly web interface.
Key Features

Review Processing: Users can submit reviews with multiple images, which are analyzed to extract dish details, quality scores, and search metadata, adhering to a MongoDB-compatible schema and client JSON structure.
Search & Discovery: Supports text-based search (e.g., "spicy chicken near me") with keyword matching, semantic similarity, and personalized ranking based on user preferences.
Reach Calculation: Implements a client-defined algorithm to calculate content reach based on quality, reputation, and geographic weighting.
Qdrant Cloud Integration: Stores review metadata and embeddings in Qdrant Cloud for scalable vector search, addressing local storage limitations.
Streamlit Interface: Provides three tabs:
Process Review: Submit and view review analysis.
Search & Discovery: Search reviews with customizable user preferences.
Algorithm Demo: Showcase the client’s algorithm components and flow.



Technology Stack

Backend: Python 3.8+, Google Gemini API (image analysis), Qdrant Cloud (vector storage), Sentence Transformers (text embeddings), Nomic (image embeddings).
Frontend: Streamlit for web interface.
Dependencies: streamlit, google-generativeai, python-dotenv, pillow, qdrant-client, spacy, sentence-transformers, nomic, scikit-learn.
Environment: Qdrant Cloud for vector storage, .env for configuration.

Setup Instructions
Prerequisites

Python 3.8+ installed.
A Qdrant Cloud account (cloud.qdrant.io).
A Google Gemini API key for image analysis.
Basic familiarity with Python and command-line tools.

Installation

Clone the Repository:
git clone <repository-url>
cd khapey-mvp


Install Dependencies:
pip install streamlit google-generativeai python-dotenv pillow qdrant-client spacy sentence-transformers nomic scikit-learn
python -m spacy download en_core_web_sm


Set Up Qdrant Cloud:

Log in to cloud.qdrant.io.
Create a cluster (e.g., KhapeyCluster) in the free tier (1GB).
Copy the Cluster URL (e.g., https://xyz-example.eu-central.aws.cloud.qdrant.io:6333).
Generate an API Key in the Cluster Detail Page > API Keys tab.
Test connectivity:curl -X GET 'https://your-cluster-url:6333' --header 'api-key: your-api-key'

Expected response: {"title":"qdrant - vector search engine","version":"1.x.x"}.


Configure Environment:

Create a .env file in the project root:GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=https://your-cluster-url:6333
QDRANT_API_KEY=your-qdrant-api-key


Replace placeholders with your actual Gemini API key, Qdrant Cluster URL, and API key.
Add .env to .gitignore to keep credentials secure.


Run the Application:
streamlit run mvp_milestone.py


Access the app at http://localhost:8501 in your browser.



Usage
Process Review

Tab: Process Review
Steps:
Enter restaurant details (e.g., name, branch, service type, bill).
Rate user experience (food taste, ambience, staff, recommendation) using sliders.
Add user thoughts and hashtags.
Upload one or more images (JPEG/PNG) of food or restaurant.
Click Process Review to analyze images, aggregate results, and store in Qdrant Cloud.


Output:
Quality metrics (score, reputation, reach level).
AI analysis summary (dishes, cuisine, ingredients, or ambience details).
Aggregated JSON output and reach calculation details in expanders.



Search & Discovery

Tab: Search & Discovery
Steps:
Set preferences in the Set Your Preferences form (e.g., cuisine weights, quality expectation).
Enter a search query (e.g., "spicy chicken near me").
View ranked results with restaurant details, dish information, and relevance scores.


Output:
List of reviews matching the query, sorted by relevance.
Details include relevance score, keyword/semantic/personal/geo scores.



Algorithm Demo

Tab: Algorithm Demo
Details:
Lists implemented features (e.g., image analysis, search pipeline).
Outlines next steps (e.g., content moderation, engagement tracking).
Describes the algorithm flow (review processing to search ranking).
Shows sample results for different restaurant scenarios.



Project Structure

mvp_milestone.py: Main application code with review processing, search pipeline, and Streamlit interface.
.env: Configuration file for API keys and Qdrant Cloud URL (not in version control).
requirements.txt (optional): List of dependencies for deployment.

Algorithm Details

Review Processing:
Uses Google Gemini to analyze images, producing JSON output per the client’s specification.
Aggregates multiple image analyses into a single JSON with fields: meta, quality, food/ambience, search, moderation.
Stores results in Qdrant Cloud with embeddings for semantic search.


Search Pipeline:
Combines keyword matching (using spaCy) and semantic search (using Sentence Transformers and Qdrant).
Ranks results with the formula:Relevance = (0.35 × Keyword Match) + (0.25 × Semantic Similarity) + (0.20 × Content Quality) + (0.15 × Personal Preference Fit) + (0.05 × Geographic Proximity).


Reach Calculation:
Combines quality score, reputation multiplier, and geographic weighting to determine content distribution level (Viral, High, Default, Low, Search-only).



Limitations

Geolocation: Uses simulated distances (5km). Real geolocation (e.g., Google Maps API) is a future enhancement.
Image Embeddings: Falls back to text embeddings (sentence-transformers, 384 dimensions) if image embeddings (nomic-embed-vision-v1, 512 dimensions) fail.
Qdrant Free Tier: Limited to 1GB storage, sufficient for testing but may require a paid plan for production.
Moderation: Basic safety flags; full content moderation is planned for the next milestone.

Troubleshooting

Qdrant Connection Errors:
Verify QDRANT_URL and QDRANT_API_KEY in .env.
Check cluster status in Qdrant Cloud dashboard.
Regenerate API key if expired or invalid.


No Search Results:
Ensure reviews are uploaded to populate the khapey_reviews collection.
Test queries with keywords matching review data (e.g., "chicken", "spicy").


Gemini API Errors:
Confirm GEMINI_API_KEY is valid and has quota.


Qdrant Dashboard: Access at https://your-cluster-url:6333/dashboard to inspect collections and points.

Future Steps

Content Moderation (Section 6):
Implement multi-stage filtering (technical, safety, content analysis).
Add manual review triggers and automatic rejection rules.


Engagement Tracking (Section 3):
Track likes, shares, and views for dynamic reach and decay.


Personalization (Section 4):
Enhance user profiles with interaction-based weights and decay functions.


Geolocation:
Integrate real geolocation for accurate geo_score calculation.


Advertising System (Section 8):
Implement ad inventory qualification and targeting.



Development Notes

Qdrant Cloud: Using the free tier for development. Monitor storage usage and consider upgrading for production.
Security: Keep .env secure and avoid exposing API keys.
Testing: Upload multiple reviews to test search functionality. Use diverse queries (e.g., "biryani", "family dining") to validate ranking.
Support: Refer to Qdrant Documentation or support.qdrant.io for issues.

License
This project is for internal development and client demonstration purposes. Ensure compliance with client agreements and API usage policies (Google Gemini, Qdrant Cloud).