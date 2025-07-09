# Khapey: Food Review App

## Overview

**Khapey** is a Streamlit-based web application designed for a Pakistani food review platform. It allows users to submit reviews for restaurants, including 1–3 images, a text review, and 1–5 star ratings for food quality, quantity, staff, ambience, and more.

It uses the **Google Gemini 2.5 Flash-Lite Preview model** (`gemini-2.5-flash-lite-preview-06-17`) to analyze images and text — extracting details like cuisine type, ambience, and sentiment — with cultural sensitivity to the **Pakistani/Muslim context**.

---

## ✨ Features

- **Image Upload**: Upload 1–3 images (JPG/PNG) of food, ambience, or dining space.
- **Text Review**: Write a descriptive review of your dining experience.
- **Ratings**: Provide 1–5 star ratings for:
  - Food Quality
  - Food Quantity
  - Staff
  - Dining Experience
  - Ambience
  - Overall
- **Gemini AI Integration**:
  - Image analysis: cuisine type, spice level, ambience, halal hints, etc.
  - Text analysis: sentiment, key topics, specific mentions.
- **Output**: JSON-based structured response shown in the app and available for download.
- **Cultural Sensitivity**: Reviews are interpreted with halal and cultural considerations.
- **Model Debugging**: Lists supported Gemini models for transparency.

---

## 🛠️ Prerequisites

- Python 3.8+
- Gemini API key with access to multimodal models
- Internet connection

---

## 🔧 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Omair22f3651/Khapey-MVP.git
cd Khapey-MVP
2. Create and Activate a Virtual Environment
On Windows:

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate
On macOS/Linux:

bash
Copy
Edit
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
This includes:

streamlit

google-generativeai

pillow

pandas

nomic

python-dotenv

4. Set Up Environment Variables
Option A: Create a .env file (Recommended)
Create a file named .env:

env
Copy
Edit
GEMINI_API_KEY=your-gemini-api-key
In review.py, make sure you have:

python
Copy
Edit
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
Option B: Set the variable manually
On macOS/Linux:

bash
Copy
Edit
export GEMINI_API_KEY=your-gemini-api-key
On Windows CMD:

cmd
Copy
Edit
set GEMINI_API_KEY=your-gemini-api-key
On Windows PowerShell:

powershell
Copy
Edit
$env:GEMINI_API_KEY="your-gemini-api-key"
5. Run the App
bash
Copy
Edit
streamlit run review.py
Open http://localhost:8501 in your browser.

🧠 How Gemini is Used
The app sends a structured prompt and multimodal content (text + images) to the Gemini API.

Gemini returns a structured response including:

image_analysis: details like cuisine, lighting, ambience

text_analysis: sentiment, key topics, halal mentions

ratings: your given input

timestamp: ISO format

🛠️ Troubleshooting
❌ Model Not Found
Ensure your key supports gemini-2.5-flash-lite-preview-06-17.

❌ JSON Decode Error
If the response is wrapped in markdown:

python
Copy
Edit
response_text = response_text.strip()
if response_text.startswith("```json"):
    response_text = response_text[7:-3].strip()
❌ Nomic Errors (for image embeddings)
Make sure you're logged in:

bash
Copy
Edit
nomic login
Then paste the access token when prompted.

📦 Optional: Update Requirements
If you add new libraries:

bash
Copy
Edit
pip freeze > requirements.txt
🛣️ Roadmap & Improvements
Add retry logic for API failures

Store data in a real database

Enhance UI with charts or tables

Support user login and saved reviews

Add support for video or audio reviews (future Gemini features)

📜 License
This MVP is built for the Khapey internal prototype. Respect usage limits and terms of the Google Gemini API.