
---

````markdown
# Khapey: Food Review App

## Overview
Khapey is a Streamlit-based web application designed for a Pakistani food review platform. It allows users to submit reviews for restaurants, including 1-3 images, a text review, and 1-5 star ratings for food quality, food quantity, staff, dining experience, ambience, and overall experience. The app uses the Google Gemini 2.5 Flash-Lite preview model (`gemini-2.5-flash-lite-preview-06-17`) to analyze images and text, extracting details like cuisine type, lighting quality, and sentiment, with cultural sensitivity to the Pakistani/Muslim community. The output is a structured JSON file, displayed in the UI and available for download.

---

## Features
- **Image Upload**: Upload 1-3 images (JPEG or PNG) of food, ambience, or other relevant scenes.
- **Text Review**: Write a free-text review of the dining experience.
- **Ratings**: Provide 1-5 star ratings for food quality, food quantity, staff, dining experience, ambience, and overall experience.
- **Gemini API Integration**: Uses the Gemini 2.5 Flash-Lite preview model to analyze images and text, extracting:
  - **Image Analysis**: Cuisine type, lighting quality score (1-5), face/food detection, ambience, image quality (1-5), resolution, and item description.
  - **Text Analysis**: Sentiment (positive, negative, neutral), key topics, and specific mentions (e.g., dishes, staff).
- **Output**: Generates a JSON file with analysis results and ratings, displayed in the UI and downloadable.
- **Cultural Sensitivity**: Ensures analysis respects Pakistani/Muslim context (e.g., halal status, cultural dining norms).
- **Debugging**: Lists available Gemini models to verify API access.

---

## Prerequisites
- Python 3.8+
- Google Gemini API key with access to `gemini-2.5-flash-lite-preview-06-17`
- Internet connection for API calls

---

## 🔧 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Omair22f3651/Khapey-MVP.git
cd Khapey-MVP
````

### 2. Create and Activate Virtual Environment

#### 🪟 On Windows (CMD or PowerShell):

```bash
python -m venv venv
venv\Scripts\activate
```

#### 🍎 On macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:

* streamlit
* google-generativeai
* pillow
* pandas
* nomic
* python-dotenv

> You can inspect or modify `requirements.txt` as needed.

---

### 4. Set Up Gemini API Key

#### ✅ Option A: Use `.env` file (Recommended)

Create a file named `.env` in the root directory:

```
GEMINI_API_KEY=your-real-gemini-api-key
```

Make sure your Python code includes:

```python
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
```

#### 🔁 Option B: Set it as an environment variable manually

##### On macOS/Linux:

```bash
export GEMINI_API_KEY=your-real-gemini-api-key
```

##### On Windows (CMD):

```cmd
set GEMINI_API_KEY=your-real-gemini-api-key
```

##### On Windows (PowerShell):

```powershell
$env:GEMINI_API_KEY="your-real-gemini-api-key"
```

---

### 5. Run the Application

```bash
streamlit run review.py
```

This will launch the app at:
📍 `http://localhost:8501`

---

### 🧹 Optional: Freeze Dependencies

If you add or update packages, update the requirements with:

```bash
pip freeze > requirements.txt
```

---

## Gemini API Prompt

The app uses a detailed prompt to instruct the Gemini API, ensuring:

* **Strict Analysis Protocols**: Only uses visible image/text data.
* **Data Extraction Matrix**: Extracts different data based on image type.
* **Output Format**: Always returns clean JSON with relevant keys:

  * `image_analysis[]`
  * `text_analysis{}`
  * `ratings{}`
  * `timestamp`

---

## Troubleshooting

### ❌ Error: Model Not Found

* Ensure `gemini-2.5-flash-lite-preview-06-17` is listed in the "Available Models" section.
* Verify your API key has access to Gemini multimodal preview models.

### ❌ Error: JSON Parsing Failed

* If Gemini returns wrapped markdown or text:

````python
response_text = response_text.strip()
if response_text.startswith("```json") and response_text.endswith("```"):
    response_text = response_text[7:-3].strip()
result = json.loads(response_text)
````

### ❌ API Errors

* Check if your API key is valid and has correct scopes.
* Ensure internet connection is available.

### 🖼️ Image Resolution Issues

* Make sure your uploaded files are valid `.jpg` or `.png` formats.
* Pillow (`PIL`) is used to extract dimensions.

---

## Notes

* **Model**: This uses `gemini-2.5-flash-lite-preview-06-17`, designed for low-latency multimodal tasks.
* **Cultural Context**: Built to respect Pakistani/Muslim norms (e.g., halal detection).
* **Security**: Do not hardcode the API key. Always use environment variables.
* **JSON Output**: Includes extracted metadata, user input, and a UTC timestamp.

---

## 🚀 Future Improvements

* Add retry logic for Gemini API failures.
* Store review data to a database (PostgreSQL, Firebase, etc).
* Enhance UI with colored or interactive tables.
* Add video review support if Gemini adds support.
* Enable user authentication and profile history.

---

## 📄 License

This project is for internal use and MVP purposes for the *Khapey* app.
Ensure compliance with Google Gemini API Terms of Service.

```
