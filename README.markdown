# Khapey: Food Review App

## Overview
Khapey is a Streamlit-based web application designed for a Pakistani food review platform. It allows users to submit reviews for restaurants, including 1-3 images, a text review, and 1-5 star ratings for food quality, food quantity, staff, dining experience, ambience, and overall experience. The app uses the Google Gemini 2.5 Flash-Lite preview model (`gemini-2.5-flash-lite-preview-06-17`) to analyze images and text, extracting details like cuisine type, lighting quality, and sentiment, with cultural sensitivity to the Pakistani/Muslim community. The output is a structured JSON file, displayed in the UI and available for download.

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

## Prerequisites
- Python 3.8+
- Google Gemini API key with access to `gemini-2.5-flash-lite-preview-06-17`
- Internet connection for API calls

## Installation
1. **Clone the Repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd Khapey
   ```

2. **Install Dependencies**:
   Install the required Python packages using pip:
   ```bash
   pip install streamlit google-generativeai pillow
   ```

3. **Set Up the Gemini API Key**:
   - Replace `"your-gemini-api-key"` in `food_review_app.py` with your actual Gemini API key, or
   - Set the API key as an environment variable:
     ```bash
     export GOOGLE_API_KEY="your-gemini-api-key"
     ```
     If using the environment variable, remove the `genai.configure(api_key="your-gemini-api-key")` line from the code.

## Usage
1. **Run the App**:
   ```bash
   streamlit run food_review_app.py
   ```
   This starts a local web server, typically at `http://localhost:8501`.

2. **Interact with the App**:
   - **Upload Images**: Select 1-3 images (JPEG or PNG) via the file uploader.
   - **Write Review**: Enter a text review in the provided text area.
   - **Rate Experience**: Use sliders to rate food quality, food quantity, staff, dining experience, ambience, and overall experience (1-5 stars).
   - **Submit**: Click the "Submit Review" button to process the input.

3. **View Results**:
   - The app displays the uploaded images and the analysis result in JSON format.
   - Download the JSON output by clicking the "Download JSON" button.
   - Check the "Available Models" section to confirm the Gemini model is accessible.

## Project Structure
- `food_review_app.py`: Main Streamlit app script, containing the UI and Gemini API integration.
- `README.md`: This file, providing setup and usage instructions.

## Gemini API Prompt
The app uses a detailed prompt to instruct the Gemini API, ensuring:
- **Strict Analysis Protocols**: Extracts data only from visible image content, prioritizes accuracy, and respects cultural context.
- **Data Extraction Matrix**: Defines what to extract based on image type (e.g., food close-up, ambience shot).
- **Output Format**: Returns a JSON object with image analysis, text analysis, ratings, and a timestamp.

## Troubleshooting
- **Error: Model Not Found**:
  - Check the "Available Models" section in the UI to confirm `gemini-2.5-flash-lite-preview-06-17` is listed.
  - Ensure your API key has access to the preview model. Contact your API provider or Google Cloud support if needed.
- **Error: JSON Parsing Failure**:
  - If the app displays a raw response, inspect it to determine the format (e.g., plain text, markdown).
  - Modify the `call_gemini_api` function to handle non-JSON responses, e.g., strip markdown code fences:
    ```python
    response_text = response_text.strip()
    if response_text.startswith("```json") and response_text.endswith("```"):
        response_text = response_text[7:-3].strip()
    result = json.loads(response_text)
    ```
- **API Call Errors**:
  - Verify the API key is correct and has sufficient permissions.
  - Ensure an internet connection is available.
- **Image Resolution Issues**:
  - The app uses Pillow to extract image resolution. Ensure uploaded images are valid JPEG or PNG files.

## Notes
- **Model**: The app uses `gemini-2.5-flash-lite-preview-06-17`, a preview model optimized for low-latency multimodal tasks (text and images). Verify availability with your API key.
- **Cultural Context**: The prompt ensures analysis respects Pakistani/Muslim dining norms, such as halal status.
- **Output**: The JSON output includes image analysis, text analysis, user ratings, and a timestamp in ISO format.
- **Security**: Store the API key securely (e.g., as an environment variable) to avoid exposing it in the code.

## Future Improvements
- Add retry logic for API call failures.
- Support additional image formats or video uploads (if Gemini API allows).
- Enhance UI with styled output (e.g., formatted tables for analysis).
- Integrate with a backend to store reviews.

## License
This project is for internal use and tailored to the Khapey app. Ensure compliance with Google Gemini API terms of service.