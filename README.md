# Educational Content Generation API

This project provides an API for generating, refining, and bias-detecting educational content using state-of-the-art Natural Language Processing (NLP) models. The API is built using FastAPI and leverages pre-trained transformer models for text generation, readability improvement, and bias mitigation.

## Features
- **Content Generation**: Uses Google's FLAN-T5 model to generate educational content from user prompts.
- **Content Refinement**: Enhances readability and checks text coherence using spaCy and Sentence Transformers.
- **Bias Detection & Mitigation**: Identifies and mitigates biased language using a hate speech detection model.

## Technologies Used
- FastAPI
- Uvicorn
- Transformers
- PyTorch
- spaCy
- Sentence Transformers
- Scikit-learn

## Installation
### Prerequisites
Ensure you have Python 3.8 or later installed on your system.

### Steps
1. Clone the repository:
   ```sh
   git clone <repository_url>
   cd <repository_name>
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the required spaCy model:
   ```sh
   python -m spacy download en_core_web_sm
   ```

## Running the Application
Start the FastAPI server:
```sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API Endpoints
### Generate Content
**Endpoint:**
```
POST /generate-content
```
**Request Body:**
```json
{
  "prompt": "Explain the concept of gravity"
}
```
**Response:**
```json
{
  "content": "Gravity is a force that attracts two bodies toward each other..."
}
```

### Root Endpoint
**Endpoint:**
```
GET /
```
**Response:**
```json
{
  "message": "Educational Content Generation API"
}
```

## Docker Support
To run the API in a Docker container:
```sh
docker build -t content-api .
docker run -p 8000:8000 content-api
```

## Future Improvements
- Support for multiple languages
- Advanced bias detection using LLMs
- Enhanced content personalization

## License
This project is licensed under the MIT License.

