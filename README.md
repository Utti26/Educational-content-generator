# Educational-content-generator
This project provides an API for generating and refining educational content tailored to a specific subject or grade level. It uses state-of-the-art natural language processing (NLP) models to generate content, refine it for readability and coherence, and mitigate biases to ensure inclusivity.

Features
Content Generation:

Generate educational content based on user prompts using a pre-trained transformer model (e.g., google/flan-t5-large).

Content Refinement:

Improve readability by simplifying sentences.

Check coherence using sentence embeddings.

Bias Detection and Mitigation:

Detect biases in the generated content using a hate speech detection model.

Replace biased terms with neutral alternatives.

API Endpoint:

Expose the pipeline as a REST API using FastAPI.

Setup
Prerequisites
Python 3.9 or higher

Docker (optional, for containerization)

AWS CLI (optional, for cloud deployment)

Install Dependencies
Clone the repository:

bash
Copy
git clone https://github.com/your-repo/educational-content-api.git
cd educational-content-api
Install Python dependencies:

bash
Copy
pip install -r requirements.txt
Download the spaCy model:

bash
Copy
python -m spacy download en_core_web_sm
Usage
Run the Application Locally
Start the FastAPI server:

bash
Copy
uvicorn main:app --host 0.0.0.0 --port 8000
Access the API at http://localhost:8000.

Example API Request
Send a POST request to /generate-content with a JSON body:

json
Copy
{
  "prompt": "Explain the water cycle for 4th graders."
}
Example Response
json
Copy
{
  "content": "The water cycle is the process by which water moves through the Earth's atmosphere, oceans, and land. It includes evaporation, condensation, precipitation, and collection."
}
Deployment
Docker Deployment
Build the Docker image:

bash
Copy
docker build -t educational-content-api .
Run the Docker container:

bash
Copy
docker run -d -p 8000:8000 --name edu-content-api educational-content-api
Access the API at http://localhost:8000.

Cloud Deployment (AWS Elastic Beanstalk)
Install AWS CLI and Elastic Beanstalk CLI:

bash
Copy
pip install awscli awsebcli
Initialize Elastic Beanstalk:

bash
Copy
eb init -p python-3.9 educational-content-api
Create an Elastic Beanstalk environment:

bash
Copy
eb create educational-content-env
Deploy the application:

bash
Copy
eb deploy
Access the API using the URL provided by Elastic Beanstalk.

Project Structure
Copy
educational-content-api/
├── main.py                # FastAPI application and pipeline logic
├── Dockerfile             # Docker configuration for containerization
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
API Endpoints
POST /generate-content
Generate educational content based on a user prompt.

Request Body
json
Copy
{
  "prompt": "Explain photosynthesis for 5th graders."
}
Response
json
Copy
{
  "content": "Photosynthesis is the process by which plants make their own food using sunlight, water, and carbon dioxide."
}
GET /
Root endpoint to check if the API is running.

Response
json
Copy
{
  "message": "Educational Content Generation API"
}
Technologies Used
Python: Primary programming language.

FastAPI: Web framework for building the API.

Hugging Face Transformers: Pre-trained NLP models for content generation.

spaCy: Linguistic analysis for readability improvement.

Sentence Transformers: Sentence embeddings for coherence checking.

Docker: Containerization for deployment.

AWS Elastic Beanstalk: Cloud deployment platform.
