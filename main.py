# Install required libraries
# !pip install transformers torch spacy sentence-transformers scikit-learn fastapi uvicorn

# Download spaCy model
# !python -m spacy download en_core_web_sm

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Define request model
class PromptRequest(BaseModel):
    prompt: str

# Content Generation Module
class ContentGenerator:
    def __init__(self, model_name="google/flan-t5-large"):
        self.generator = pipeline("text2text-generation", model=model_name)

    def generate_content(self, prompt, max_length=200):
        """
        Generate educational content based on a user prompt.
        """
        generated_text = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return generated_text[0]['generated_text']

# Content Refinement Module
class ContentRefiner:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentence_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def improve_readability(self, text):
        """
        Simplify sentences for better readability.
        """
        doc = self.nlp(text)
        simplified_sentences = [sent.text for sent in doc.sents if len(sent) < 20]  # Keep short sentences
        return " ".join(simplified_sentences)

    def check_coherence(self, text):
        """
        Check coherence of the text using sentence embeddings.
        """
        sentences = [sent.text for sent in self.nlp(text).sents]
        if len(sentences) < 2:
            return 1.0  # Single sentence is always coherent

        embeddings = self.sentence_model.encode(sentences)
        similarity_matrix = cosine_similarity(embeddings)
        np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
        coherence_score = similarity_matrix.mean()  # Average similarity between sentences
        return coherence_score

# Bias Detection and Mitigation Module
class BiasDetector:
    def __init__(self):
        self.classifier = pipeline("text-classification", model="Hate-speech-CNERG/dehatebert-mono-english")

    def detect_bias(self, text):
        """
        Detect bias in the text using a hate speech detection model.
        """
        result = self.classifier(text)
        return result[0]['label'], result[0]['score']

    def mitigate_bias(self, text):
        """
        Replace biased terms with neutral alternatives.
        """
        # Example: Replace biased terms (this can be expanded)
        replacements = {
            "he": "they",
            "she": "they",
            "his": "their",
            "her": "their"
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

# Educational Content Pipeline
class EducationalContentPipeline:
    def __init__(self):
        self.generator = ContentGenerator()
        self.refiner = ContentRefiner()
        self.detector = BiasDetector()

    def run_pipeline(self, prompt):
        # Step 1: Generate content
        content = self.generator.generate_content(prompt)
        print("Generated Content:\n", content)

        # Step 2: Refine content
        refined_content = self.refiner.improve_readability(content)
        coherence_score = self.refiner.check_coherence(refined_content)
        print("Refined Content:\n", refined_content)
        print("Coherence Score:", coherence_score)

        # Step 3: Detect and mitigate bias
        bias_label, bias_score = self.detector.detect_bias(refined_content)
        mitigated_content = self.detector.mitigate_bias(refined_content)
        print("Bias Detection:", bias_label, bias_score)
        print("Mitigated Content:\n", mitigated_content)

        return mitigated_content

# Initialize the pipeline
pipeline = EducationalContentPipeline()

# FastAPI Endpoints
@app.post("/generate-content")
async def generate_content(request: PromptRequest):
    try:
        # Run the pipeline
        final_content = pipeline.run_pipeline(request.prompt)
        return {"content": final_content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {"message": "Educational Content Generation API"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)