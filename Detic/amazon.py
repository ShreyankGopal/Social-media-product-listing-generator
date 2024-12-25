import google.generativeai as genai
import json
from typing import Dict, Any
import time
from datetime import datetime

class ProductExtractor:
    def __init__(self, api_key: str):
        """Initialize with your Google API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        
    def query_model(self, prompt: str, retries: int = 3, delay: int = 2) -> str:
        """
        Query the Gemini model with retry mechanism
        """
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                if response.text:
                    return response.text.strip()
                    
            except Exception as e:
                if attempt == retries - 1:
                    print(f"Error querying Gemini API: {str(e)}")
                    return "Error in extraction"
                time.sleep(delay * (attempt + 1))  # Exponential backoff
                
        return "Failed to get response"

    def extract_product_info(self, text: str) -> Dict[str, Any]:
        """
        Extract product information using specific prompts
        """
        extraction_prompts = {
            'price': """
            Task: Extract the current selling price from the text.
            Rules: 
            - Return only the final price with currency symbol
            - Return 'Not specified' if none found
            Text: """,
            
            'offer': """
            Task: Extract any discount or promotional offer.
            Rules:
            - Include the amount or percentage saved
            - Return 'No offer available' if none found
            Text: """,
            
            'product_name': """
            Task: Extract the main product name.
            Rules:
            - Include brand and model if mentioned
            Text: """,
            
            'country_of_origin': """
            Task: Extract the country of origin for the product from the text.
            Rules:
            - Return only the country name (e.g., 'Italy', 'USA')
            - If not explicitly mentioned, return 'Not specified'
            Text: """,
    
            'sizes': """
            Task: Extract the available sizes for the product from the text.
            Rules:
            - Return all sizes mentioned (e.g., '15"x12"x5"', 'Large', 'Medium', 'small')
            - If no sizes are mentioned, return 'Not specified'
            Text: """,
            
            'title': """
            Task: What is the product about?.
            Rules:
            - must be 1 sentence only
            Text: """,
            
            'color': """
            Task: Extract all available colors mentioned for the product.
            Rules:
            - List all colors mentioned
            - If multiple colors exist, separate them with commas
            - Return 'Not specified' if no colors are mentioned
            Text: """,

            'material': """
            Task: Extract the materials used in the product.
            Text: """,

            'features': """
            Extract the features and capabilities in this text. If there are many features seperate them by semi-colons.
            Text: """,
            
            'technology': """
            Task: Extract all technological features and capabilities mentioned.
            Rules:
            - List all technological features (e.g., bluetooth, speakers, apps etc)
            - Include smart features and connectivity options
            - Include sensors and monitoring capabilities
            - List any software or app integrations
            - Specify compatibility with other devices/systems
            - Return 'No technology features mentioned' if none found
            Text: """,
            
            'target': """
            Task: From the following advertisement text, determine who the target audience might be for this product. write in 2 to 3 words.
            Text: """
        }
        
        results = {}
        print("Starting extraction process...")
        
        for field, prompt in extraction_prompts.items():
            print(f"Extracting {field}...")
            full_prompt = f"{prompt}\n{text}"
            results[field] = self.query_model(full_prompt)
            time.sleep(0.5)  # Small delay to prevent rate limiting
            
        description_prompt = f"""
        Using the details provided, generate a compelling product description of max 50 words, suitable for an Amazon listing. Include the product's main features, benefits, and any unique aspects that make it appealing. Ensure the tone is engaging and informative.

        Product Name: {results['product_name']}
        Features: {results['features']}
        Target Audience and use case: {results['target']}
        """
        results['description'] = self.query_model(description_prompt,)
            
        return results

def mainLLM(txt):
    # Your Google API key
    API_KEY = "xyzjhv"  # Replace with your actual Google API key
    
    # Initialize extractor
    extractor = ProductExtractor(API_KEY)
    
    # Process text and get results
    try:
        print("\nProcessing product description...")
        results = extractor.extract_product_info(txt)
        results['paragraph'] = txt
        # Print results in a formatted way
        print("\nExtracted Information:")
        print(json.dumps(results, indent=2))
        return results
        
    except Exception as e:
        print(f"Error during processing: {str(e)}")