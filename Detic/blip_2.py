from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

class AdImageAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load processor and model
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b", 
            torch_dtype=torch.float16
        )
        self.model.to(self.device)
        
    def load_image(self, image_path_or_url):
        """Load image from local path or URL"""
        try:
            if image_path_or_url.startswith('http'):
                image = Image.open(requests.get(image_path_or_url, stream=True).raw)
            else:
                image = Image.open(image_path_or_url)
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def analyze_image(self, image):
        """Analyze the image with four different prompts"""
        prompts = [
            "Question: What is only the main color of the product in this advertisement? Answer:",
            "Question: What product is being advertised in this image? Answer:",
            "Question: What brand is being advertised in this image? Answer:",
            "Question: Are there any discounts, offers, or promotional deals visible in this image? If yes, state only the offer amount or deal. If no, say 'No offers'. Answer:"
        ]
        
        results = {}
        
        for prompt in prompts:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                num_beams=5,
                min_length=1,
                length_penalty=1.0
            )
            
            # Decode the generated text
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)
            results[prompt.split('?')[0].replace('Question: ', '')] = generated_text.strip()
            
        return results

# Example usage
if __name__ == "__main__":
    analyzer = AdImageAnalyzer()
    
    # Example with a URL
    url = "https://i.pinimg.com/736x/67/98/01/679801d5e7bb32aa5097eebeddb3e2af.jpg"
    image = analyzer.load_image(url)
    
    if image:
        results = analyzer.analyze_image(image)
        for question, answer in results.items():
            print(f"{question}:")
            print(f"- {answer}\n")