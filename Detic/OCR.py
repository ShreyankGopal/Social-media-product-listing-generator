# Import necessary libraries
from paddleocr import PaddleOCR
from PIL import Image

def OCR(img_path):
    image = Image.open(img_path)

# Initialize PaddleOCR (you can specify the language; 'en' for English)
    ocr = PaddleOCR(use_angle_cls=True, lang='en')  # use_angle_cls helps with tilted text

# Use PaddleOCR to extract text from the image
    result = ocr.ocr(img_path, cls=True)

# Process and print the results
    final_text=""
    for line in result:
        for word_info in line:
            text = word_info[1][0]  # The text is in the second element
        
            final_text=final_text+text
            final_text=final_text+"\n"
    print(final_text)
    return final_text

# If your image is saved in your local environment, upload it to Colab
# from google.colab import files
# uploaded = files.upload()

# Load the image you uploaded (replace 'test1.png' with your filename)

