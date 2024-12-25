from flask import Flask, render_template, request, redirect, url_for
import os
from deticCode import mainDetic
from amazon import mainLLM
from OCR import OCR
import shutil
import time

app = Flask(__name__)

# Shared data for storing results
image_results = []
llm_result = None

def process_images(images, save_dir):
    global image_results
    
    result = []
    ocr_text = []
    for img in images:
        # Generate a unique file name for each image
        image_path = os.path.join(save_dir, img.filename)
        print("printing image path")
        print(image_path)
       
        try:
            img.save(image_path)  # Save the image
            ocr = OCR(image_path)
            ocr_text.append(ocr)
            detic_result = mainDetic(image_path)
            for i in detic_result:  # Process image with Detic
                result.append(i)
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
        finally:
            # Cleanup saved image
            if os.path.exists(image_path):
                os.remove(image_path)

    # Return local results
    return ocr_text, result

def process_paragraph(paragraph):
    global llm_result
    try:
        llm_result = mainLLM(paragraph)
    except Exception as e:
        print(f"Error processing paragraph: {e}")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/process', methods=['POST'])
def process():
    global image_results, llm_result

    # Reset results for this request
    image_results = []
    llm_result = None
    
    # Get uploaded files and text input
    images = request.files.getlist('images')
    paragraph = request.form.get('paragraph')
    original_para=paragraph
    # Directory to save the uploaded images
    save_dir = "../sample-images"
    os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist

    # Process images and paragraph sequentially
    ocr_text, local_result = process_images(images, save_dir)
    space = " "
    paragraph=paragraph+'\n'
    for txt in ocr_text:
        paragraph = paragraph + txt + '\n'
    print(paragraph)   
    process_paragraph(paragraph)

    print("Uploaded images and their results:", local_result)
    print("Paragraph result:", llm_result)

    # Redirect to the listing page with results
    return redirect(url_for('listing', local_result=local_result,original_para=original_para))

@app.route('/listing', methods=['GET'])
def listing():
    global llm_result

    # Retrieve local_result from the query parameters
    local_result = request.args.getlist('local_result')
    original_para = request.args.getlist('original_para')
    print(local_result)
    for i in range(len(local_result)):
        local_result[i] = local_result[i][7:]
    print(llm_result)
    llm_result['paragraph'] = original_para[0]
    print(f'printing local result {local_result}')

    # Render the template
    response = render_template('list.html', llm_result=llm_result, local_result=local_result)
    # time.sleep(5.5)
    # # Directory to clear
    # static_images_dir = "static/images/cropped_images"
    # if os.path.exists(static_images_dir):
    #     for file_name in os.listdir(static_images_dir):
    #         file_path = os.path.join(static_images_dir, file_name)
    #         try:
    #             if os.path.isfile(file_path):
    #                 os.remove(file_path)
    #         except Exception as e:
    #             print(f"Error deleting file {file_path}: {e}")
    # os.remove('static/images/detection_visualization.jpg')
    return response

if __name__ == '__main__':
    app.run(debug=True, port=5003, use_reloader=True)
