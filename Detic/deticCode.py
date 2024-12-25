import os
import sys

# Add the CenterNet2 path since we're running from Detic directory
current_dir = os.getcwd()  # This will be the Detic directory
centernet_path = os.path.join(current_dir, 'third_party', 'CenterNet2')

# Add paths
sys.path.insert(0, centernet_path)
sys.path.insert(0, current_dir)

import cv2
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from transformers import pipeline
from datetime import datetime

# Import the required modules
from centernet.config import add_centernet_config
from detic.config import add_detic_config
from detic.modeling.utils import reset_cls_test
matplotlib.use('agg')
# Setup detectron2 logger
setup_logger()

def setup_detectron():
    """Setup Detectron2 configuration"""
    cfg = get_cfg()
    add_centernet_config(cfg)
    add_detic_config(cfg)
    cfg.merge_from_file("configs/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml")
    cfg.MODEL.WEIGHTS = 'https://dl.fbaipublicfiles.com/detic/Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Lower threshold for initial detection
    cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
    cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = False  # Changed to False to detect multiple classes
    cfg.MODEL.DEVICE = 'cpu'  # Change to 'cuda' if using GPU
    return DefaultPredictor(cfg)

def setup_vocabulary(vocabulary='lvis'):
    """Setup vocabulary and metadata"""
    BUILDIN_CLASSIFIER = {
        'lvis': 'datasets/metadata/lvis_v1_clip_a+cname.npy',
        'objects365': 'datasets/metadata/o365_clip_a+cnamefix.npy',
        'openimages': 'datasets/metadata/oid_clip_a+cname.npy',
        'coco': 'datasets/metadata/coco_clip_a+cname.npy',
    }

    BUILDIN_METADATA_PATH = {
        'lvis': 'lvis_v1_val',
        'objects365': 'objects365_v2_val',
        'openimages': 'oid_val_expanded',
        'coco': 'coco_2017_val',
    }

    metadata = MetadataCatalog.get(BUILDIN_METADATA_PATH[vocabulary])
    classifier = BUILDIN_CLASSIFIER[vocabulary]
    num_classes = len(metadata.thing_classes)
    return metadata, classifier, num_classes

def process_image(image_path, predictor, metadata):
    """Process image for object detection and cropping"""
    # Create output directories with timestamp to avoid overwriting
    output_dir = f'static/images'
    cropped_dir = os.path.join(output_dir, 'cropped_images')
    os.makedirs(cropped_dir, exist_ok=True)

    # Read image
    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Could not read image at {image_path}")

    # Run detection
    outputs = predictor(im)
    instances = outputs["instances"]
    
    # Filter instances based on confidence score
    high_conf_mask = instances.scores > 0.65
    relevant_instances = instances[high_conf_mask]

    # Get class names for detected objects
    class_ids = relevant_instances.pred_classes.cpu().numpy()
    class_names = [metadata.thing_classes[class_id] for class_id in class_ids]
    scores = relevant_instances.scores.cpu().numpy()

    print(f"Found {len(relevant_instances)} objects with confidence > 0.72")
    
    # Visualize all detections
    v = Visualizer(im[:, :, ::-1], metadata)
    out = v.draw_instance_predictions(relevant_instances.to("cpu"))
    plt.figure(figsize=(10, 10))
    plt.imshow(out.get_image())
    plt.axis('off')
    plt.savefig(os.path.join(output_dir, 'detection_visualization.jpg'))
    plt.close()

    # Process each detection
    boxes = relevant_instances.pred_boxes.tensor.cpu().numpy()
    cropped_paths = []
    
    for i, (box, class_name, score) in enumerate(zip(boxes, class_names, scores)):
        try:
            # Add padding to crop
            print(f'score: {score}')
            x1, y1, x2, y2 = map(int, box)
            padding = 10
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(im.shape[1], x2 + padding)
            y2 = min(im.shape[0], y2 + padding)

            # Crop
            cropped_region = im[y1:y2, x1:x2]
            if cropped_region.size == 0:
                print(f"Warning: Empty crop for {class_name}")
                continue

            # Save cropped image with unique identifier
            crop_filename = f'{class_name}_{i+1}_conf{score:.2f}.jpg'
            crop_path = os.path.join(cropped_dir, crop_filename)
            cv2.imwrite(crop_path, cropped_region)
            
            print(f"Saved {class_name} (confidence: {score:.2f}) to {crop_path}")
            cropped_paths.append(crop_path)

        except Exception as e:
            print(f"Error processing detection {i} ({class_name}): {str(e)}")
            continue

    return cropped_paths
def remove_background(image_paths, output_dir):
    """Remove background from images using RMBG-1.4"""
    print("Initializing background removal model...")
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True)
    
    no_bg_dir = os.path.join(output_dir, 'without_background')
    os.makedirs(no_bg_dir, exist_ok=True)
    
    for image_path in image_paths:
        try:
            # Get original filename without directory
            filename = os.path.basename(image_path)
            
            # Remove background
            pillow_image = pipe(image_path)
            pillow_image = pillow_image.convert("RGBA")
            opencv_image = np.array(pillow_image)

            if opencv_image.shape[-1] == 4:
                rgb, alpha = opencv_image[:, :, :3], opencv_image[:, :, 3]
                white_background = np.ones_like(rgb, dtype=np.uint8) * 255
                opencv_image = np.where(alpha[:, :, None] == 0, white_background, rgb)

            # Convert to BGR for OpenCV to preserve original colors
            opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
            
            # Save with same filename pattern in no_bg directory
            save_path = os.path.join(no_bg_dir, f'no_bg_{filename}')
            cv2.imwrite(save_path, opencv_image)
            print(f"Saved background-removed image to {save_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            continue

def mainDetic(img_path):
    """Main execution function"""
    
    # Setup
    predictor = setup_detectron()
    metadata, classifier, num_classes = setup_vocabulary('lvis')
    reset_cls_test(predictor.model, classifier, num_classes)

    # Process image
    image_path = img_path
    
    print("Starting object detection and cropping...")
    cropped_paths = process_image(image_path, predictor, metadata)
    print(cropped_paths)
    # if cropped_paths:
    #     print("\nStarting background removal...")
    #     remove_background(cropped_paths, output_dir)
    #     print("\nProcessing complete!")
    # else:
    #     print("No objects were detected with confidence > 0.72")
    return cropped_paths


