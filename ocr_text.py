import cv2
import pytesseract
import json
from datetime import datetime
import os

# Set the path to Tesseract executable (adjust if necessary)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows example

def read_text(img):
    # Load the preprocessed denoised image
    image_path = 'denoised_image.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    print(f"Loaded image from {image_path} with shape {image.shape} (width: {image.shape[1]}, height: {image.shape[0]})")
    print(f"Image stats - Mean: {image.mean():.2f}, Std: {image.std():.2f}, Min: {image.min()}, Max: {image.max()}")

    # Step 1: Attempt OCR with multiple configurations to maximize text detection
    print("Attempting OCR with various page segmentation modes to extract all text...")
    configs = [
        r'--oem 3 --psm 3 -l eng',   # Fully automatic page segmentation
        r'--oem 3 --psm 6 -l eng',   # Single uniform block of text
        r'--oem 3 --psm 11 -l eng',  # Sparse text, find as much as possible
        r'--oem 3 --psm 1 -l eng'    # Automatic page segmentation with OSD
    ]
    paragraph_text = ""
    successful_config = "None"

    for config in configs:
        print(f"\nTrying OCR with config: {config}")
        text = pytesseract.image_to_string(image, config=config).strip()
        if text:
            paragraph_text = text
            successful_config = config
            print(f"Text detected with config {config} (length: {len(paragraph_text)} characters):")
            print(f"'{paragraph_text[:50]}{'...' if len(paragraph_text) > 50 else ''}'")
            break
        else:
            print(f"No text detected with config {config}")

    # If no text found, try detailed word-level detection as fallback
    if not paragraph_text:
        print("\nNo paragraph text found. Attempting word-level detection as fallback...")
        ocr_data = pytesseract.image_to_data(image, config=r'--oem 3 --psm 11 -l eng', output_type=pytesseract.Output.DICT)
        words = [word.strip() for word in ocr_data['text'] if word.strip() and float(ocr_data['conf'][ocr_data['text'].index(word)]) > 10]
        if words:
            paragraph_text = " ".join(words)
            successful_config = "Word-level fallback (--oem 3 --psm 11)"
            print(f"Fallback succeeded - Detected {len(words)} words, combined into paragraph (length: {len(paragraph_text)}):")
            print(f"'{paragraph_text[:50]}{'...' if len(paragraph_text) > 50 else ''}'")
        else:
            print("Fallback failed - No words detected even at word level.")

    # Step 2: Compile metadata
    print("\nCompiling metadata for JSON output...")
    metadata = {
        "image_file": image_path,
        "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "image_dimensions": {"width": image.shape[1], "height": image.shape[0]},
        "image_stats": {
            "mean_intensity": float(image.mean()),
            "std_dev": float(image.std()),
            "min_value": int(image.min()),
            "max_value": int(image.max())
        },
        "text_length": len(paragraph_text),
        "ocr_engine": "Tesseract",
        "configs_tried": configs,
        "successful_config": successful_config
    }

    output_data = {
        "metadata": metadata,
        "paragraph_text": paragraph_text
    }

    # Step 3: Save to JSON file
    json_file = 'paragraph_text.json'
    print(f"Saving results to {json_file}...")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
    print(f"JSON file saved with paragraph of {len(paragraph_text)} characters.")

    # Step 4: Diagnostic visualization
    visual_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if paragraph_text:
        cv2.putText(visual_image, "Text Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(visual_image, "No Text Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.imwrite('ocr_diagnostic.png', visual_image)
    print("Diagnostic image saved as 'ocr_diagnostic.png'.")

    # Summary
    if paragraph_text:
        print(f"Summary: Extracted paragraph with {len(paragraph_text)} characters using {successful_config}")
    else:
        print("Summary: No text detected despite multiple attempts.")

    return paragraph_text