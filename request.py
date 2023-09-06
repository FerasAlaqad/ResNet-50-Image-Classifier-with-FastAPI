import os
import requests
import json

# Define the URL of your FastAPI server
url = "http://localhost:5000/predict"  # Update this URL if your server is running on a different host or port

# Load ImageNet class labels
imagenet_labels_url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
labels_json = requests.get(imagenet_labels_url)
labels = json.loads(labels_json.text)

# Define the folder containing the images
image_folder = "images"

# Loop through each file in the folder
for filename in os.listdir(image_folder):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        image_path = os.path.join(image_folder, filename)
        
        # Open and send the image file
        with open(image_path, "rb") as image_file:
            files = {"image": (filename, image_file, "image/png")}  # Assuming all images are PNG format
            response = requests.post(url, files=files)

        # Check the response
        if response.status_code == 200:
            result = response.json()
            predicted_label = result["predicted_label"]
            predicted_class_name = labels[predicted_label]
            print(f"Image: {filename}, Predicted Label: {predicted_class_name}")
        else:
            print(f"Error predicting {filename}: {response.status_code} - {response.text}")
