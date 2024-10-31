# Import necessary libraries
from ultralytics import YOLO
import gradio as gr
import cv2

# Load the YOLOv8 model
# Replace "path/to/your/yolov8-weights.pt" with the path to your trained model weights
model = YOLO("best.pt")

# Define the prediction function
def predict(image):
    # Run inference on the input image
    results = model.predict(image)

    # Retrieve the annotated image with bounding boxes and labels drawn
    annotated_image = results[0].plot()  # This will draw boxes on the image
    
    return annotated_image

# Create the Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Image(type="numpy"),
    title="Blood Cell Count Detection",
    description="Upload an image of blood cells, and the model will detect and count the cells."
)

# Launch the Gradio app
interface.launch()
