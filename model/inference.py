
import torch
from torchvision import transforms
from model import TrashDetectionModel
from PIL import Image

def predict(image_path, model_path='trash_detection_model.pth', num_trash_classes=60):
    """
    Performs inference on a single image.
    """
    # Load the model
    model = TrashDetectionModel(num_trash_classes=num_trash_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Define transformations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")
    image = data_transform(image).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        person_bbox, person_logits, face_embedding, trash_logits, disposal_logits = model(image)

    # Post-process the output
    # This is a simplified post-processing step. In a real scenario, you would need to apply non-maximum suppression for bounding boxes and convert logits to probabilities.
    person_class = torch.argmax(person_logits, dim=1).item()
    trash_class = torch.argmax(trash_logits, dim=1).item()
    disposal_class = torch.argmax(disposal_logits, dim=1).item()

    return {
        'person_bbox': person_bbox.tolist(),
        'person_class': person_class,
        'face_embedding': face_embedding.tolist(),
        'trash_class': trash_class,
        'disposal_class': disposal_class
    }

if __name__ == '__main__':
    image_path = 'c:\\Users\\KUNAL SHEDGE\\Desktop\\New folder\\trash_detect\\data\\unified_dataset\\images\\batch_1_000003.jpg'
    predictions = predict(image_path)
    print(predictions)
