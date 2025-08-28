
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from model import TrashDetectionModel
from dataset import TrashDetectionDataset

def train_model(data_dir, num_epochs=10, batch_size=32, learning_rate=0.001, num_trash_classes=60):
    """
    Trains the TrashDetectionModel.
    """
    # Define transformations
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = TrashDetectionDataset(data_dir, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model
    model = TrashDetectionModel(num_trash_classes=num_trash_classes)

    # Define loss functions and optimizer
    criterion_bbox = torch.nn.SmoothL1Loss()
    criterion_class = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(num_epochs):
        for images, targets in dataloader:
            # Forward pass
            person_bbox, person_logits, face_embedding, trash_logits, disposal_logits = model(images)

            # Calculate loss
            # This is a simplified loss calculation. In a real scenario, you would need to match the predictions with the targets.
            loss = criterion_class(person_logits, targets['person_class']) + \
                   criterion_class(trash_logits, targets['trash_class']) + \
                   criterion_class(disposal_logits, targets['disposal_class']) + \
                   criterion_bbox(person_bbox, targets['person_bbox'])

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    # Save the model
    torch.save(model.state_dict(), 'trash_detection_model.pth')

if __name__ == '__main__':
    data_dir = 'c:\\Users\\KUNAL SHEDGE\\Desktop\\New folder\\trash_detect\\data\\unified_dataset'
    train_model(data_dir)
