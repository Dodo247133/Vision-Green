# This file will contain the PyTorch model definition.

import torch
import torch.nn as nn
import torchvision.models as models

class TrashDetectionModel(nn.Module):
    def __init__(self, num_trash_classes, pretrained=True):
        super(TrashDetectionModel, self).__init__()

        # --- Feature Extractor ---
        # Use a pre-trained backbone (e.g., ResNet, MobileNet) for feature extraction.
        # This will be shared across all tasks.
        if pretrained:
            self.backbone = models.resnet50(pretrained=True)
            # Remove the original classification head
            self.backbone = nn.Sequential(*(list(self.backbone.children())[:-1]))
        else:
            self.backbone = models.resnet50(pretrained=False) # Or build from scratch
            self.backbone = nn.Sequential(*(list(self.backbone.children())[:-1]))

        # Get the output features size from the backbone
        # A dummy forward pass might be needed to determine this dynamically
        # For resnet50, it's typically 2048 after the avgpool layer
        self.feature_size = 2048 # This might need to be adjusted based on the chosen backbone

        # --- Task-Specific Heads ---

        # 1. Person Detection Head (e.g., Bounding Box Regression and Classification)
        # This would typically be a more complex object detection head (e.g., Faster R-CNN, YOLO).
        # For simplicity, we'll use placeholder linear layers.
        self.person_bbox_head = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 4) # 4 for [x_min, y_min, x_max, y_max]
        )
        self.person_class_head = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2) # 2 for [not_person, person]
        )

        # 2. Face Recognition Head (e.g., Embedding for similarity comparison)
        # This would typically involve a specialized face recognition architecture (e.g., ArcFace, FaceNet).
        self.face_embedding_head = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128) # 128-dim embedding
        )

        # 3. Trash Classification Head
        self.trash_class_head = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_trash_classes) # Number of trash categories
        )

        # 4. Disposal Status Head (Proper/Improper)
        self.disposal_status_head = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 2) # 2 for [improper, proper]
        )

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.size(0), -1) # Flatten features

        person_bbox = self.person_bbox_head(features)
        person_logits = self.person_class_head(features)
        face_embedding = self.face_embedding_head(features)
        trash_logits = self.trash_class_head(features)
        disposal_logits = self.disposal_status_head(features)

        return person_bbox, person_logits, face_embedding, trash_logits, disposal_logits

# --- Model Development Guidance ---
# As per GEMINI.md, you can use pre-trained models or build from scratch.
# Given the complexity and time constraints, using pre-trained models for the backbone
# and fine-tuning them for specific tasks is highly recommended.

# **Multi-task Learning:**
# The model above is designed for multi-task learning, where a shared backbone extracts features,
# and separate heads perform different tasks (person detection, face recognition, trash classification, disposal status).
# This can improve efficiency and generalization.

# **Training Strategy:**
# 1.  **Pre-training:** Start with a backbone pre-trained on a large dataset (e.g., ImageNet).
# 2.  **Fine-tuning:** Fine-tune the entire model or just the task-specific heads on your merged dataset.
# 3.  **Loss Functions:** Use appropriate loss functions for each task:
#     *   Person BBox: SmoothL1Loss or IoU Loss
#     *   Person Class: CrossEntropyLoss
#     *   Face Embedding: TripletLoss, ArcFaceLoss (requires careful data preparation)
#     *   Trash Class: CrossEntropyLoss
#     *   Disposal Status: CrossEntropyLoss
# 4.  **Optimization:** Adam or SGD optimizer.
# 5.  **Evaluation Metrics:**
#     *   Person Detection: mAP (mean Average Precision)
#     *   Face Recognition: Accuracy, F1-score on verification tasks
#     *   Trash Classification: Accuracy, Precision, Recall, F1-score
#     *   Disposal Status: Accuracy, Precision, Recall, F1-score

# **Considerations for Real-time Performance (1-hour constraint):**
# *   **Model Size:** Choose lightweight backbones (e.g., MobileNet, EfficientNet) if real-time performance on edge devices is critical.
# *   **Inference Optimization:** Use techniques like quantization, pruning, and ONNX export for faster inference.

# Example Usage:
# num_trash_categories = 6 # Example: plastic, paper, metal, glass, cardboard, organic
# model = TrashDetectionModel(num_trash_classes=num_trash_categories, pretrained=True)
# dummy_input = torch.randn(1, 3, 224, 224) # Batch size 1, 3 channels, 224x224 image
# person_bbox, person_logits, face_embedding, trash_logits, disposal_logits = model(dummy_input)
# print(person_bbox.shape, person_logits.shape, face_embedding.shape, trash_logits.shape, disposal_logits.shape)