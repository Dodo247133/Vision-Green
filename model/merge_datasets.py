
import os
import json
import shutil
from sklearn.model_selection import train_test_split

def process_taco_dataset(data_dir, output_dir):
    """
    Processes the TACO dataset to create a unified dataset.
    """
    annotations_file = os.path.join(data_dir, 'annotations.json')
    with open(annotations_file, 'r') as f:
        taco_data = json.load(f)

    images = taco_data['images']
    annotations = taco_data['annotations']
    categories = taco_data['categories']

    # Create a mapping from category id to category name
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # Create directories for the unified dataset
    unified_images_dir = os.path.join(output_dir, 'images')
    unified_labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(unified_images_dir, exist_ok=True)
    os.makedirs(unified_labels_dir, exist_ok=True)

    # Process each image and its annotations
    for img_info in images:
        img_id = img_info['id']
        img_filename = img_info['file_name']
        img_width = img_info['width']
        img_height = img_info['height']

        # Get all annotations for the current image
        img_annotations = [ann for ann in annotations if ann['image_id'] == img_id]

        if not img_annotations:
            continue

        # Copy the image to the unified directory
        shutil.copy(os.path.join(data_dir, img_filename), os.path.join(unified_images_dir, os.path.basename(img_filename)))

        # Create the label file
        label_filename = os.path.splitext(os.path.basename(img_filename))[0] + '.txt'
        with open(os.path.join(unified_labels_dir, label_filename), 'w') as f:
            for ann in img_annotations:
                cat_id = ann['category_id']
                cat_name = cat_id_to_name[cat_id]
                bbox = ann['bbox']

                # Convert bbox to YOLO format (x_center, y_center, width, height) normalized
                x_center = (bbox[0] + bbox[2] / 2) / img_width
                y_center = (bbox[1] + bbox[3] / 2) / img_height
                width = bbox[2] / img_width
                height = bbox[3] / img_height

                # Write the label in the format: class_id x_center y_center width height
                # For now, we will use the category id as the class id.
                # We will need to create a mapping of all class names to class ids later.
                f.write(f"{cat_id} {x_center} {y_center} {width} {height}\n")

def process_lfw_dataset(data_dir, output_dir):
    """
    Processes the LFW dataset to create a unified dataset.
    """
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            for img_filename in os.listdir(person_dir):
                shutil.copy(os.path.join(person_dir, img_filename), os.path.join(images_dir, img_filename))
                # Create an empty label file for now
                label_filename = os.path.splitext(img_filename)[0] + '.txt'
                with open(os.path.join(labels_dir, label_filename), 'w') as f:
                    pass

def process_human_detection_dataset(data_dir, output_dir):
    """
    Processes the Human Detection dataset to create a unified dataset.
    """
    images_dir = os.path.join(output_dir, 'images')
    labels_dir = os.path.join(output_dir, 'labels')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # The dataset has images and labels in separate directories
    img_dir = os.path.join(data_dir, 'images')
    lbl_dir = os.path.join(data_dir, 'labels')

    for img_filename in os.listdir(img_dir):
        shutil.copy(os.path.join(img_dir, img_filename), os.path.join(images_dir, img_filename))
        label_filename = os.path.splitext(img_filename)[0] + '.txt'
        shutil.copy(os.path.join(lbl_dir, label_filename), os.path.join(labels_dir, label_filename))

if __name__ == '__main__':
    output_dir = 'c:\\Users\\KUNAL SHEDGE\\Desktop\\New folder\\trash_detect\\data\\unified_dataset'

    # Process TACO dataset
    taco_data_dir = 'c:\\Users\\KUNAL SHEDGE\\Desktop\\New folder\\trash_detect\\data\\TACO\\data'
    process_taco_dataset(taco_data_dir, output_dir)

    # Process LFW dataset
    lfw_data_dir = 'c:\\Users\\KUNAL SHEDGE\\Desktop\\New folder\\trash_detect\\data\\lfw-deepfunneled'
    process_lfw_dataset(lfw_data_dir, output_dir)

    # Process Human Detection dataset
    human_detection_data_dir = 'c:\\Users\\KUNAL SHEDGE\\Desktop\\New folder\\trash_detect\\data\\human-detection-dataset'
    process_human_detection_dataset(human_detection_data_dir, output_dir)

