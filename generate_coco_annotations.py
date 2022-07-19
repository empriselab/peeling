import cv2 as cv
import json
import numpy as np
import os
# from pycocotools import mask as maskUtils
# import cvbase as cvb

root = "DATASET_PATH"
train_json_path = root+"/annotations/"+"instances_train.json"

# COCO_CLASSES = coco_classes or your own dataset classes here

COCO_LABEL_MAP = {1: 1, 2: 2, 4: 3, 5: 4,
                  6: 5, 7: 6, 8: 7}



def get_bounding_box(mask):
    """
    Returns bbox of mask

    Parameters
    ----------
        mask : numpy array
            Binary mask

    Returns
    -------
        int, int, int, int
            x, y, w, h where x and y are the top left bbox coordinates and w 
            and h are the width and height
    """
    coords = np.transpose(np.nonzero(mask))
    y, x, h, w = cv.boundingRect(coords)
    return x, y, w, h



def process_one_image(folder_path):
    """
    Returns image and annotation data for one image in COCO format

    Parameters
    ----------
        folder_path : string
            path containing the following:
                image: RGB.png
                bounding box: BoundingBox.png
                segmentation mask: Mask.png

    Returns
    -------
        dictionary, list of dictionaries
            img_dict (dictionary) contains image data in COCO format
            anns_list (list of dictionaries) contains the annotation data
                in COCO format
    """

    # Generate image section
    file_num = folder_path.split(os.sep)[-1]
    height, width, _ = cv.imread(folder_path + "/RGB.png").shape
    img_dict = {
        "license": 1,
        "file_name": file_num,
        "height": width,
        "width": height,
        "id": file_num
    }

    # Generate annotation section
    # TODO: Need to account for when there may be multiple, individual peels 
    # (aka more than one annotation for this image). Use for loop and go through
    # different blobs
    anns_list = []
    ann_dict = {
        "segmentation": [[]],
        "area": -1,
        "iscrowd": -1,
        "image_id": -1,
        "bbox": [],
        "category_id": -1,
        "id": -1
    }
    anns_list.append(ann_dict)
    
    return img_dict, anns_list



if __name__ == "__main__":
    with open('annotations.json', 'w') as output:
        print("Created JSON file")

        # Info section
        info_dict = {
            "description": "EmPRISE Peeling Dataset",
            "url": "https://emprise.cs.cornell.edu/",
            "version": "0.0",
            "year": 2022,
            "contributor": "Daniel Stabile",
            "date_created": "2022/07/12"
        }

        # Liscense section
        licenses_list = [
            {
                "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
                "id": 1,
                "name": "Attribution-NonCommercial-ShareAlike License"
            },
        ]

        # Categories Section
        categories_list = [
            {
                "supercategory": "food","id": 1,"name": "apple", # TODO Does the id/supercat need to be consistent with COOC?
            },
        ]

        all_anns_list = []
        all_imgs_list = []

        # Get subfolders in dataset path
        dataset_path = 'apple_peeling' # TODO turn this into an argument
        list_subfolder_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
        
        # Iterate through subfolders and generate image/annotation data
        for folder_path in list_subfolder_paths:
            print(f"Generating data for {folder_path}")
            
            # Get image and annotation dictionaries for this image
            img_dict, anns_list = process_one_image(folder_path)
            
            # Add image and annotations to lists
            all_imgs_list.append(img_dict)
            all_anns_list.extend(anns_list)

        # Combine all sections to create final annotation
        json_annotation = {
            "info": info_dict,
            "licenses": licenses_list,
            "images": all_imgs_list,
            "annotations": all_anns_list,
            "categories": categories_list,
        }
        
        json.dump(json_annotation, output)
