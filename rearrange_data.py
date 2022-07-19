# Python 3 code to rename multiple
# files in a directory or folder
 
# importing os module
from statistics import median
import cv2 as cv
import os
import shutil
 
# Function to rename multiple files
def main():
   
    # Get subfolders in dataset path
    dataset_path = 'apple_peeling' # TODO turn this into an argument
    list_subfolder_paths = [f.path for f in os.scandir(dataset_path) if f.is_dir()]
        
    # Iterate through subfolders and generate image/annotation data
    for folder_path in list_subfolder_paths:
        file_num = folder_path.split(os.sep)[-1]
        image_id = f"10{file_num.rjust(6, '0')}"

        # Copy RGB
        dst = f"apples/train/apples_train2022/{image_id}.jpg"
        src = f"{folder_path}/RGB.png"  # foldername/filename, if .py file is outside folder
        shutil.copy(src, dst)

        # Copy Mask
        dst = f"{image_id}_flesh_1{file_num.rjust(2,'0')}{'1'.rjust(4,'0')}"
        dst = f"apples/train/annotations/{dst}.png"
        src = f"{folder_path}/Mask.png"  # foldername/filename, if .py file is outside folder
        shutil.copy(src, dst)

        # Copy Inverse Mask
        dst = f"{image_id}_skin_{file_num.rjust(2,'0')}{'2'.rjust(4,'0')}"
        dst = f"apples/train/annotations/{dst}.png"
        image = cv.imread(src, 0)
        # Invert but don't change value == 29
        image[image == 255] = 1
        image[image == 0] = 0
        image[image == 1] = 0
        cv.imwrite(dst, image)
 
# Driver Code
if __name__ == '__main__':
     
    # Calling main() function
    main()