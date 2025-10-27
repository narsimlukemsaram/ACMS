import os
import random
import shutil

train_dir = 'train'
test_dir = 'test'

total_test_images = 1000  # desired total test images

# Get class folders
class_folders = [f for f in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, f))]
print('the class folders are------>',class_folders)
num_classes = len(class_folders)
print('the num_classes are--------->',num_classes)
images_per_class = total_test_images // num_classes

print('the images_per_classs are--------->',images_per_class)

os.makedirs(test_dir, exist_ok=True)
moved_count = 0

for class_name in class_folders:
    print('the class_name is------->',class_name)
    src_folder = os.path.join(train_dir, class_name)
    print(src_folder)
    dst_folder = os.path.join(test_dir, class_name)
    print(dst_folder)
    os.makedirs(dst_folder, exist_ok=True)

    images = [img for img in os.listdir(src_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
    n = min(images_per_class, len(images))
    selected = random.sample(images, n)

    for img in selected:
        print('the img is--------->',img)
        shutil.move(os.path.join(src_folder, img), os.path.join(dst_folder, img))
        moved_count += 1

print(f"✅ Moved {moved_count} images from train → test (no overlap).")

