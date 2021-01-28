import os
import cv2
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm

def from_iiit_dataset(input_dir, output_dir, splits_dict, labels_dict):
    make_filename = lambda *args: "_".join([str(a) for a in args]) + ".jpg"
    label_counts = {l: 0 for l in labels_dict.keys()}
    errors = []
    
    for split in splits_dict.keys():
        in_path = os.path.join(input_dir, split)
        imgs_path, xmls_path = in_path + "_images", in_path + "_xml"  
        img_paths, xml_paths = os.listdir(imgs_path), os.listdir(xmls_path)
        
        assert len(img_paths) == len(xml_paths)
        
        print(f"Creating {split} dir...")
        
        for img_path, xml_path in zip(img_paths, xml_paths):
            img = cv2.imread(os.path.join(imgs_path, img_path))
            tree = ET.parse(os.path.join(xmls_path, xml_path))
            root = tree.getroot()
            
            if all([label_counts[n] > splits_dict[split] for n in label_counts.keys()]):
                break

            for obj in root.iter("object"):
                label_name = obj.find("name").text
                
                if label_name not in labels_dict.keys():
                    continue
                
                if label_counts[label_name] > splits_dict[split]:
                    continue
                
                label_num = labels_dict[label_name]
                
                bndbox = obj.find("bndbox")
                xmin, ymin, xmax, ymax = [int(coord.text) for coord in bndbox]

                out_path = os.path.join(output_dir, split)
                out_name = make_filename(label_num, label_counts[label_name])
                try:
                    cv2.imwrite(os.path.join(out_path, out_name), img[ymin:ymax, xmin:xmax])
                    label_counts[label_name] += 1
                except Exception as e:
                    errors.append(str(e))
                
                if label_counts[label_name] % 100 == 0:
                    print(f"made {label_counts[label_name]} {label_name} images in {split}")
    with open("errors.txt", "wb") as f:
        f.write("Error Count: " + str(len(errors)) + "\n")
        f.writelines(errors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create balanced datasets with equal number of training data for each class")
    parser.add_argument("input_dir", type=str, help="Path to directory with images and xml folders")
    parser.add_argument("output_dir", type=str, help="Path to output directory containing train, val, and test folders")
    parser.add_argument("--train_size", type=int, default=2000, help="Size of train dir")
    parser.add_argument("--val_size", type=int, default=400, help="Size of train dir")
    parser.add_argument("--test_size", type=int, defualt==400, help="Size of train dir")
    args = parser.parse_args()

    splits_dict = {"train": args.train_size, "val": args.val_size, "test": args.test_size}
    labels_dict = {"table": 0, "figure": 1, "natural_image": 2}
    
    from_iiit_dataset(args.output_dir, args.input_dir, splits_dict, labels_dict)

