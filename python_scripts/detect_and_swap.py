import argparse
import os
import sys
from pathlib import Path
import numpy
import cv2

from torchvision.utils import save_image
from tqdm.auto import tqdm
from utils.database_utils import convert_imgs_to_db, load_db

from hair_swap import HairFast, get_parser
from swinface_onnx import SwinFaceORT

def init_models(model_args, args):
    print("====Initializing SwinFace model====")
    swinface = SwinFaceORT(args.swinface_model_path, cpu=False)
    error = swinface.check()
    if error is not None:
        print('error:', error)
        exit()
    print("====Models initialized successfully!====")
    print("====Initializing HairFast models====")
    hairfast = HairFast(model_args)
    print("====HairFast models initialized successfully!====")
    return hairfast, swinface

def init_database(args, swinface):
    print("====Initializing database====")
    if args.database_images_dir is not None:
        print("====Converting images to database====")
        if not os.path.exists(args.database_dir_path):
            os.makedirs(args.database_dir_path)
        return convert_imgs_to_db(args.database_images_dir,
                                  args.database_dir_path,
                                  swinface)
    else:
        print("====Loading existed database====")
        if not os.path.exists(args.database_dir_path):
            raise FileNotFoundError(f"Database directory {args.database_dir_path} not found")
        else:
            return load_db(args.database_dir_path)

def main(hair_fast, swinface, image_db, embedings_db, metadata, args):
    while True:
        face_path = input("Enter the path to the face image (Empty for default, q to quit): ")
        if face_path == "":
            face_path = args.face_path.__str__()
        if face_path == "q" or face_path == "Q":
            exit()
        if not os.path.exists(face_path):
            print(f"File {face_path} not found")
            continue
        user_input_face = cv2.imread(face_path)
        user_input_face = cv2.cvtColor(user_input_face, cv2.COLOR_BGR2RGB)
        facial_embeding = swinface.forward(user_input_face)
        cosine_distances = swinface.compare_with_db(facial_embeding, embedings_db)
        best_suitable_image_name = metadata[str(cosine_distances.argmin())]['name']
        best_suitable_image = image_db[cosine_distances.argmin()]
        minimum_distance = cosine_distances.min()
        print(f"Best suitable image: {best_suitable_image_name} and distance: {minimum_distance}", )
        gen_image_choice = input("Do you want to generate an image? (y/n): ")
        if gen_image_choice == "n":
            continue
        while True:
            color_mode = input("Enter the color mode (0 - black, 1 - yellow, 2 - white, 3 -  orange): ")
            if color_mode == "0":
                color_img = cv2.imread("input/6.png")
                break
            elif color_mode == "1":
                color_img = cv2.imread("input/4.jpg")
                break
            elif color_mode == "2":
                color_img = cv2.imread("input/2.png")
                break
            elif color_mode == "3":
                color_img = cv2.imread("input/8.png")
                break
            else:
                print("Invalid color mode")
                continue
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        output_image_path = args.output_dir / f"{Path(face_path).stem}_output.png"
        final_image, face_align, shape_align, color_align = \
            hair_fast.swap(user_input_face,
                           best_suitable_image,
                           color_img,
                           align=True)
        save_image(final_image, output_image_path)
        print("Image saved to", output_image_path)

if __name__ == "__main__":
    hairfast_model_parser = get_parser()
    parser = argparse.ArgumentParser(description='HairFast evaluate')
    parser.add_argument('--face_path', type=Path, default='input/0.png', help='Path to the face image')
    parser.add_argument('--swinface_model_path', type=Path, default='onnx_models/', help='Path to the swinface model')
    parser.add_argument('--database_dir_path', type=Path, default=Path('database'), help='The directory for the database')
    parser.add_argument('--database_images_dir', type=Path, default=None, help='The directory for the images to create the database')
    parser.add_argument('--output_dir', type=Path, default=Path('output'), help='The directory for final results')

    args, unknown1 = parser.parse_known_args()
    hairfast_args, unknown2 = hairfast_model_parser.parse_known_args()


    unknown_args = set(unknown1) & set(unknown2)
    if unknown_args:
        file_ = sys.stderr
        print(f"Unknown arguments: {unknown_args}", file=file_)

        print("\nExpected arguments for the model:", file=file_)
        hairfast_model_parser.print_help(file=file_)

        print("\nExpected arguments for evaluate:", file=file_)
        parser.print_help(file=file_)

        sys.exit(1)
    
    hairfast, swinface = init_models(hairfast_args, args)
    image_db, embedings_db, metadata = init_database(args, swinface)
    main(hairfast, swinface, image_db, embedings_db, metadata, args)
