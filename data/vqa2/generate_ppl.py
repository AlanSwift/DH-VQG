import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from data.object_detector import ImageObject, ObjectDetector
import argparse, os


def main(args):
    processor = ObjectDetector.from_config(args)
    obj_list = []
    for file_name in os.listdir(args.image_dir):

        input_path = os.path.join(args.image_dir, file_name)
        clean_name = file_name.replace("COCO_val2014_", "").replace(".jpg", "")

        output_path = os.path.join(args.output_dir, clean_name) + "_ppl.pkl"
        obj_list.append(ImageObject(origin_path=input_path, ppl_save_path=output_path))
    processor.extract(obj_list)


def parse_opt():
    parser = argparse.ArgumentParser()
    # # Data input settings
    parser.add_argument('--model', type=str, default='COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
                     help='')
    parser.add_argument('--image-dir', type=str, default='/home/shiina/data/coco2014/val2014/pic',
                     help='')
    parser.add_argument('--output-dir', type=str, default='/home/shiina/data/aaai/coco_visual_features/val')
    parser.add_argument('--thresh', type=float, default=0, help="thresh value for inference")
    parser.add_argument('--nms-thresh', type=float, default=0.5, help='nms thresh')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_opt()
    main(args)