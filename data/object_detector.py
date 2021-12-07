import argparse
import pickle

import cv2
import torch
import tqdm
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN


class ImageObject:
    def __init__(self, origin_path, ppl_save_path):
        """
            image object data structure
        :param origin_path: the path which saves the raw image(*.jpg)
        :param ppl_save_path: the path which should save the output json(*.pkl)
        """
        self.origin_path = origin_path
        self.ppl_save_path = ppl_save_path
        pass

    def __repr__(self):
        return "Origin image path: {}, Output pkl path: {}".format(self.origin_path, self.ppl_save_path)


class ObjectDetector:
    def __init__(self, model_yaml, thresh=0.5, nms_thresh=0.7, verbose=0):
        """
            tool to extract objects in images
        :param model_yaml: model config file, please specify the detectron2
        :param thresh: the score threshold to filter the objects
        :param nms_thresh: the threshold for nms
        :param verbose: 0: no log infos, 1: print the config file, 2: print all log infos
        """
        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_yaml))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thresh
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_yaml)
        self.predictor = DefaultPredictor(self.cfg)
        self.verbose = verbose
        if self.verbose > 0:
            print(self.cfg)

    @classmethod
    def from_config(cls, args):
        return cls(model_yaml=args.model, thresh=args.thresh, nms_thresh=args.nms_thresh, verbose=1)

    def extract(self, image_list: [ImageObject]) -> None:
        """
            api for this tool
        :param image_list: a list of ImageObject
        :return None: no returns
        """
        for obj in tqdm.tqdm(image_list):
            ori_img_path = obj.origin_path
            original_image = cv2.imread(ori_img_path)
            ppl_json = self._process_image(original_image)
            if self.verbose == 2:
                print(ppl_json)
            with open(obj.ppl_save_path, "wb") as f:
                pickle.dump(ppl_json, f)

    def _process_image(self, cv2_image) -> dict:
        """
            the single step for extracting objects
            Note that it can only process ONE image
        :param cv2_image: the return of cv2.imread()
        :return dic: keys:["height", "width", "num_boxes", "box_scores", "box_cls", "box_feats"]
        """
        original_image = cv2_image
        with torch.no_grad():
            if self.predictor.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.predictor.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            inputs = {"image": image, "height": height, "width": width}

            images = self.predictor.model.preprocess_image([inputs])
            features = self.predictor.model.backbone(images.tensor)
            proposals, _ = self.predictor.model.proposal_generator(images, features, None)

            features_tmp = [features[f] for f in self.predictor.model.roi_heads.box_in_features]

            box_features = self.predictor.model.roi_heads.box_pooler(features_tmp,
                                                                     [x.proposal_boxes for x in proposals])
            # print(box_features.shape, "-------")
            box_features = self.predictor.model.roi_heads.box_head(box_features)

            predictions = self.predictor.model.roi_heads.box_predictor(box_features)
            del box_features
            pred_instances, _ = self.predictor.model.roi_heads.box_predictor.inference(predictions, proposals)
            pred_instances = self.predictor.model.roi_heads.forward_with_given_boxes(features, pred_instances)

            inst = GeneralizedRCNN._postprocess(pred_instances, [inputs], images.image_sizes)[0]

            #  for output

            inst = inst['instances']
            height, width = inst.image_size
            num_boxes = len(inst)
            dic = inst.get_fields()
            boxes_coordinate = dic['pred_boxes']
            box_scores = dic['scores']
            box_cls = dic['pred_classes']

            output_json = {
                'height': height, 'width': width, 'num_boxes': num_boxes,
                "boxes_coordinate": boxes_coordinate.tensor.cpu().numpy(),
                'box_scores': box_scores.cpu().numpy(),
                'box_cls': box_cls.cpu().numpy()
            }

            box_features = self.predictor.model.roi_heads.box_pooler(features_tmp, [boxes_coordinate])

            box_features = self.predictor.model.roi_heads.box_head(box_features)

            output_json["box_feats"] = box_features.cpu().numpy()
            return output_json


def parse_opt():
    parser = argparse.ArgumentParser()
    # # Data input settings
    parser.add_argument('--model', type=str, default='COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
                        help='')
    parser.add_argument('--image-path', type=str, default='cocoqa',
                        help='')
    parser.add_argument('--thresh', type=float, default=0, help="thresh value for inference")
    parser.add_argument('--nms-thresh', type=float, default=0.5, help='nms thresh')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_opt()
    detector = ObjectDetector.from_config(args)
    image_obj = [ImageObject(origin_path="/home/shiina/data/coco2014/train2014/pic/COCO_train2014_000000016449.jpg",
                             ppl_save_path="/home/shiina/shiina/detectron/detectron2/input.pkl")]
    detector.extract(image_list=image_obj)
