import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random, torch

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
original_image = cv2.imread("/home/shiina/shiina/detectron/detectron2/input.jpg")
cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")    # COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml
predictor = DefaultPredictor(cfg)

# detect
with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
    # Apply pre-processing to image.
    if predictor.input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        original_image = original_image[:, :, ::-1]
    height, width = original_image.shape[:2]
    image = predictor.aug.get_transform(original_image).apply_image(original_image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    inputs = {"image": image, "height": height, "width": width}

    images = predictor.model.preprocess_image([inputs])
    features = predictor.model.backbone(images.tensor)
    proposals, _ = predictor.model.proposal_generator(images, features, None)
    # print(proposals[0].proposal_boxes, "--------+++++++++")
    # print(proposals[0].proposal_boxes.shape, "--------")
    # exit(0)

    features_tmp = [features[f] for f in predictor.model.roi_heads.box_in_features]
    box_features = predictor.model.roi_heads.box_pooler(features_tmp, [x.proposal_boxes for x in proposals])
    print(box_features.shape, "-------")
    box_features = predictor.model.roi_heads.box_head(box_features)
    print(box_features.shape, "========")
    predictions = predictor.model.roi_heads.box_predictor(box_features)
    del box_features
    pred_instances, _ = predictor.model.roi_heads.box_predictor.inference(predictions, proposals)


    pred_instances = predictor.model.roi_heads.forward_with_given_boxes(features, pred_instances)

    #results, _ = predictor.model.roi_heads(images, features, proposals, None)

    outputs = GeneralizedRCNN._postprocess(pred_instances, [inputs], images.image_sizes)[0]

print(outputs)