import cv2
import json
import pickle



def draw_bbox(img, bboxes, mask):
    for i in range(mask.shape[0]):
        if mask[i] == 0:
            continue
        bbox = bboxes[i]
        x1, y1, x2, y2 = bbox
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness=2)


if __name__ == "__main__":
    question_id_list = [181666003]
    # question_id_list = [451412000, 96207003, 46924001, 39322006, 39322007, 459801001]
    vis_obj_path = "/home/shiina/shiina/question/aaai/results/0.5_0.01_0.1_gcn/vis-15.pkl"
    ann_path = "/home/shiina/data/aaai/vqa2/val_split_dic_unique.pkl"
    with open(vis_obj_path, "rb") as f:
        vis_obj = pickle.load(f)
    with open(ann_path, "rb") as f:
        ann = pickle.load(f)

    for obj in vis_obj:
        if obj["question_id"]  not in question_id_list:
            continue
        question_id = obj["question_id"]
        ann_obj = ann[question_id]
        ppl_info_path = ann_obj['ppl_json_path']
        with open(ppl_info_path, "rb") as f:
            ppl_info = pickle.load(f)
        bbox_coordinates = ppl_info["boxes_coordinate"]

        gt = obj["vh_gt"]
        pred = obj["vh_pred"]

        src_img = cv2.imread(ann_obj["image_path"])
        draw_bbox(src_img, bbox_coordinates, gt)
        output_name = str(question_id) + "_gt.jpg"
        cv2.imwrite(output_name, src_img)

        src_img = cv2.imread(ann_obj["image_path"])
        draw_bbox(src_img, bbox_coordinates, pred)
        output_name = str(question_id) + "_pred.jpg"
        cv2.imwrite(output_name, src_img)