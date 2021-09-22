""" Evaluation script to evaluate Out-of-Context Detection Accuracy"""

import cv2
import os
import json
from timeit import default_timer as dt
from utils.config import *
from utils.text_utils import get_text_metadata
from model_archs.models import CombinedModelMaskRCNN
from utils.common_utils import read_json_data
from utils.eval_utils import *


# Word Embeddings
text_field, word_embeddings, vocab_size = get_text_metadata()

# Models (create model according to text embedding)
if embed_type == 'use':
    # For USE (Universal Sentence Embeddings)
    model_name = 'img_use_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(hidden_size=300, use=True).to(device)
else:
    # For Glove and Fasttext Embeddings
    model_name = 'img_lstm_glove_rcnn_margin_10boxes_jitter_rotate_aug_ner'
    combined_model = CombinedModelMaskRCNN(use=False, hidden_size=300, embedding_length=word_embeddings.shape[1]).to(device)

print("Total Params", sum(p.numel() for p in combined_model.parameters() if p.requires_grad))


def get_scores(v_data):
    """
        Computes score for the two captions associated with the image

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            score_c1 (float): Score for the first caption associated with the image
            score_c2 (float): Score for the second caption associated with the image
    """
    checkpoint = torch.load(BASE_DIR + 'models_final/' + model_name + '.pt')
    combined_model.load_state_dict(checkpoint)
    combined_model.to(device)
    combined_model.eval()

    img_path = os.path.join(DATA_DIR, v_data["img_local_path"])
    bbox_list = v_data['maskrcnn_bboxes']
    bbox_classes = [-1] * len(bbox_list)
    img = cv2.imread(img_path)
    img_shape = img.shape[:2]
    bbox_list.append([0, 0, img_shape[1], img_shape[0]])  # For entire image (global context)
    bbox_classes.append(-1)
    cap1 = v_data['caption1_modified']
    cap2 = v_data['caption2_modified']

    img_tensor = [torch.tensor(img).to(device)]
    bboxes = [torch.tensor(bbox_list).to(device)]
    bbox_classes = [torch.tensor(bbox_classes).to(device)]

    if embed_type != 'use':
        # For Glove, Fasttext embeddings
        cap1_p = text_field.preprocess(cap1)
        cap2_p = text_field.preprocess(cap2)
        embed_c1 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap1_p]).unsqueeze(
            0).to(device)
        embed_c2 = torch.stack([text_field.vocab.vectors[text_field.vocab.stoi[x]] for x in cap2_p]).unsqueeze(
            0).to(device)
    else:
        # For USE embeddings
        embed_c1 = torch.tensor(use_embed([cap1]).numpy()).to(device)
        embed_c2 = torch.tensor(use_embed([cap2]).numpy()).to(device)

    with torch.no_grad():
        z_img, z_t_c1, z_t_c2 = combined_model(img_tensor, embed_c1, embed_c2, 1, [embed_c1.shape[1]],
                                               [embed_c2.shape[1]], bboxes, bbox_classes)

    z_img = z_img.permute(1, 0, 2)
    z_text_c1 = z_t_c1.unsqueeze(2)
    z_text_c2 = z_t_c2.unsqueeze(2)

    # Compute Scores
    score_c1 = torch.bmm(z_img, z_text_c1).squeeze()
    score_c2 = torch.bmm(z_img, z_text_c2).squeeze()

    return score_c1, score_c2

def evaluate_context_with_bbox_overlap_original(v_data):
    """
        Computes predicted out-of-context label for the given data point

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            context_label (int): Returns 0 if its same/similar context and 1 if out-of-context
    """
    bboxes = v_data['maskrcnn_bboxes']
    score_c1, score_c2 = get_scores(v_data)
    textual_sim = float(v_data['bert_base_score'])

    top_bbox_c1, sorted_bbox_c1_scores = top_bbox_from_scores_original(bboxes, score_c1)
    top_bbox_c2, sorted_bbox_c2_scores = top_bbox_from_scores_original(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, 0.5)
    if bbox_overlap:
        # Check for captions with same context : Same grounding with high textual overlap (Not out of context)
        if textual_sim >= textual_sim_threshold:
            context = 0
        # Check for captions with different context : Same grounding with low textual overlap (Out of context)
        else:
            context = 1
        return context, (sorted_bbox_c1_scores, sorted_bbox_c2_scores)
    else:
        # Check for captions with same context : Different grounding (Not out of context)
        return 0, (sorted_bbox_c1_scores, sorted_bbox_c2_scores)

def evaluate_context_with_bbox_overlap(v_data):
    """
        Computes predicted out-of-context label for the given data point

        Args:
            v_data (dict): A dictionary holding metadata about on one data sample

        Returns:
            context_label (int): Returns 0 if its same/similar context and 1 if out-of-context
    """
    bboxes = v_data['maskrcnn_bboxes']
    score_c1, score_c2 = get_scores(v_data)
    textual_sim = float(v_data['bert_base_score'])
    scores_c1 = top_scores(score_c1)
    scores_c2 = top_scores(score_c2)
    top_bbox_c1, top_bbox_next_c1 = top_bbox_from_scores(bboxes, score_c1)
    top_bbox_c2, top_bbox_next_c2 = top_bbox_from_scores(bboxes, score_c2)
    bbox_overlap = is_bbox_overlap(top_bbox_c1, top_bbox_c2, iou_overlap_threshold)
    bbox_overlap_next = is_bbox_overlap(top_bbox_next_c1, top_bbox_next_c2, iou_overlap_threshold)
    iou = bb_intersection_over_union(top_bbox_c1, top_bbox_c2)
    if os.getenv("COSMOS_WORD_DISABLE") is None and \
        (textual_sim > 0.5 and (is_fake(v_data)[0] or is_opposite(v_data)[0])):
        context = 1
    else:
        if bbox_overlap:
            # Check for captions with same context : Same grounding with high textual overlap (Not out of context)
            if textual_sim >= textual_sim_threshold:
                context = 0
            # Check for captions with different context : Same grounding with low textual overlap (Out of context)
            else:
                context = 1
        else:
            # Check for captions with same context : Different grounding (Not out of context)
            context = 0
    return iou, scores_c1, scores_c2, context

def logger(state, context, v_data, iou, bbox_scores):
    level = os.getenv("COSMOS_COMPARE_LEVEL")
    level = 0 if level is None else int(level)

    if level == 0:
        print(state, context, v_data["img_local_path"])
    elif level == 1:
        print(state, context, \
                v_data["img_local_path"], \
                "is_fake", is_fake(v_data), \
                "is_opposite", is_opposite(v_data))
    elif level == 2:
        print(state, context, \
                v_data["img_local_path"], \
                "is_fake", is_fake(v_data), \
                "is_opposite", is_opposite(v_data), \
                "bert", float(v_data['bert_base_score']), \
                "iou", iou, \
                "bbox", v_data['maskrcnn_bboxes'], \
                "bbox_scores_c1", bbox_scores[0], \
                "bbox_scores_c2", bbox_scores[1])

if __name__ == "__main__":
    """ Main function to compute out-of-context detection accuracy"""

    if (not torch.cuda.is_available()):
        print("A GPU is required for the model to run")

    test_samples = read_json_data(os.path.join(DATA_DIR, 'annotations', 'test_data.json'))
    ours_correct = 0
    lang_correct = 0
    compare_flag = os.getenv("COSMOS_COMPARE") is not None
    duration = 0
    metrics = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
    }

    start = dt()
    for i, v_data in enumerate(test_samples):
        actual_context = int(v_data['context_label'])
        language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1
        iou, _, _, pred_context = evaluate_context_with_bbox_overlap(v_data)

        if actual_context == 1:
            if actual_context == pred_context:
                metrics["TP"] += 1
            else:
                metrics["FN"] += 1
        else:
            if actual_context == pred_context:
                metrics["TN"] += 1
            else:
                metrics["FP"] += 1

        if compare_flag:
            pred_context_original, bbox_scores = evaluate_context_with_bbox_overlap_original(v_data)

            if pred_context == actual_context and pred_context_original == actual_context:
                logger("BOTH CORRECT", actual_context, v_data, iou, bbox_scores)
            elif pred_context != actual_context and pred_context_original == actual_context:
                logger("ORIGINAL CORRECT", actual_context, v_data, iou, bbox_scores)
            elif pred_context == actual_context and pred_context_original != actual_context:
                logger("OURS CORRECT", actual_context, v_data, iou, bbox_scores)
            else:
                logger("BOTH FALSE", actual_context, v_data, iou, bbox_scores)

        if pred_context == actual_context:
            ours_correct += 1

        if language_context == actual_context:
            lang_correct += 1

    with open("/tmp/metrics.json", "w") as fp:
        json.dump(metrics, fp)

    print("\n")
    print("Cosmos on Steroids Accuracy", ours_correct / len(test_samples))
    print(f"Cosmos on Steroids Inference Latency {(dt() - start) / len(test_samples):.6f} seconds")
    print("Language Baseline Accuracy", lang_correct / len(test_samples))
