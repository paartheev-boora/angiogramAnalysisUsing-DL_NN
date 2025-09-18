# fusion_module.py

import torch

def fuse_decisions(video_pred_prob, numerical_features, thresholds=None):
    """
    Combine CNN+LSTM model output with numerical features.

    Args:
        video_pred_prob (float): Probability of PCI prediction from CNN+LSTM.
        numerical_features (dict): Contains avg_flow, avg_contrast, avg_area.
        thresholds (dict): Optional thresholds to control fusion rules.

    Returns:
        final_decision (str): 'PCI Recommended' or 'No PCI Required'
    """
    if thresholds is None:
        thresholds = {
            "flow": 0.4,
            "contrast": 25,
            "area": 500,
            "cnn_prob": 0.5
        }

    # Weighted logic: boost decision confidence based on supportive features
    score = 0
    score += 1 if video_pred_prob > thresholds["cnn_prob"] else -1
    score += 1 if numerical_features["avg_flow"] > thresholds["flow"] else -1
    score += 1 if numerical_features["avg_contrast"] > thresholds["contrast"] else -1
    score += 1 if numerical_features["avg_area"] > thresholds["area"] else -1

    final_decision = "PCI Recommended" if score >= 2 else "No PCI Required"
    return final_decision
