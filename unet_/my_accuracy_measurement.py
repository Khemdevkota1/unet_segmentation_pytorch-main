from sklearn.metrics import confusion_matrix
import numpy as np

def iou_score_per_patch(y_true, y_pred):
    y_true=y_true.detach().cpu().numpy()
    y_pred=y_pred.detach().cpu().numpy()
    y_true=np.round(y_true,0)
    y_pred=np.round(y_pred,0)
    batch_size = y_true.shape[0]  # Assuming shape (batch_size, 1, height, width)
    iou_per_patch = []
    
    for i in range(batch_size):
        # Squeeze the single channel dimension to get (height, width)
        y_true_flat = y_true[i].flatten()  # Flatten the ground truth (1, height, width)
        y_pred_flat = y_pred[i].flatten()  # Flatten the prediction (1, height, width)
        
        # Compute the confusion matrix for each image (patch)
        cm = confusion_matrix(y_true_flat, y_pred_flat,labels=np.unique(y_true_flat))
        
        # Extract true positives (diagonal), false positives, and false negatives
        intersection = np.diag(cm)  # True positives per class
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection  # Union per class
        
        # Avoid division by zero for any class with no true or predicted instances
        iou = intersection / np.maximum(union, 1)
        
        # Add IoU for this image to the list
        iou_per_patch.append(iou)
    
    # Convert list to numpy array for easier manipulation
    iou_per_patch = np.array(iou_per_patch)
    
    # Mean IoU across all images in the batch
    mean_iou_batch = np.mean(np.mean(iou_per_patch, axis=0))
    return float(mean_iou_batch)