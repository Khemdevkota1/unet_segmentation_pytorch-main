#### -------->Install & Imports
# If you're in a Colab environment, install torchmetrics first

import os
import shutil
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.image as Imagemat

from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, Binarizer
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import io, transforms
from torchvision.transforms.v2 import Compose, ToTensor, Normalize, ToDtype
from torchmetrics.segmentation import MeanIoU
from torchmetrics import F1Score

import torch
import progressbar
from progressbar import ProgressBar

warnings.filterwarnings('ignore')





####-------> Connect to Google Drive
from google.colab import drive
drive.mount('/content/drive')

MAIN_PATH = "/content/drive/My Drive/Finalset_For_Model/"

TRAIN_IMAGES_PATH  = MAIN_PATH + "Training/Training_images/"
TRAIN_MASKS_PATH   = MAIN_PATH + "Training/Training_masks/"
TEST_IMAGES_PATH   = MAIN_PATH + "Validation/Validation_images/"
TEST_MASKS_PATH    = MAIN_PATH + "Validation/Validation_masks/"



# ---------> Define the UNet Model
from torch.hub import load
from torch.nn import Softmax, Conv2d

def unet_model():
    """
    Returns a UNet model with 4 output channels (for 4 classes).
    Pretrained is set to False. You can set pretrained to True if available.
    """
    model = load('mateuszbuda/brain-segmentation-pytorch', 
                 'unet',
                 in_channels=3, 
                 out_channels=1, 
                 init_features=32, 
                 pretrained=False)
    
    # Change the last convolutional layer to have 4 outputs instead of 1.
    # This is because we have 4 distinct classes in our segmentation task.
    model.conv = Conv2d(32, 4, kernel_size=(1,1), stride=(1,1))
    
    return model



# -----------> Dataset & Data Preparation
class UnetDataPreparation(Dataset):
    """
    A PyTorch Dataset class to load images and their corresponding masks.
    - Automatically resizes images to 512x512.
    - Encodes mask pixel values to class indices.
    """

    def __init__(self, ImagesDirectory, MasksDirectory):
        self.ImagesDirectory = ImagesDirectory
        self.MasksDirectory = MasksDirectory
        
        self.images = os.listdir(self.ImagesDirectory)
        self.masks  = os.listdir(self.MasksDirectory)

        # Define transformations for mask (resize only, NEAREST for segmentation)
        self.mask_transforms = transforms.Compose([
            transforms.Resize((512,512), interpolation=transforms.InterpolationMode.NEAREST),
        ])

        # Define transformations for images (resize and convert to Tensor)
        self.image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512,512), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x[0:3])  # Ensure we only take the first 3 channels if present
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """
        Returns a tuple: (image_tensor, mask_tensor)
        """
        img_name = self.images[index]
        img_path = os.path.join(self.ImagesDirectory, img_name)

        # Read and decode the image
        img = io.read_file(img_path)
        img = io.decode_png(img)  # shape: C,H,W

        # Construct the corresponding mask name
        mask_name = img_name.replace(' ', '_').replace('.PNG', '_mask.PNG')
        mask_path = os.path.join(self.MasksDirectory, mask_name)

        # Read the mask
        mask = io.read_image(mask_path)  # shape: C,H,W

        # Transform both image and mask
        img  = self.image_transforms(img)
        mask = self.mask_transforms(mask)

        # Recode mask to have class indices. Example mapping:
        # 0   -> 0
        # 255 -> 1
        # 85  -> 2
        # 170 -> 3
        mask_recode_dict = {170:3, 85:2, 0:0, 255:1}
        recoded_mask = torch.zeros_like(mask, dtype=torch.long)
        for k,v in mask_recode_dict.items():
            recoded_mask[mask == k] = v

        return img, recoded_mask



# ----------> Visualization Utilities
def compare_mask(true_mask, pred_mask, index):
    """
    Compare and plot the true mask vs predicted mask.
    Saves the plot in `subplots/` folder with filename: {index}.png
    """

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Move tensors to CPU and detach
    true_mask = true_mask[0].detach().cpu()

    # Recode the true mask to 4 classes
    mask_recode_dict = {170:3, 85:2, 0:0, 255:1}
    recoded_mask = torch.zeros_like(true_mask, dtype=torch.long)
    for k, v in mask_recode_dict.items():
        recoded_mask[true_mask == k] = v
    true_mask = recoded_mask

    pred_mask = pred_mask[0].detach().cpu().argmax(dim=0)

    LABELS = [0,1,2,3]
    NUM_CLASSES = len(LABELS)
    cmap = plt.cm.get_cmap('tab20', NUM_CLASSES)
    norm = mcolors.Normalize(vmin=LABELS[0], vmax=LABELS[-1])

    # True mask
    ax1.imshow(true_mask, cmap=cmap, norm=norm)
    ax1.set_title("TRUE MASK")
    ax1.axis('off')

    # Predicted mask
    ax2.imshow(pred_mask, cmap=cmap, norm=norm)
    ax2.set_title("PRED MASK")
    ax2.axis('off')

    # Add colorbar
    cbar = fig.colorbar(ax1.images[0], ax=[ax1, ax2], orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_ticks(range(NUM_CLASSES))
    cbar.set_label('Class Index')

    # Save figure
    plt.savefig(f"subplots/{index}.png", dpi=100)
    plt.close()


def compare_image_true_mask_pred_mask(true_mask, pred_mask, true_image, save_path):
    """
    Creates a side-by-side figure of [True Image, True Mask, Predicted Mask]
    and saves it to the specified path.
    """

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    LABELS = [0,1,2,3]
    NUM_CLASSES = len(LABELS)
    cmap = plt.cm.get_cmap('tab20', NUM_CLASSES)
    norm = mcolors.Normalize(vmin=LABELS[0], vmax=LABELS[-1])

    # Move tensors to CPU and detach
    true_mask_cpu = true_mask.detach().cpu()

    # Recode mask to 4 classes
    mask_recode_dict = {170:3, 85:2, 0:0, 255:1}
    recoded_mask = torch.zeros_like(true_mask_cpu, dtype=torch.long)
    for k,v in mask_recode_dict.items():
        recoded_mask[true_mask_cpu == k] = v
    true_mask_cpu = recoded_mask

    pred_mask_cpu = pred_mask.detach().cpu()

    # 1) True Image
    ax1.imshow(true_image.cpu().permute(1,2,0))
    ax1.set_title("TRUE IMAGE")
    ax1.axis('off')

    # 2) True Mask
    im2 = ax2.imshow(true_mask_cpu, cmap=cmap, norm=norm)
    ax2.set_title("TRUE MASK")
    ax2.axis('off')

    # 3) Predicted Mask
    im3 = ax3.imshow(pred_mask_cpu, cmap=cmap, norm=norm)
    ax3.set_title("PRED MASK")
    ax3.axis('off')

    # Add colorbar
    cbar = fig.colorbar(im2, ax=[ax2, ax3], orientation='vertical', fraction=0.05, pad=0.05)
    cbar.set_ticks(range(NUM_CLASSES))
    cbar.set_label('Class Index')

    # Save figure
    plt.savefig(save_path, dpi=100)
    plt.close()



# ----------> Create Datasets & Dataloaders
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create training dataset and dataloader
train_dataset = UnetDataPreparation(TRAIN_IMAGES_PATH, TRAIN_MASKS_PATH)
train_dataloader = DataLoader(train_dataset, batch_size=16, num_workers=3, shuffle=True)

# Create validation dataset and dataloader
test_dataset = UnetDataPreparation(TEST_IMAGES_PATH, TEST_MASKS_PATH)
test_dataloader = DataLoader(test_dataset, batch_size=16, num_workers=3, shuffle=False)



# ----------->
#Instantiate Model, Loss, Optimizer, and Metrics
model = unet_model().to(device)

# If there is a saved best_model.pth, load it
if os.path.exists("best_model.pth"):
    weights = torch.load("best_model.pth")
    model.load_state_dict(weights)
    print("Existing model weights loaded.")
else:
    print("No pre-trained model found. Using a newly initialized model.")

# Hyperparameters
BATCH_SIZE = 16
LR = 0.0005           # 1e-3 / 2 = 0.0005
B1 = 0.9
B2 = 0.999
n_epochs = 800

# Define loss function and optimizer
loss_fn  = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LR, betas=(B1, B2))

# Define metrics
iou_train = MeanIoU(include_background=False, num_classes=4).to(device)
f1_train  = F1Score(task="multiclass", num_classes=4).to(device)
iou_test  = MeanIoU(include_background=False, num_classes=4).to(device)
f1_test   = F1Score(task="multiclass", num_classes=4).to(device)


# ---------->
#Validation Function
def validate_model(model, dataloader, loss_fn, iou_metric, f1_metric):
    """
    Validates the model on a given dataloader.
    Returns: (avg_loss, iou, f1)
    """

    # ProgressBar setup for validation
    widgets_valid = [
        ' [', progressbar.Percentage(), '] ',
        ' (', progressbar.ETA(), ') ',
        '\t', progressbar.Variable('test_loss'),
        '\t', progressbar.Variable('iou_test_accuracy'),
        '\t', progressbar.Variable('f1_test_accuracy')
    ]
    
    progress = ProgressBar(max_value=len(dataloader), widgets=widgets_valid, prefix="Validation")
    model.eval()
    
    total_loss = 0.0
    iou_metric.reset()
    f1_metric.reset()

    for i, (val_x, val_y) in enumerate(dataloader):
        val_x, val_y = val_x.to(device), val_y.to(device)

        # Convert mask to (N,H,W) instead of (N,C,H,W)
        N, C, H, W = val_y.shape
        val_y = val_y.reshape((N, H, W)).long()

        with torch.no_grad():
            pred = model(val_x)
            loss = loss_fn(pred, val_y)
            total_loss += loss.item()

            # Update metrics
            iou_metric.update(pred.argmax(dim=1), val_y)
            f1_metric.update(pred.argmax(dim=1), val_y)

            progress.update(
                test_loss=loss.item(), 
                iou_test_accuracy=iou_metric.compute(), 
                f1_test_accuracy=f1_metric.compute()
            )
            progress.next()
    
    avg_loss = total_loss / len(dataloader)
    iou_val  = iou_metric.compute()
    f1_val   = f1_metric.compute()
    return avg_loss, iou_val, f1_val


# ----------->
#TensorBoard Logging
writer = SummaryWriter(log_dir="./logdir")

def tensor_board_writer(train_iou, test_iou, train_loss, test_loss, epoch, writer):
    """
    Write train and test metrics to TensorBoard.
    """
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("acc/train", train_iou, epoch)   # Using iou as 'acc'
    writer.add_scalar("Loss/test", test_loss, epoch)
    writer.add_scalar("acc/test", test_iou, epoch)


# ------------->
#Utility to Save Training Summary
def save_summary(df, epoch, train_loss, test_loss,
                 iou_train_val, iou_test_val,
                 f1_train_val, f1_test_val, best_acc):
    """
    Saves or appends a row of metrics to 'summary.csv'.
    If the CSV doesn't exist, it creates one with appropriate columns.
    """
    # Convert to float for safe CSV writing
    iou_train_val = float(iou_train_val.cpu().item())
    iou_test_val  = float(iou_test_val.cpu().item())
    f1_train_val  = float(f1_train_val.cpu().item())
    f1_test_val   = float(f1_test_val.cpu().item())
    best_acc      = float(best_acc.cpu().item())
    
    if os.path.exists("summary.csv"):
        df = pd.read_csv("summary.csv")
    df.loc[len(df)] = {
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
        "iou_train": iou_train_val,
        "iou_test": iou_test_val,
        "f1_train": f1_train_val,
        "f1_test": f1_test_val,
        "best_acc": best_acc
    }
    df.to_csv("summary.csv", index=False)


# ---------->
#Prepare for Training
# Initialize a DataFrame to track training progress
df_summary = pd.DataFrame(columns=[
    "epoch","train_loss","test_loss",
    "iou_train","iou_test","f1_train","f1_test","best_acc"
])

best_acc    = torch.tensor(-1.0).to(device)
best_epoch  = -1
threshold   = 2  # If the accuracy doesn't improve for 'threshold' consecutive epochs, we stop early
consec_fail = 0  # Count of consecutive epochs not improving




# --------------->
#Training Loop
for epoch in range(n_epochs):
    # Progress bar for training
    widgets_train = [
        ' [', progressbar.Percentage(), '] ',
        ' (', progressbar.ETA(), ') ',
        '\t', progressbar.Variable('epoch'),
        '\t', progressbar.Variable('train_loss'),
        '\t', progressbar.Variable('iou_train_accuracy'),
        '\t', progressbar.Variable('f1_train_accuracy')
    ]
    
    progress = ProgressBar(max_value=len(train_dataloader), widgets=widgets_train, prefix="Training")
    
    model.train()
    total_loss_epoch = 0.0
    
    # Reset training metrics for each epoch
    iou_train.reset()
    f1_train.reset()

    # ---- Training Step ----
    for i, (train_x, train_y) in enumerate(train_dataloader):
        train_x, train_y = train_x.to(device), train_y.to(device)
        
        # Convert mask to (N,H,W) for CrossEntropy
        N, C, H, W = train_y.shape
        train_y = train_y.reshape((N, H, W)).long()

        pred = model(train_x)

        # Optional: We can visualize sample predictions
        # compare_mask(train_y, pred, f"epoch_{epoch}_batch_{i}")

        loss = loss_fn(pred, train_y)
        total_loss_epoch += loss.item()

        # Update training metrics
        iou_train.update(pred.argmax(dim=1), train_y)
        f1_train.update(pred.argmax(dim=1), train_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        progress.update(
            epoch=epoch, 
            train_loss=loss.item(),
            iou_train_accuracy=iou_train.compute(),
            f1_train_accuracy=f1_train.compute()
        )
        progress.next()

    avg_train_loss = total_loss_epoch / len(train_dataloader)
    
    # ---- Validation Step ----
    avg_val_loss, iou_val, f1_val = validate_model(model, test_dataloader, loss_fn, iou_test, f1_test)

    # Print a summary for this epoch
    print(f"\nEpoch: {epoch}/{n_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Train IoU: {iou_train.compute():.4f}, "
          f"Train F1: {f1_train.compute():.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Val IoU: {iou_val:.4f}, "
          f"Val F1: {f1_val:.4f}, "
          f"Best Acc so far: {best_acc:.4f}\n")

    # Write logs to TensorBoard
    tensor_board_writer(
        iou_train.compute(), iou_val, 
        avg_train_loss, avg_val_loss, 
        epoch, writer
    )

    # Save to CSV summary
    save_summary(
        df_summary, 
        epoch, avg_train_loss, avg_val_loss, 
        iou_train.compute(), iou_val, 
        f1_train.compute(), f1_val, 
        best_acc
    )

    # Early stopping logic:
    # Check if current IoU is better than best_acc
    if iou_val > best_acc:
        # Found a new best model
        best_acc   = iou_val
        best_epoch = epoch
        torch.save(model.state_dict(), "best_model.pth")
        torch.save(optimizer.state_dict(), "best_optimizer.pth")
        print(f"New best model saved at epoch {epoch}. IoU: {best_acc:.4f}")
        consec_fail = 0  # reset
    else:
        consec_fail += 1
        print(f"Validation IoU did not improve. (Fail count: {consec_fail}/{threshold})")

    if consec_fail >= threshold:
        print(f"Early stopping triggered. No improvement for {consec_fail} consecutive epochs.")
        break


# --------->
#Post-Training / Best Model
# Load the best model
if os.path.exists("best_model.pth"):
    best_weights = torch.load("best_model.pth")
    model.load_state_dict(best_weights)
    print("Loaded best model weights from 'best_model.pth'. Best epoch:", best_epoch)



# ---------->
#Generate Comparison Images
# Make a folder to store final comparisons
if os.path.exists("COMPARED_IMAGES"):
    shutil.rmtree("COMPARED_IMAGES")
os.mkdir("COMPARED_IMAGES")

def preprocess_image(image_path):
    """
    Load a single image from disk, resize, and convert to Tensor (C,H,W).
    Returns:
        original_image (tensor, shape [C,H,W]) - unresized
        transformed_image (tensor, shape [C,H,W]) - resized
    """
    image = io.read_image(image_path)
    image_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512,512), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[0:3])
    ])
    # original image as read
    original_image = image
    # resized image
    transformed    = image_transforms(image)
    return original_image, transformed


# ----------->
# Iterate over images in the validation set for demonstration
val_images = os.listdir(TEST_IMAGES_PATH)
for img_name in val_images:
    if not img_name.lower().endswith('.png'):
        continue

    # Construct the mask name
    mask_name = img_name.replace(' ', '_').replace('.PNG', '_mask.PNG')
    mask_full_path = os.path.join(TEST_MASKS_PATH, mask_name)

    # Load the image
    img_full_path, img_transformed = preprocess_image(os.path.join(TEST_IMAGES_PATH, img_name))

    # Load the mask
    mask_readed = io.read_image(mask_full_path).squeeze(0)  # shape [H,W]

    # Run inference
    with torch.no_grad():
        pred_logits = model(img_transformed.unsqueeze(0).to(device))
    pred_mask = pred_logits[0].argmax(dim=0)

    # Save comparison
    save_path = f"COMPARED_IMAGES/{img_name}"
    compare_image_true_mask_pred_mask(mask_readed, pred_mask, img_full_path, save_path)

print("Comparison images saved in COMPARED_IMAGES folder.")


# ---------->
# Testing on Unused / Extra Data
# For a new test set
unused_test_images_path = MAIN_PATH + "Testing_unused_data/Testing_images/"
unused_test_masks_path  = MAIN_PATH + "Testing_unused_data/Testing_masks/"

unused_test_dataset = UnetDataPreparation(unused_test_images_path, unused_test_masks_path)
unused_test_loader  = DataLoader(unused_test_dataset, batch_size=16, num_workers=3)

# Evaluate
unused_test_loss, unused_test_iou, unused_test_f1 = validate_model(
    model, 
    unused_test_loader, 
    loss_fn, 
    iou_test, 
    f1_test
)

print(f"Unused Data - Loss: {unused_test_loss:.4f}, IoU: {unused_test_iou:.4f}, F1: {unused_test_f1:.4f}")

