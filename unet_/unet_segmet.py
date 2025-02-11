"""unet main segnemtation"""
from torch.utils.data import DataLoader
from datapreparation import UnetDataPreparation
from my_unet import unet_model
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from visualization import input_image_visualizer
from progressbar import ProgressBar
import progressbar
from my_accuracy_measurement import iou_score_per_patch
import warnings
warnings.filterwarnings('ignore')
"""FILE PATHS"""
TRAIN_IMAGES_PATH="/home/jeshanpokharel/Downloads/carvana-image-masking-challenge/train/"
TRAIN_MASK="/home/jeshanpokharel/Downloads/carvana-image-masking-challenge/train_masks/"

#TRAIN_IMAGES_PATH="/home/jeshanpokharel/Downloads/khimsir new data/train_images/"
#TRAIN_MASK="/home/jeshanpokharel/Downloads/khimsir new data/train_masks/"

device="cuda"

# TRAIN datasets and dataloader
train_dataset=UnetDataPreparation(TRAIN_IMAGES_PATH, TRAIN_MASK)
train_dataloader=DataLoader(train_dataset,batch_size=8)


#define model
model=unet_model()
model=model.to(device)
#load loss and optimizer
loss_fn=BCEWithLogitsLoss()
optimizer=Adam(model.parameters())



#widget for progress bar
widgets = [
    ' [', progressbar.Percentage(), '] ',
    ' (', progressbar.ETA(), ') ',
    ' Loss: ', progressbar.Variable('loss'),  # Display loss
    ' Accuracy: ', progressbar.Variable('accuracy')  # Display accuracy
]





#train model
n_epochs=10
for epoch in range(n_epochs):
    progress=ProgressBar(min_value=0,max_value=len(train_dataloader),widgets=widgets)
    for train_x,train_y in train_dataloader:
        train_x,train_y=train_x.to(device),train_y.to(device)
        pred=model(train_x)
        #visualizer
        input_image_visualizer(pred[0].detach().cpu())
        
        #accuracy calculaltion per batch
        acc=iou_score_per_patch(train_y,pred)
        
        
        loss=loss_fn(pred,train_y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        #update progress and variables
        progress.update(loss=loss.item(),accuracy=acc)
        progress.next()
    progress.finish()
    print(f"{epoch} loss:{loss.item()}")
    
        
        




