from data import *
from model import *
from train import *


dir_data = os.path.abspath("data")
dir_truth = os.path.join(dir_data, "gtFine")
dir_input = os.path.join(dir_data, "leftImg8bit")
sample_size = (256, 128)
dir_truth_pp, dir_input_pp = (f'{d}_{sample_size[0]}_{sample_size[1]}' for d in (dir_truth, dir_input))
ds_split = {
    name:CityscapesDataset(os.path.join(dir_input_pp, name), os.path.join(dir_truth_pp, name), sample_size, classes)
    for name in ("train", "val", "test")
}
model = UNet(3, 30)
#Train the passthrough network
print("Testing training process...")
trainer = Trainer(model, ds_split)
trainer.fit(epochs=15, batch_size=10, ds_split = ds_split)
torch.save(model.state_dict(), './model_Unet_210409_2.pth')