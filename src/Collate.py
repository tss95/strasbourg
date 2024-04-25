import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt
from time import time
from PIL import ImageOps
#import transforms from pytorch
from torchvision import transforms

class CollateFn:
    def __init__(self, gaf_trace_transform = True, apply_randomcolormap= True, image_size=(256, 256)):
        self.image_size = image_size  # image_size should be (height, width)
        self.gaf_trace_transform = None
        if gaf_trace_transform:
            self.gaf_trace_transform = GAFTransform()
        self.apply_randomcolormap = None
        if apply_randomcolormap:
            self.apply_randomcolormap = ApplyRandomColormapBatch()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.min_length_trace = None
        self.min_length_mfr = None

    def __call__(self, batch):
        
        start = time()
        processed_data = {"img1": [], "img2" : []}
        
        for item in batch:
            item = {k: v.to(self.device) for k, v in item.items()}
            
        if self.gaf_trace_transform:
            if not self.min_length_trace:
                self.min_length_trace = min([item['max_trace1'].shape[0] for item in batch]) 
                self.min_length_mfr = min([item['mfr1'].shape[0] for item in batch])
            max_trace1_batch = torch.stack([item['max_trace1'][:self.min_length_trace] for item in batch])
            max_trace2_batch = torch.stack([item['max_trace2'][:self.min_length_trace] for item in batch])

            max_trace1_batch = self.gaf_trace_transform(max_trace1_batch)
            max_trace2_batch = self.gaf_trace_transform(max_trace2_batch)
            for i, item in enumerate(batch):
                item['max_trace1'] = max_trace1_batch[i]
                item['max_trace2'] = max_trace2_batch[i]

        mfr1_batch = torch.stack([item['mfr1'][:self.min_length_mfr] for item in batch])
        mfr2_batch = torch.stack([item['mfr2'][:self.min_length_mfr] for item in batch])
        max_trace1_batch = torch.stack([item['max_trace1'] for item in batch])
        max_trace2_batch = torch.stack([item['max_trace2'] for item in batch])
        mfr1_batch = mfr1_batch.to(self.device)
        mfr2_batch = mfr2_batch.to(self.device)
        max_trace1_batch = max_trace1_batch.to(self.device)
        max_trace2_batch = max_trace2_batch.to(self.device)
        
        
        mfr_height = self.image_size[0] // 2
        max_trace_height = self.image_size[0] // 2
        
        mfr1_batch = F.interpolate(mfr1_batch.unsqueeze(1), size=(mfr_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(0)
        max_trace1_batch = F.interpolate(max_trace1_batch.unsqueeze(1), size=(max_trace_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(0)
        mfr2_batch = F.interpolate(mfr2_batch.unsqueeze(1), size=(mfr_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(0)
        max_trace2_batch = F.interpolate(max_trace2_batch.unsqueeze(1), size=(max_trace_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(0)
        color_start = time()
        if self.apply_randomcolormap:
            mfr1_batch = self.apply_randomcolormap(mfr1_batch)
            mfr2_batch = self.apply_randomcolormap(mfr2_batch)
            max_trace1_batch = self.apply_randomcolormap(max_trace1_batch)
            max_trace2_batch = self.apply_randomcolormap(max_trace2_batch)
        print(f"Time taken for color transform: {time() - color_start}")
        
        img1_batch = torch.cat([mfr1_batch, max_trace1_batch], dim=2)
        img2_batch = torch.cat([mfr2_batch, max_trace2_batch], dim=2)
        
        print(f"Time taken for collate function: {time() - start}")
        
        return img1_batch, img2_batch
    

class GAFTransform:
    def __call__(self, batch):
        # Assuming input batch is a tensor of shape [batch_size, height, width]
        # Normalize to [0,1] if not already done
        #batch = (batch - batch.min()) / (batch.max() - batch.min())
        # Convert to angles
        phi = torch.acos(batch.clamp(-1, 1))  # Clamp to ensure the values are within the valid range for acos
        # Compute the Gramian Angular Field for each sample in the batch
        gaf = torch.cos(phi.unsqueeze(-1) - phi.unsqueeze(-2))
        return gaf
    
class ApplyRandomColormapBatch:
    def __init__(self):
        self.colormaps = [
            ('black', 'white'),
            ('red', 'cyan'),
            ('blue', 'yellow'),
            ('green', 'magenta'),
            ('orange', 'purple'),
            ('brown', 'pink'),
            ('darkblue', 'lightgreen'),
            ('darkred', 'lightblue'),
            ('darkgreen', 'lightpink'),
            ('darkorange', 'plum'),  # replaced 'lightpurple' with 'plum'
        ]

    def __call__(self, batch):
        # Select one random colormap for the entire batch
        colormap = random.choice(self.colormaps)
        
        # Apply the colormap to each sample in the batch
        batch_list = []
        for img in batch:
            # Convert tensor to PIL Image
            img = transforms.ToPILImage()(img)
            
            # Apply colormap
            img = ImageOps.colorize(img, colormap[0], colormap[1])
            
            # Convert PIL Image back to tensor
            img = transforms.ToTensor()(img)
            
            batch_list.append(img)
        
        batch = torch.stack(batch_list)
        
        return batch

# DataLoader usage remains the same
