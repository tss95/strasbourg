import torch.nn.functional as F
import torch
import random
import matplotlib.pyplot as plt
from time import time

class CollateFn:
    def __init__(self, gaf_trace_transform = True, apply_randomcolormap= True, color_maps = ['viridis', 'plasma', 'inferno', 'magma'], image_size=(256, 256)):
        self.image_size = image_size  # image_size should be (height, width)
        self.gaf_trace_transform = None
        if gaf_trace_transform:
            self.gaf_trace_transform = GAFTransform()
        self.apply_randomcolormap = None
        if apply_randomcolormap:
            self.apply_randomcolormap = ApplyRandomColormapBatch(colormaps=color_maps)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.min_length = None

    def __call__(self, batch):
        
        start = time()
        processed_data = {"img1": [], "img2" : []}
        
        for item in batch:
            item = {k: v.to(self.device) for k, v in item.items()}
            
        if self.gaf_trace_transform:
            if not self.min_length:
                self.min_length = min([item['max_trace1'].shape[0] for item in batch]) 
            max_trace1_batch = torch.stack([item['max_trace1'][:self.min_length] for item in batch])
            max_trace2_batch = torch.stack([item['max_trace2'][:self.min_length] for item in batch])

            max_trace1_batch = self.gaf_trace_transform(max_trace1_batch)
            max_trace2_batch = self.gaf_trace_transform(max_trace2_batch)
            for i, item in enumerate(batch):
                item['max_trace1'] = max_trace1_batch[i]
                item['max_trace2'] = max_trace2_batch[i]

        for item in batch:
            # Ensure both tensors are on the correct device before processing
            mfr1, max_trace1 = item['mfr1'], item['max_trace1']
            mfr2, max_trace2 = item['mfr2'], item['max_trace2']
            print(f"mfr1 shape: {mfr1.shape}")
            print(f"max_trace1 shape: {max_trace1.shape}")

            # Calculate individual heights for mfr and max_trace
            mfr_height = self.image_size[0] // 2
            max_trace_height = self.image_size[0] // 2

            # Interpolate mfr and max_trace separately to occupy half the image height each 
            mfr1 = F.interpolate(mfr1.unsqueeze(0), size=(mfr_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(0)
            max_trace1 = F.interpolate(max_trace1.unsqueeze(0).unsqueeze(0), size=(max_trace_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(0)
            mfr2 = F.interpolate(mfr2.unsqueeze(0), size=(mfr_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(0)
            max_trace2 = F.interpolate(max_trace2.unsqueeze(0).unsqueeze(0), size=(max_trace_height, self.image_size[1]), mode='bilinear', align_corners=False).squeeze(0)
            
            # Ensure the dimensions are consistent
            if mfr1.size(1) != max_trace1.size(1) or mfr1.size(2) != max_trace1.size(2):
                raise RuntimeError(f"Dimension mismatch: MFR dimensions {mfr1.size()} do not match Max Trace dimensions {max_trace1.size()} after interpolation.")

            # Concatenate mfr and max_trace along the height dimension
            img1 = torch.cat([mfr1, max_trace1], dim=1)  # Change to dim=1 to concatenate along the channel dimension
            img2 = torch.cat([mfr2, max_trace2], dim=1)

            processed_data["img1"].append(img1)
            processed_data["img2"].append(img2)

        processed_data['img1'] = torch.stack(processed_data['img1'])
        processed_data['img2'] = torch.stack(processed_data['img2'])
        
        if self.apply_randomcolormap:
            processed_data['img1'] = self.apply_randomcolormap(processed_data['img1'])
            processed_data['img2'] = self.apply_randomcolormap(processed_data['img2'])
        print(f"Time taken for collate function: {time() - start}")
        return processed_data

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
    def __init__(self, colormaps):
        self.colormaps = colormaps

    def __call__(self, batch):
        # Select one random colormap for the entire batch
        colormap = random.choice(self.colormaps)
        cmap = plt.cm.get_cmap(colormap)
        # Apply the colormap to each sample in the batch
        batch = cmap(batch.numpy())[:, :, :, :3]  # Apply colormap and ignore the alpha channel
        return torch.from_numpy(batch).permute(0, 3, 1, 2)  # Rearrange dimensions to [batch_size, channels, height, width]


# DataLoader usage remains the same
