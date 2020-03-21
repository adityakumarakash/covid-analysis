import torch
import numpy as np
import copy
import matplotlib.cm as mpl_color_map
import pdb
from torchvision import transforms
from PIL import Image
import skimage
from skimage import util

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
    
    def close(self):
        self.hook.remove()

def threshold_activations(activation, threshold=False):
    if not threshold:
        return activation
    activation[activation<200] = 70
    return activation
        
# Refrerence from https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/misc_functions.py
def apply_colormap_on_image(org_im, activation, colormap_name='jet', threshold=False):
    """
    Apply heatmap on image
    Args:
    org_img (PIL img): Original image
    activation_map (numpy arr): Activation map (grayscale) 0-255
    colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(threshold_activations(activation, threshold))
    
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    
    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return heatmap_on_image
        
        
def get_cam_images(model, last_conv_layer, images, corona_idx=2):
    # corona_idx is the index of the class which corresponds to covid19
    # Add hook to get last conv layer's features
    hook = Hook(model._modules.get(last_conv_layer))
    model.eval()

    # Get the weights for the last global average pooled layer
    params = list(model.parameters())
    weight_softmax = params[-2]
    
    pred = model(images)
    
    def get_cam_activations(conv_features, weights, class_idx):
        # conv_features = batch * channels * w * h
        # weights = classes * channels
        # class_idx = class for which cam is to be generated
        _, _, w, h = conv_features.shape
        class_weights = weights[class_idx]  # dim = channels
        cam_features = torch.sum(conv_features * class_weights.view(1, -1, 1, 1), 1)
        cam_features -= cam_features.view(-1, w*h).min(1)[0].view(-1, 1, 1)
        cam_features /= cam_features.view(-1, w*h).max(1)[0].view(-1, 1, 1)
        return np.uint8(255 * cam_features.cpu().detach().numpy())
    
    cam_activations = get_cam_activations(hook.output, weight_softmax, corona_idx)
    cam_activations = [util.img_as_ubyte(skimage.transform.resize(inp, images[0].shape[-2:])) for inp in cam_activations]
    hook.close()
    transform = transforms.ToPILImage()
    pil_images = [transform(img) for img in images.cpu()]
    color_heatmaps = [apply_colormap_on_image(img, act) for img, act in zip(pil_images, cam_activations)]
    return color_heatmaps
        