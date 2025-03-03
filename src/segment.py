from nn.UNet import UNet
import torch as th
import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
from torchvision import transforms
import tifffile
def main():
    transform = transforms.Compose([transforms.ToTensor()])
    # image = transform(np.astype(tifffile.imread("/mnt/d/d4/a118/15_d6.tif")[-1, :, :] / 255.0, np.float32))
    image = transform(Image.open("/home/quantum/MScThesisCode/dataset/images/02_C4.tif_14.png").convert("L"))
    ground = transform(Image.open("/home/quantum/MScThesisCode/dataset/ground/02_C4.tif_14.png").convert("L"))


    ground:th.Tensor = (ground > 0.5).float() 
    

    model = UNet(1, 1)
    model.load_state_dict(th.load("/home/quantum/MScThesisCode/src/nn/unet.th", weights_only=True))

    image_forward = image.unsqueeze(0)
    segment = model(image_forward)
    
    img = image.squeeze().numpy()
    grd = ground.squeeze().numpy()
    sgm = (segment > 0.5).float()
    sgm = sgm.squeeze().squeeze().detach().numpy()


    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(img, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(grd, cmap="gray")
    ax[1].set_title("Original Segmentation")
    ax[1].axis("off")

    
    ax[2].imshow(sgm, cmap="gray")
    ax[2].set_title("Inferred Segmentation")
    ax[2].axis("off")

    plt.show()

if __name__ == '__main__':
    main()