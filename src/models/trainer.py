import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.utils.data as data

import src.models.unet as unet

batch_size = 4
#img_size = 100*2 + 1
img_size = 256
lr = 0.0002
epoch = 100

out_dir = "/Users/danielfisher/Projects/kcl-ltss-bioatm/data/cnn_output"
img_dir = "/Users/danielfisher/Projects/kcl-ltss-bioatm/data/cnn_input"

#img_dir = '/Users/danielfisher/Projects/kcl-ltss-bioatm/data/maps/'

#img_data = dset.ImageFolder(root=img_dir, transform = transforms.Compose([
#                                            transforms.Scale(size=img_size),
#                                            transforms.CenterCrop(size=(img_size,img_size*2)),
#                                            transforms.ToTensor(),
#                                            ]))

img_data = dset.DatasetFolder(root=img_dir, loader=np.load, extensions=['npy',])

img_batch = data.DataLoader(img_data, batch_size=batch_size,
                            shuffle=True, num_workers=2)


# input channels (10), output_channels (also 10?), filter size (64 seems large!)
generator = nn.DataParallel(unet.UnetGenerator(9, 3, 64), device_ids=1)
recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

# training
f = open(out_dir + '/unet_mse_loss', 'w')
for i in range(epoch):
    for _, (image, label) in enumerate(img_batch):
        #satel_image, map_image = torch.chunk(image, chunks=2, dim=3)
        satel_image, map_image = torch.split(image, split_size_or_sections=[9,1], dim=1)

        gen_optimizer.zero_grad()

        x = Variable(satel_image).double() #.cuda(0)
        y_ = Variable(map_image).double() #.cuda(0)
        y = generator.forward(x)

        loss = recon_loss_func(y, y_)
        f.write(str(loss) + "\n")
        loss.backward()
        gen_optimizer.step()

        if _ % 400 == 0:
            print(i)
            print(loss)
            v_utils.save_image(x.cpu().data, out_dir + "/original_image_{}_{}.png".format(i, _))
            v_utils.save_image(y_.cpu().data, out_dir + "/label_image_{}_{}.png".format(i, _))
            v_utils.save_image(y.cpu().data, out_dir + "/gen_image_{}_{}.png".format(i, _))
            torch.save(generator, './model/{}.pkl'.format(unet))
