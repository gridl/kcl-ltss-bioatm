import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.utils as v_utils


import src.models.unet as unet

batch_size = 4
img_size = 256
lr = 0.0002
epoch = 100

generator = nn.DataParallel(unet.UnetGenerator(3, 3, 64), device_ids=1)
recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)

# training
f = open('./unet_mse_loss', 'w')
for i in range(epoch):
    for _, (image, label) in enumerate(img_batch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3)

        gen_optimizer.zero_grad()

        x = Variable(satel_image).cuda(0)
        y_ = Variable(map_image).cuda(0)
        y = generator.forward(x)

        loss = recon_loss_func(y, y_)
        f.write(str(loss) + "\n")
        loss.backward()
        gen_optimizer.step()

        if _ % 400 == 0:
            print(i)
            print(loss)
            v_utils.save_image(x.cpu().data, "./result/original_image_{}_{}.png".format(i, _))
            v_utils.save_image(y_.cpu().data, "./result/label_image_{}_{}.png".format(i, _))
            v_utils.save_image(y.cpu().data, "./result/gen_image_{}_{}.png".format(i, _))
            torch.save(generator, './model/{}.pkl'.format(args.network))
