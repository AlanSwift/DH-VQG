import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"

import torch
import torch.nn as nn
import torch.nn.functional as F
from module.submodule.resnet import resnet


class CNNEncoder(nn.Module):
    def __init__(self, cnn_backend, cnn_weight_path, fixed_block, dim_cnn, dim_out=None, dropout=0.2):
        super(CNNEncoder, self).__init__()
        if cnn_backend == "resnet101":
            self.cnn = resnet(cnn_weight_path, _num_layers=101, _fixed_block=fixed_block, pretrained=True)
        else:
            raise NotImplementedError()
        self.dim_cnn = dim_cnn
        self.dim_out = dim_out
        self.require_projection = dim_out is not None
        if self.require_projection:
            self.prj = nn.Sequential(nn.Linear(self.dim_cnn, self.dim_out),
                                     nn.ReLU(),
                                     nn.Dropout(dropout))

    def forward(self, image):
        conv_feats = self.cnn(image)
        batch_size, dim_cnn, step1, step2 = conv_feats.shape
        assert dim_cnn == self.dim_cnn
        conv_feats = conv_feats.view(batch_size, dim_cnn, -1).transpose(1, 2).contiguous()
        if self.require_projection:
            conv_feats = self.prj(conv_feats)
        fc_feats = conv_feats.mean(1)
        return conv_feats, fc_feats

if __name__ == "__main__":
    import os
    from PIL import Image
    import numpy as np
    import torchvision.transforms as transforms

    image_size=576

    # train_dir = '/home/shiina/data/coco2014/train2014/pic'
    # output_dir = "/home/shiina/data/coco2014/train2014/resnetfeats"

    train_dir = '/home/shiina/data/coco2014/val2014/pic'
    output_dir = "/home/shiina/data/coco2014/val2014/resnetfeats"

    Resize = transforms.Resize((image_size, image_size))
    extractor = CNNEncoder(cnn_backend="resnet101",
                           cnn_weight_path="/home/shiina/shiina/question/aaai/data/imagenet_weights",
                           fixed_block=4,
                           dim_cnn=2048
                           ).cuda()
    extractor.eval()

    img_process = transforms.Compose([transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    with torch.no_grad():
        cnt_all = len(os.listdir(train_dir))
        cnt = 0
        for img_name in os.listdir(train_dir):
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt, cnt_all)
            img_path = os.path.join(train_dir, img_name)

            output_path = os.path.join(output_dir, img_name.replace("jpg", "npy"))
            img = Image.open(img_path).convert('RGB')
            width, height = img.size
            img = Resize(img)
            img = img_process(img)
            img = img.cuda()
            img = img.unsqueeze(0)
            feats, _ = extractor(img)
            feats = feats.squeeze(0).detach().cpu().numpy()
            np.save(output_path, feats)
