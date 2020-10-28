import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_


def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseExpNet(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        # cnv1 to cnv5b are shared between pose and explainability prediction
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7) # 这里的 +1 和 *3 是什么
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5)
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        # TODO 论文 Fig.4(b) 中第 5、6、7 个网络的分辨率降低
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[5], conv_planes[6])

        # TODO kernel_size = 1
        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if self.output_exp:
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4],   upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

            # TODO 如何保持此处 nb_ref_imgs 和输入图像的对应关系
            self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        # 预测结果 pose 是 target_image 相对于每一个 ref_imgs 的位姿. 因为 len(ref_imgs)=2, 于是 pose.size = 12
        # 另外, 注意多个 ref_imgs 相机的相对位姿是在同一个网络里同时得到的
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1)
        # input.size = torch.Size([4, 9, 128, 416]), nb_ref_imgs=2
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7)    # pose.size = torch.Size([4, 12, 1, 4])
        pose = pose.mean(3).mean(2)         # pose.size = torch.Size([4, 12])
        # TODO Empicically we found that scaling by a small constant facilitates training
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)
        '''
tensor([[-0.0013, -0.0212,  0.0132,  0.0062, -0.0466, -0.0094,  0.0207, -0.0008,
          0.0041,  0.0163,  0.0002, -0.0334],
        [-0.0014, -0.0202,  0.0086,  0.0096, -0.0417, -0.0099,  0.0100,  0.0049,
          0.0055,  0.0118, -0.0002, -0.0347],
        [-0.0062, -0.0288,  0.0214,  0.0061, -0.0511, -0.0148,  0.0235,  0.0053,
          0.0073,  0.0209,  0.0127, -0.0522],
        [-0.0078, -0.0353,  0.0133,  0.0156, -0.0572, -0.0108,  0.0092,  0.0055,
          0.0079,  0.0112,  0.0085, -0.0447]], device='cuda:0',
       grad_fn=<MeanBackward1>)

tensor([[[-1.2517e-05, -2.1237e-04,  1.3247e-04,  6.2456e-05, -4.6634e-04,
          -9.4197e-05],
         [ 2.0663e-04, -7.6395e-06,  4.0701e-05,  1.6271e-04,  1.8701e-06,
          -3.3418e-04]],

        [[-1.4472e-05, -2.0226e-04,  8.5677e-05,  9.5942e-05, -4.1658e-04,
          -9.8791e-05],
         [ 1.0026e-04,  4.9248e-05,  5.4912e-05,  1.1803e-04, -2.2200e-06,
          -3.4749e-04]],

        [[-6.2011e-05, -2.8792e-04,  2.1373e-04,  6.1065e-05, -5.1095e-04,
          -1.4782e-04],
         [ 2.3513e-04,  5.3112e-05,  7.3335e-05,  2.0878e-04,  1.2734e-04,
          -5.2197e-04]],

        [[-7.8230e-05, -3.5260e-04,  1.3343e-04,  1.5612e-04, -5.7239e-04,
          -1.0843e-04],
         [ 9.2208e-05,  5.5011e-05,  7.8559e-05,  1.1233e-04,  8.4889e-05,
          -4.4743e-04]]], device='cuda:0', grad_fn=<MulBackward0>
        '''

        if self.output_exp:
            out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]
            '''
            tgt_img.size =  torch.Size([4, 3, 128, 416])
            out_conv5.size = torch.Size([4, 256, 4, 13])
            out_conv4.size = torch.Size([4, 128, 8, 26])
            out_conv3.size = torch.Size([4, 64, 16, 52])
            out_conv2.size = torch.Size([4, 32, 32, 104])
            out_conv1.size = torch.Size([4, 16, 64, 208])
            '''

            exp_mask4 = sigmoid(self.predict_mask4(out_upconv4))
            exp_mask3 = sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
