import models.backbones as backbones
import models.decoder as decoder

def build_model(model, n_class):
    if model=='HRNet_W48_OCR':
        backbone = backbones.HRNet.HRNet_W48()
        model = decoder.ocrnet.OCRNet(num_classes=n_class,
                                      backbone=backbone,
                                      backbone_indices=(0,))
        return model

    elif model=='UNet':
        model = backbones.unet.UNet(num_classes=n_class)
        return model

    elif model=='deeplabv3p':
        backbone = backbones.ResNet.ResNet101_vd(output_stride=8,
                                                 multi_grid=[1, 2, 4])
        model = decoder.DeepLabV3P.DeepLabV3P(num_classes=n_class,
                                              backbone=backbone,
                                              backbone_indices=[0, 3],
                                              aspp_ratios=[1, 12, 24, 36],
                                              aspp_out_channels=256)
        return model
