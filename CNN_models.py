import torchvision.models as models
from make_models import save_model_onnx

alexnet = models.alexnet(pretrained=True)
# vgg16 = models.vgg16(pretrained=True)

# make onnx model
save_model_onnx(alexnet)
