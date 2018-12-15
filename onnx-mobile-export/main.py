'''Ainda não funciona!'''

import onnx
import torch
from PIL import Image
from caffe2.python import workspace, net_printer
from model import SuperResolutionNet, SRResNet
from skimage import io, transform
from torch.utils import model_zoo
import caffe2.python.onnx.backend as backend
import numpy as np
from torch.autograd import Variable
from caffe2.python.predictor import mobile_exporter

from torchvision.transforms import *

import os

img_to_tensor = ToTensor()
tensor_to_img = ToPILImage()

root_dir = 'onnx-mobile-export'

input_names = ["input1"]

output_names = ["output1"]

img_in = io.imread(os.path.join(root_dir, 'data', 'cat.jpg'))

img = transform.resize(img_in, [224, 224])

io.imsave(os.path.join(root_dir, 'data', 'cat_224x224.jpg'), img)

img = Image.open(os.path.join(root_dir, 'data', 'cat_224x224.jpg')).convert("RGB")

img_tensor = img_to_tensor(img)

img = np.array(img)
img = img.reshape(1, img.shape[2], img.shape[1], img.shape[0]).astype(np.float32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch_model = SRResNet(rescale_factor=4, n_filters=64, n_blocks=8).to(device).float()
#torch_model = SuperResolutionNet(upscale_factor=3)

model_url = 'https://s3.amazonaws.com/pytorch/demos/srresnet-e10b2039.pth'
#model_url = 'https://s3.amazonaws.com/pytorch/test_data/export/superres_epoch100-44c6958e.pth'
batch_size = 1

map_location = lambda storage, loc: storage

if torch.cuda.is_available():
    map_location = None
torch_model.load_state_dict(model_zoo.load_url(model_url, map_location=map_location))


torch_model.train(False)

x = Variable(torch.randn(batch_size, 3, 224, 224, requires_grad=True, dtype=torch.float32)).to(device)
torch_out = torch.onnx._export(torch_model, x, os.path.join(root_dir, 'data', 'super_resolution.onnx'), input_names=input_names, output_names=output_names, export_params=True)
model = onnx.load(os.path.join(root_dir, 'data', 'super_resolution.onnx'))
onnx.checker.check_model(model)
print(1)
prep_backend = backend.prepare(model, device='CUDA:0')
print(model.graph.input[0].name)
W = {model.graph.input[0].name: img}
print(W['input1'].shape)
c2_out = prep_backend.run(W)[0]
c2_out = np.absolute(c2_out) 
print(c2_out, c2_out.shape)
c2_out = c2_out[0].reshape(c2_out.shape[3], c2_out.shape[2], c2_out.shape[1]).clip(0, 255).astype(np.uint8)
img_out_pil_c = Image.fromarray(c2_out)
img_out_pil_c.save(os.path.join(root_dir, 'data', 'cat_superres_caffe2.jpg'))
raise Exception('Parando aqui!')

img = img.reshape(img.shape[2], img.shape[1], img.shape[0]).astype(np.float32)
print(img.shape)
input_data = img_to_tensor(img).view(1, img.shape[2], img.shape[1], img.shape[0]).to(device).float()
torch_out = torch_model(input_data)
np.testing.assert_almost_equal(torch_out.data.cpu().numpy(), c2_out, decimal=3)

c2_workspace = prep_backend.workspace
c2_model = prep_backend.predict_net

init_net, predict_net = mobile_exporter.Export(c2_workspace, c2_model, c2_model.external_input)

with open(os.path.join(root_dir, 'data', 'init_net.pb'), "wb") as fopen:
    fopen.write(init_net.SerializeToString())
with open(os.path.join(root_dir, 'data', 'predict_net.pb'), "wb") as fopen:
    fopen.write(predict_net.SerializeToString())

workspace.RunNetOnce(init_net)
workspace.RunNetOnce(predict_net)

workspace.FeedBlob('input1', img.reshape(1, img.shape[2], img.shape[1], img.shape[0]).astype(np.float32))
workspace.RunNetOnce(predict_net)
img_out_c = workspace.FetchBlob('output1')


print(input_data.size())
img_out = torch_model(input_data).cpu()

img_out_tensor = torch.squeeze(img_out, 0).clamp(0, 1)
print('Tensor de imagem: ', img_out_tensor)

print('Array de imagem:', img_out_c)
img_out_pil_c = Image.fromarray(img_out_c)
img_out_pil_c.save(os.path.join(root_dir, 'data', 'cat_superres_caffe2.jpg'))
img_out_pil = tensor_to_img(img_out_tensor)
img_out_pil.save(os.path.join(root_dir, 'data', 'cat_superres.jpg'))
print(img_out_pil.size)

print('Concluído!!!')