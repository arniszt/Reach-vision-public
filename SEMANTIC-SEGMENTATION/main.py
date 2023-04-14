import os, csv, torch, numpy, scipy.io, PIL.Image, torchvision.transforms
# Our libs
from mit_semseg.models import ModelBuilder, SegmentationModule
from mit_semseg.utils import colorEncode
import cv2
from PIL import Image
import numpy as np
#2.136.142.197/32



# SSH connection to remote machine







'''
2.136.142.197/32
chmod 400 SemanticSegmentation.pem
ssh -i "SemanticSegmentation.pem" ubuntu@ec2-13-51-252-124.eu-north-1.compute.amazonaws.com
yes

sudo apt-get update
sudo apt-get install -y nvidia-driver-450

sudo apt-get install -y nvidia-cuda-toolkit
scp -r /Users/username/Projects/myproject username@<instance-ip-address>:/home/username/myproject
[scp -r /Users/arniszt/Downloads/Mckinsey/MACHINE\ LEARNING/REACH-VISION arniszt(ubuntu?)@3.235.22.155:/home/arniszt/myproject]

alternativament:
git clone https://github.com/arniszt/my-reach-vision/

'''



colors = scipy.io.loadmat('data/color150.mat')['colors']
names = {}
with open('data/object150_info.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        names[int(row[0])] = row[5].split(";")[0]

image_index = 0

image_name = 'restaurant.png'

def visualize_result(img, pred, name, index=None):
    # filter prediction class if requested
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        print(f'{names[index+1]}:')
    global image_index
    # colorize prediction
    pred_color = colorEncode(pred, colors).astype(numpy.uint8)

    # aggregate images and save
    im_vis = numpy.concatenate((img, pred_color), axis=1)
    #display(PIL.Image.fromarray(im_vis))
    image=Image.fromarray(im_vis)

    image.save(image_name +"."+ str(image_index)+ "-"+ name+ ".jpg")
    
    image_index = image_index+1
    #time.sleep(5)


# Network Builders
net_encoder = ModelBuilder.build_encoder(
    arch='resnet50dilated',
    fc_dim=2048,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/encoder_epoch_20.pth')
net_decoder = ModelBuilder.build_decoder(
    arch='ppm_deepsup',
    fc_dim=2048,
    num_class=150,
    weights='ckpt/ade20k-resnet50dilated-ppm_deepsup/decoder_epoch_20.pth',
    use_softmax=True)

crit = torch.nn.NLLLoss(ignore_index=-1)
segmentation_module = SegmentationModule(net_encoder, net_decoder, crit)
segmentation_module.eval()
segmentation_module.cuda()

pil_to_tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], # These are RGB mean+std values
        std=[0.229, 0.224, 0.225])  # across a large photo dataset.
])


#OPEN IMAGE
pil_image = PIL.Image.open(image_name).convert('RGB')
#RESIZE IMAGE
res_x, res_y = pil_image.size
if res_x > 1000:
    res_y =int((res_y * 1000)/res_x)
    res_x =int(1000)
img = pil_image.resize((res_x, res_y), resample=Image.BOX)
img_array = numpy.array(img)

#IMAGE TO TENSOR
img_data = pil_to_tensor(img)
singleton_batch = {'img_data': img_data[None].cuda()}
output_size = img_data.shape[1:]

#COMPUTE THE SCORES
with torch.no_grad():
    o_s = torch.Size((res_y,res_x))
    scores = segmentation_module(singleton_batch, segSize=o_s)


_, pred = torch.max(scores, dim=1)
pred = pred.cpu()[0].numpy()

#BUILD THE DICTIONARY: 'NAME' -> 'SCORE'
D = {}
for i in range(len(names)):
    D[names[i+1]] = np.count_nonzero(pred.flatten() == i)

#SORT THE DICTIONARY AND PICK THE HIGHEST ENTRIES IN SCORE
predicted_classes = numpy.bincount(pred.flatten()).argsort()[::-1]

#INIT RESULT AND INDEX
result = []
i = 0

#print("PREDICTED CLASSES")
#for c in predicted_classes:
#    print("class -> ", names[int(c)+1], " has a SCORE OF: ", D[names[int(c)+1]])

c3 = predicted_classes[:10][2]
for c in predicted_classes[:10]:
    scorec = D[names[int(c)+1]]
    if scorec > (D[names[int(c3)+1]]/5) or (i < 4 and scorec > 100):
       visualize_result(img, pred, names[int(c)+1], c)
       result.append(names[int(c)+1])
    i = i+1
print(result)