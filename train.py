import os
import torch
from tqdm import tqdm
from metric import *
from transform import *
from scheduler import *
from pspnet import PSPNet
from dataset import SemDataset
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

cv2.setNumThreads(0)
cudnn.benchmark = True
cv2.ocl.setUseOpenCL(False)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
writer = SummaryWriter(comment='train')

max_epoch = 100

model = PSPNet(num_classes=21).train()

transform = Compose([
    RandScale([0.5, 2],),
    RandRotate([-10, 10], padding=[255, 255, 255], ignore_label=255),
    RandomHorizontalFlip(),
    Crop([473, 473], crop_type='rand', padding=[255, 255, 255], ignore_label=255),
    ToTensor(),
    Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
])

transformTest = Compose([
    Crop([473, 473], padding=[255, 255, 255], ignore_label=255),
    ToTensor(),
    Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255], [0.229 * 255, 0.224 * 255, 0.225 * 255]),
])

data_root = '/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012'
trainTxt = '/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt'
testTxt = '/root/PycharmProjects/pythonProject/VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt'

traindataset = SemDataset(split='train', data_root=data_root, data_list=trainTxt, transform=transform)
testdataset = SemDataset(split='test', data_root=data_root, data_list=testTxt, transform=transformTest)

trainLoader = DataLoader(traindataset, batch_size=14, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)
testLoader = DataLoader(testdataset, batch_size=14, num_workers=2, shuffle=True, pin_memory=True, drop_last=True)

param_group = [
    {'params': model.backbone.parameters(), 'lr': 0.001, 'weight_decay': 1e-4, 'momentum': 0.9},
    {'params': model.master_branch.parameters(), 'lr': 0.01, 'weight_decay': 1e-4, 'momentum': 0.9},
]

crition = torch.nn.CrossEntropyLoss(ignore_index=255)
optimter = torch.optim.SGD(param_group, lr=0.1)
scheduler = Poly(optimter, num_epochs=max_epoch, iters_per_epoch=len(trainLoader))

model = torch.nn.parallel.DataParallel(model.cuda(), output_device=0)
max_mIOU = 0.78

@torch.no_grad()
def evaluate(epoch):
    cnt = 0
    total_loss = 0
    hist = np.zeros((21, 21))
    for image, label in tqdm(testLoader):
        with torch.no_grad():
            image = image.cuda(non_blocking=True).float()
            label = label.cuda(non_blocking=True).long()

        output = model(image)
        loss = crition(output, label)

        with torch.no_grad():
            cnt += 1
            total_loss += loss.item()
            hist += compute_mIoU(labels=label, preds=output)
            # if cnt % 100 == 0:
            #     writer.add_images(tag=f'epoch-label:{epoch}', img_tensor=torch.unsqueeze(label, dim=1))
            #     print(output.shape)
            #     writer.add_images(tag=f'epoch-output:{epoch}', img_tensor=torch.argmax(output, keepdim=True))

    mIoUs = per_class_iu(hist)
    print(f'test_loss: {total_loss / cnt}, mIOU: {round(np.nanmean(mIoUs) * 100, 2)}')
    writer.add_scalar('mIOU', np.nanmean(mIoUs), epoch)

for e in range(max_epoch + 1):
    cnt = 0
    epoch_loss = 0

    for image, label in tqdm(trainLoader):
        with torch.no_grad():
            image = image.cuda(non_blocking=True).float()
            label = label.cuda(non_blocking=True).long()

        output = model(image)
        loss = crition(output, label)
        scheduler.step(epoch=e-1)
        optimter.zero_grad()
        loss.backward()
        optimter.step()
        epoch_loss += loss.item()
        cnt += 1

    print(f'epoch: {e}, train loss: {epoch_loss / cnt}')

    pre_epoch_loss = epoch_loss / cnt

    model.eval()
    evaluate(e)
    model.train()

    torch.save(model.state_dict(), f'logs/Epoch{e}.pth')
