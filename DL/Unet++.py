import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import copy
import torchsummary
from thop import profile
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from Dataset import GetDataloader
class ContinusParalleConv(nn.Module):
    # 一个连续的卷积模块，包含BatchNorm 在前 和 在后 两种模式
    def __init__(self, in_channels, out_channels, pre_Batch_Norm=True):
        super(ContinusParalleConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        if pre_Batch_Norm:
            self.Conv_forward = nn.Sequential(
                nn.BatchNorm2d(self.in_channels),
                nn.ReLU(),
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1))

        else:
            self.Conv_forward = nn.Sequential(
                nn.Conv2d(self.in_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1),
                nn.BatchNorm2d(self.out_channels),
                nn.ReLU())

    def forward(self, x):
        x = self.Conv_forward(x)
        return x


class UnetPlusPlus(nn.Module):
    def __init__(self, num_classes, deep_supervision=False):
        super(UnetPlusPlus, self).__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.filters = [64, 128, 256, 512, 1024]

        self.CONV3_1 = ContinusParalleConv(512 * 2, 512, pre_Batch_Norm=True)

        self.CONV2_2 = ContinusParalleConv(256 * 3, 256, pre_Batch_Norm=True)
        self.CONV2_1 = ContinusParalleConv(256 * 2, 256, pre_Batch_Norm=True)

        self.CONV1_1 = ContinusParalleConv(128 * 2, 128, pre_Batch_Norm=True)
        self.CONV1_2 = ContinusParalleConv(128 * 3, 128, pre_Batch_Norm=True)
        self.CONV1_3 = ContinusParalleConv(128 * 4, 128, pre_Batch_Norm=True)

        self.CONV0_1 = ContinusParalleConv(64 * 2, 64, pre_Batch_Norm=True)
        self.CONV0_2 = ContinusParalleConv(64 * 3, 64, pre_Batch_Norm=True)
        self.CONV0_3 = ContinusParalleConv(64 * 4, 64, pre_Batch_Norm=True)
        self.CONV0_4 = ContinusParalleConv(64 * 5, 64, pre_Batch_Norm=True)

        self.stage_0 = ContinusParalleConv(3, 64, pre_Batch_Norm=False)
        self.stage_1 = ContinusParalleConv(64, 128, pre_Batch_Norm=False)
        self.stage_2 = ContinusParalleConv(128, 256, pre_Batch_Norm=False)
        self.stage_3 = ContinusParalleConv(256, 512, pre_Batch_Norm=False)
        self.stage_4 = ContinusParalleConv(512, 1024, pre_Batch_Norm=False)

        self.pool = nn.MaxPool2d(2)

        self.upsample_3_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)

        self.upsample_2_1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.upsample_2_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)

        self.upsample_1_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.upsample_1_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.upsample_0_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_3 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.upsample_0_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)

        # 分割头
        self.final_super_0_1 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_2 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_3 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )
        self.final_super_0_4 = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.num_classes, 3, padding=1),
        )

    def forward(self, x):
        x_0_0 = self.stage_0(x)
        x_1_0 = self.stage_1(self.pool(x_0_0))
        x_2_0 = self.stage_2(self.pool(x_1_0))
        x_3_0 = self.stage_3(self.pool(x_2_0))
        x_4_0 = self.stage_4(self.pool(x_3_0))

        x_0_1 = torch.cat([self.upsample_0_1(x_1_0), x_0_0], 1)
        x_0_1 = self.CONV0_1(x_0_1)

        x_1_1 = torch.cat([self.upsample_1_1(x_2_0), x_1_0], 1)
        x_1_1 = self.CONV1_1(x_1_1)

        x_2_1 = torch.cat([self.upsample_2_1(x_3_0), x_2_0], 1)
        x_2_1 = self.CONV2_1(x_2_1)

        x_3_1 = torch.cat([self.upsample_3_1(x_4_0), x_3_0], 1)
        x_3_1 = self.CONV3_1(x_3_1)

        x_2_2 = torch.cat([self.upsample_2_2(x_3_1), x_2_0, x_2_1], 1)
        x_2_2 = self.CONV2_2(x_2_2)

        x_1_2 = torch.cat([self.upsample_1_2(x_2_1), x_1_0, x_1_1], 1)
        x_1_2 = self.CONV1_2(x_1_2)

        x_1_3 = torch.cat([self.upsample_1_3(x_2_2), x_1_0, x_1_1, x_1_2], 1)
        x_1_3 = self.CONV1_3(x_1_3)

        x_0_2 = torch.cat([self.upsample_0_2(x_1_1), x_0_0, x_0_1], 1)
        x_0_2 = self.CONV0_2(x_0_2)

        x_0_3 = torch.cat([self.upsample_0_3(x_1_2), x_0_0, x_0_1, x_0_2], 1)
        x_0_3 = self.CONV0_3(x_0_3)

        x_0_4 = torch.cat([self.upsample_0_4(x_1_3), x_0_0, x_0_1, x_0_2, x_0_3], 1)
        x_0_4 = self.CONV0_4(x_0_4)

        if self.deep_supervision:
            out_put1 = self.final_super_0_1(x_0_1)
            out_put2 = self.final_super_0_2(x_0_2)
            out_put3 = self.final_super_0_3(x_0_3)
            out_put4 = self.final_super_0_4(x_0_4)
            return [out_put1, out_put2, out_put3, out_put4]
        else:
            return self.final_super_0_4(x_0_4)



if __name__ == "__main__":
    train_img_dir = 'DRIVE_afteraug/training/images'
    train_mask_dir = 'DRIVE_afteraug/training/1st_manual'

    test_img_dir = 'DRIVE_afteraug/test/images'
    test_mask_dir = 'DRIVE_afteraug/test/1st_manual'

    trainloader = GetDataloader(train_img_dir, train_mask_dir, 'train', 2)
    testloader = GetDataloader(test_img_dir, test_mask_dir, 'test', 2)
    # 训练设备选择GPU还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnetPlusPlus(num_classes=3, deep_supervision=False)
    model.to(device)

    print(sum(p.numel() for p in model.parameters()))
    input = torch.randn(1, 3, 512, 512)
    input = input.to(device)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
    torchsummary.summary(model, (3, 512, 512))

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([1.5]).to(device))
    criterion.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #余弦退火调整学习率
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0.000001)

    train_loss = []
    test_loss = []


    def train():
        mloss = []
        for data in trainloader:
            datavalue, datalabel = data
            datavalue, datalabel = datavalue.to(device), datalabel.to(device)
            datalabel_pred = model(datavalue)
            loss = criterion(datalabel_pred, datalabel)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            mloss.append(loss.item())

        epoch_train_loss = torch.mean(torch.Tensor(mloss)).item()
        train_loss.append(epoch_train_loss)
        print("*" * 10, epoch, "*" * 10)
        print('训练集损失:', epoch_train_loss)
        test()


    # 测试函数
    def test():
        mloss = []
        with torch.no_grad():
            for testdata in testloader:
                testdatavalue, testdatalabel = testdata
                testdatavalue, testdatalabel = testdatavalue.to(device), testdatalabel.to(device)
                testdatalabel_pred = model(testdatavalue)
                loss = criterion(testdatalabel_pred, testdatalabel)
                mloss.append(loss.item())
            epoch_test_loss = torch.mean(torch.Tensor(mloss)).item()
            test_loss.append(epoch_test_loss)
            print('测试集损失', epoch_test_loss)


    bestmodel = None
    bestepoch = None
    bestloss = np.inf

    for epoch in range(1, 31):
        train()
        if test_loss[epoch - 1] < bestloss:
            bestloss = test_loss[epoch - 1]
            bestepoch = epoch
            bestmodel = copy.deepcopy(model)

    print("最佳轮次为:{},最佳损失为:{}".format(bestepoch, bestloss))

    torch.save(model, "训练好的模型权重/lastmodel.pt")
    torch.save(bestmodel, "训练好的模型权重/bestmodel.pt")
    torch.save(model.state_dict(), "trained_weights/lastmodel.pth")
    torch.save(bestmodel.state_dict(), "trained_weights/bestmodel.pth")

    plt.plot(train_loss)
    plt.plot(test_loss)
    plt.legend(['train', 'test'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('loss.png')

    #画ROC曲线并计算AUC，保存
    model = torch.load("训练好的模型权重/bestmodel.pt")
    model.eval()

    y_score = []
    y_true = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            y_score.append(outputs.cpu().numpy())
            y_true.append(labels.cpu().numpy())
    y_score = np.concatenate(y_score).reshape(-1)
    y_true = np.concatenate(y_true).reshape(-1)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f'AUC:{roc_auc:.4f}')

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC.png')

    # 画PR曲线并计算AP，保存
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    print(f'AP:{average_precision:.4f}')

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange', lw=lw, label='PR curve (area = %0.2f)' % average_precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.legend(loc="lower right")
    plt.savefig('PR.png')

    # 计算测试集上的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            total += labels.size(0) * labels.size(2) * labels.size(3)
            correct += (outputs == labels).sum().item()
    # 小数点后四位
    print(f'ACC：{1.0 * correct / total * 100:.4f}%')

    # 计算dice score
    dice = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            dice += 2 * (outputs * labels).sum().item() / (outputs.sum().item() + labels.sum().item())
            total += 1
    # 小数点后四位
    print(f'dice score：{dice / total:.4f}')

    # 计算IOU
    iou = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            outputs = outputs.long()
            labels = labels.long()
            intersection = (outputs & labels).sum().item()
            union = (outputs | labels).sum().item()
            iou += intersection / union
            total += 1
    print(f'IOU:{iou / total:.4f}')

    # 计算F1
    f1 = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            outputs = outputs.long()
            labels = labels.long()
            intersection = (outputs & labels).sum().item()
            precision = intersection / outputs.sum().item()
            recall = intersection / labels.sum().item()
            f1 += 2 * precision * recall / (precision + recall)
            total += 1
    print(f'F1:{f1 / total:.4f}')

    # 计算SE
    se = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            outputs = outputs.long()
            labels = labels.long()
            tp = (outputs & labels).sum().item()
            fp = (outputs & (1 - labels)).sum().item()
            se += tp / (tp + fp)
            total += 1
    print(f'SE:{se / total:.4f}')

    # 计算SP
    sp = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            outputs = outputs.long()
            labels = labels.long()
            tn = ((1 - outputs) & (1 - labels)).sum().item()
            fp = ((1 - outputs) & labels).sum().item()
            sp += tn / (tn + fp)
            total += 1
    print(f'SP:{sp / total:.4f}')

    print("训练最后一轮模型的性能：")
    model = torch.load("训练好的模型权重/lastmodel.pt")
    model.eval()

    y_score = []
    y_true = []
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            y_score.append(outputs.cpu().numpy())
            y_true.append(labels.cpu().numpy())
    y_score = np.concatenate(y_score).reshape(-1)
    y_true = np.concatenate(y_true).reshape(-1)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    print(f'AUC:{roc_auc:.4f}')

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('ROC1.png')

    # 画PR曲线并计算AP，保存
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    average_precision = average_precision_score(y_true, y_score)
    print(f'AP:{average_precision:.4f}')

    plt.figure()
    lw = 2
    plt.plot(recall, precision, color='darkorange', lw=lw, label='PR curve (area = %0.2f)' % average_precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall curve')
    plt.legend(loc="lower right")
    plt.savefig('PR1.png')

    # 计算测试集上的准确率
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            total += labels.size(0) * labels.size(2) * labels.size(3)
            correct += (outputs == labels).sum().item()
    # 小数点后四位
    print(f'ACC：{1.0 * correct / total * 100:.4f}%')

    # 计算dice score
    dice = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            dice += 2 * (outputs * labels).sum().item() / (outputs.sum().item() + labels.sum().item())
            total += 1
    # 小数点后四位
    print(f'dice score：{dice / total:.4f}')

    # 计算IOU
    iou = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            outputs = outputs.long()
            labels = labels.long()
            intersection = (outputs & labels).sum().item()
            union = (outputs | labels).sum().item()
            iou += intersection / union
            total += 1
    print(f'IOU:{iou / total:.4f}')

    # 计算F1
    f1 = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            outputs = outputs.long()
            labels = labels.long()
            intersection = (outputs & labels).sum().item()
            precision = intersection / outputs.sum().item()
            recall = intersection / labels.sum().item()
            f1 += 2 * precision * recall / (precision + recall)
            total += 1
    print(f'F1:{f1 / total:.4f}')

    # 计算SE
    se = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            outputs = outputs.long()
            labels = labels.long()
            tp = (outputs & labels).sum().item()
            fp = (outputs & (1 - labels)).sum().item()
            se += tp / (tp + fp)
            total += 1
    print(f'SE:{se / total:.4f}')

    # 计算SP
    sp = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)
            outputs[outputs > 0.5] = 1
            outputs[outputs <= 0.5] = 0
            outputs = outputs.long()
            labels = labels.long()
            tn = ((1 - outputs) & (1 - labels)).sum().item()
            fp = ((1 - outputs) & labels).sum().item()
            sp += tn / (tn + fp)
            total += 1
    print(f'SP:{sp / total:.4f}')
