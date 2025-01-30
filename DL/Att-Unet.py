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
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttU_Net(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, scale_factor=1):
        super(AttU_Net, self).__init__()
        filters = np.array([64, 128, 256, 512, 1024])
        filters = filters // scale_factor
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.scale_factor = scale_factor
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=n_channels, ch_out=filters[0])
        self.Conv2 = conv_block(ch_in=filters[0], ch_out=filters[1])
        self.Conv3 = conv_block(ch_in=filters[1], ch_out=filters[2])
        self.Conv4 = conv_block(ch_in=filters[2], ch_out=filters[3])
        self.Conv5 = conv_block(ch_in=filters[3], ch_out=filters[4])

        self.Up5 = up_conv(ch_in=filters[4], ch_out=filters[3])
        self.Att5 = Attention_block(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.Up_conv5 = conv_block(ch_in=filters[4], ch_out=filters[3])

        self.Up4 = up_conv(ch_in=filters[3], ch_out=filters[2])
        self.Att4 = Attention_block(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.Up_conv4 = conv_block(ch_in=filters[3], ch_out=filters[2])

        self.Up3 = up_conv(ch_in=filters[2], ch_out=filters[1])
        self.Att3 = Attention_block(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.Up_conv3 = conv_block(ch_in=filters[2], ch_out=filters[1])

        self.Up2 = up_conv(ch_in=filters[1], ch_out=filters[0])
        self.Att2 = Attention_block(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2)
        self.Up_conv2 = conv_block(ch_in=filters[1], ch_out=filters[0])

        self.Conv_1x1 = nn.Conv2d(filters[0], n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1

if __name__ == "__main__":
    train_img_dir = 'DRIVE_afteraug/training/images'
    train_mask_dir = 'DRIVE_afteraug/training/1st_manual'

    test_img_dir = 'DRIVE_afteraug/test/images'
    test_mask_dir = 'DRIVE_afteraug/test/1st_manual'

    trainloader = GetDataloader(train_img_dir, train_mask_dir, 'train', 2)
    testloader = GetDataloader(test_img_dir, test_mask_dir, 'test', 2)
    # 训练设备选择GPU还是CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttU_Net(n_channels=3, n_classes=1, scale_factor=1)
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
