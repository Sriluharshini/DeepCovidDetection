from __future__ import print_function
import torch, os, copy, time, pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.models.resnet import ResNet, BasicBlock
from torch.nn import Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Sequential
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image  
import pandas as pd
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix
import glob, pickle
import seaborn as sn
import argparse
from IPython.display import Image, display

start_time = time.time()

parser = argparse.ArgumentParser(description='COVID-19 Detection from X-ray Images')
parser.add_argument('--test_covid_path', type=str, default='/content/drive/My Drive/ColabNotebooks/DeepCovid-master/data/val/covid/',
                    help='COVID-19 test samples directory')
parser.add_argument('--test_non_covid_path', type=str, default='/content/drive/My Drive/ColabNotebooks/DeepCovid-master/data/val/non/',
                    help='Non-COVID test samples directory')
parser.add_argument('--trained_model_path', type=str, default='/content/drive/My Drive/ColabNotebooks/DeepCovid-master/ResNet18/covid_resnet18_epoch2.pt',
                    help='The path and name of trained model')
parser.add_argument('--cut_off_threshold', type=float, default=0.2,
                    help='cut-off threshold. Any sample with probability higher than this is considered COVID-19 (default: 0.2)')
parser.add_argument('--batch_size', type=int, default=20,
                    help='input batch size for training (default: 20)')
parser.add_argument('--num_workers', type=int, default=0,
                    help='number of workers to train (default: 0)')

args = parser.parse_args()

# Utility function to find sensitivity and specificity for different cut-off thresholds
def find_sens_spec(covid_prob, noncovid_prob, thresh):
    sensitivity = (covid_prob >= thresh).sum() / (len(covid_prob) + 1e-10)
    specificity = (noncovid_prob < thresh).sum() / (len(noncovid_prob) + 1e-10)
    print("sensitivity= %.3f, specificity= %.3f" % (sensitivity, specificity))
    return sensitivity, specificity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_names = ['covid', 'non']

# Test on trained model
model_name = args.trained_model_path

try:
    torch.serialization.add_safe_globals([
        ResNet, BasicBlock, Conv2d, Linear, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Sequential
    ])
    model = torch.load(model_name, map_location='cpu')
    print("Model loaded successfully with weights_only=True")
except Exception as e:
    print(f"Failed to load with weights_only=True: {e}")
    model = torch.load(model_name, map_location='cpu', weights_only=False)
    print("Model loaded with weights_only=False (fallback)")

model.eval()

# loading new images
imsize = 224
loader = transforms.Compose([transforms.Resize(imsize),
                             transforms.CenterCrop(224),
                             transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = PIL.Image.open(image_name).convert("RGB")  # Use fully qualified PIL.Image.open
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)
    return image

sm = torch.nn.Softmax(dim=1)

# Get the predicted probabilities of all samples
image_extensions = tuple(IMG_EXTENSIONS)
test_covid = [f for f in glob.glob(f"{args.test_covid_path}**/*", recursive=True) if f.lower().endswith(image_extensions)]
test_non = [f for f in glob.glob(f"{args.test_non_covid_path}**/*", recursive=True) if f.lower().endswith(image_extensions)]

covid_pred = np.zeros([len(test_covid), 1]).astype(int)
non_pred = np.zeros([len(test_non), 1]).astype(int)

covid_prob = np.zeros([len(test_covid), 1])
non_prob = np.zeros([len(test_non), 1])

for i in range(len(test_covid)):
    cur_img = image_loader(test_covid[i])
    model_output = model(cur_img)
    cur_pred = model_output.max(1, keepdim=True)[1]
    cur_prob = sm(model_output)
    covid_prob[i, :] = cur_prob.data.numpy()[0, 0]
    print("%03d Covid predicted label: %s" % (i, class_names[cur_pred.item()]))

for i in range(len(test_non)):
    cur_img = image_loader(test_non[i])
    model_output = model(cur_img)
    cur_pred = model_output.max(1, keepdim=True)[1]
    cur_prob = sm(model_output)
    non_prob[i, :] = cur_prob.data.numpy()[0, 0]
    print("%03d Non-Covid predicted label: %s" % (i, class_names[cur_pred.item()]))

# find sensitivity and specificity
thresh = args.cut_off_threshold
sensitivity_40, specificity = find_sens_spec(covid_prob, non_prob, thresh)

# Derive labels based on probabilities and cut-off threshold
covid_pred = np.where(covid_prob > thresh, 1, 0)
non_pred = np.where(non_prob > thresh, 1, 0)

# derive confusion-matrix
covid_list = [int(covid_pred[i][0]) for i in range(len(covid_pred))]
non_list = [int(non_pred[i][0]) for i in range(len(non_pred))]

covid_count = [(x, covid_list.count(x)) for x in set(covid_list)]
non_count = [(x, non_list.count(x)) for x in set(non_list)]

y_pred_list = covid_list + non_list
y_test_list = [1 for i in range(len(covid_list))] + [0 for i in range(len(non_list))]

y_pred = np.asarray(y_pred_list, dtype=np.int64)
y_test = np.asarray(y_test_list, dtype=np.int64)

cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure(figsize=(8, 6))
df_cm = pd.DataFrame(cnf_matrix, index=[i for i in class_names],
                     columns=[i for i in class_names])
ax = sn.heatmap(df_cm, cmap=plt.cm.Blues, annot=True, cbar=False, fmt='g',
                xticklabels=['Non-COVID', 'COVID-2019'], yticklabels=['Non-COVID', 'COVID-2019'])
ax.set_title("Confusion Matrix")
plt.savefig('/content/drive/My Drive/ColabNotebooks/DeepCovid-master/ResNet18/Results/confusion_matrix_100.png', bbox_inches='tight')
plt.show()
plt.close()

# Plot the predicted probability distribution
plt.figure(figsize=(10, 8))
plt.subplot(211)
bins = np.linspace(0, 1, 25)
plt.hist(covid_prob, bins, color='blue', histtype='bar', label='Probabilities of COVID-19 Samples')
plt.ylim([0, 10])
plt.legend(loc='upper center')
plt.subplot(212)
plt.hist(non_prob, bins, color='green', label='Probabilities of Non-COVID Samples')
plt.legend(loc='upper center')
plt.savefig('/content/drive/My Drive/ColabNotebooks/DeepCovid-master/ResNet18/Results/scores_histogram_100.png', bbox_inches='tight')
plt.show()
plt.close()

# ROC Curve and AUC
from sklearn.metrics import roc_curve, roc_auc_score
y_test_res18 = [1 for i in range(len(covid_prob))] + [0 for i in range(len(non_prob))]
y_pred_res18 = [float(covid_prob[i][0]) for i in range(len(covid_prob))] + [float(non_prob[i][0]) for i in range(len(non_prob))]

auc_res18 = roc_auc_score(y_test_res18, y_pred_res18)
ns_fpr_res18, ns_tpr_res18, _ = roc_curve(y_test_res18, y_pred_res18)

plt.figure(figsize=(8, 6))
plt.plot(ns_fpr_res18, ns_tpr_res18, color='darkgreen', linewidth=2, label='ResNet18, AUC= %.3f' % auc_res18)
plt.ylim([0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.legend(loc='lower right')
plt.savefig('/content/drive/My Drive/ColabNotebooks/DeepCovid-master/ResNet18/Results/ROC_covid19_100.png', bbox_inches='tight')
plt.show()
plt.close()

end_time = time.time()
tot_time = end_time - start_time
print("\nTotal Time:", tot_time)
