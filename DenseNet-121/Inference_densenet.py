import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import os, glob
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.models.densenet import DenseNet

parser = argparse.ArgumentParser(description='COVID-19 Detection using DenseNet-121')
parser.add_argument('--test_covid_path', type=str, default='/content/drive/MyDrive/ColabNotebooks/data/val/covid/', help='COVID-19 test samples directory')
parser.add_argument('--test_non_covid_path', type=str, default='/content/drive/MyDrive/ColabNotebooks/data/val/non/', help='Non-COVID test samples directory')
parser.add_argument('--trained_model_path', type=str, default='/content/drive/MyDrive/ColabNotebooks/covid_denseNet_epoch10.pt', help='Trained model path')
parser.add_argument('--cut_off_threshold', type=float, default=0.2, help='Probability threshold for classification')
args = parser.parse_args()

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', pretrained=False)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)

# Allow DenseNet to be unpickled safely
torch.serialization.add_safe_globals([DenseNet])

# Load model weights
checkpoint = torch.load(args.trained_model_path, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    model.load_state_dict(checkpoint["state_dict"])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

#Image Transformations
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = transform(image).float()
    image = image.unsqueeze(0).to(device)
    return image

sm = nn.Softmax(dim=1)

test_covid = glob.glob(os.path.join(args.test_covid_path, '*'))
test_non = glob.glob(os.path.join(args.test_non_covid_path, '*'))

covid_prob, non_prob = [], []
for img_path in test_covid:
    img = image_loader(img_path)
    output = model(img)
    prob = sm(output).cpu().data.numpy()[0, 0]
    covid_prob.append(prob)
    print(f'COVID Image: {img_path}, Probability: {prob:.3f}')

for img_path in test_non:
    img = image_loader(img_path)
    output = model(img)
    prob = sm(output).cpu().data.numpy()[0, 0]
    non_prob.append(prob)
    print(f'Non-COVID Image: {img_path}, Probability: {prob:.3f}')

# Sensitivity and Specificity Calculation
def find_sens_spec(covid_prob, non_prob, thresh):
    sensitivity = sum(p >= thresh for p in covid_prob) / len(covid_prob)
    specificity = sum(p < thresh for p in non_prob) / len(non_prob)
    print(f"Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
    return sensitivity, specificity

find_sens_spec(covid_prob, non_prob, args.cut_off_threshold)

# Confusion Matrix
y_pred = [1 if p > args.cut_off_threshold else 0 for p in covid_prob] + [0 if p > args.cut_off_threshold else 1 for p in non_prob]
y_true = [1] * len(covid_prob) + [0] * len(non_prob)
cnf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cnf_matrix, index=['COVID', 'Non-COVID'], columns=['COVID', 'Non-COVID'])
sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='g')
plt.title('Confusion Matrix')
plt.savefig('/content/drive/MyDrive/ColabNotebooks/confusion_matrix.png')

# Probability Histogram
plt.figure(figsize=(10,6))
bins = np.linspace(0, 1, 25)
plt.subplot(211)
plt.hist(covid_prob, bins, color='red', label='COVID-19 Probabilities')
plt.legend()
plt.subplot(212)
plt.hist(non_prob, bins, color='green', label='Non-COVID Probabilities')
plt.legend()
plt.savefig('/content/drive/MyDrive/ColabNotebooks/probability_histogram_dn.png')

# ROC Curve and AUC
y_scores = covid_prob + non_prob
auc = roc_auc_score(y_true, y_scores)
fpr, tpr, _ = roc_curve(y_true, y_scores)
plt.figure()
plt.plot(fpr, tpr, color='darkgreen', linewidth=2, label=f'DenseNet-121 AUC= {auc:.3f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('/content/drive/MyDrive/ColabNotebooks/ROC_covid19.png')

print("Inference Complete.")
