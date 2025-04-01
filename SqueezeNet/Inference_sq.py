from __future__ import print_function
import torch, os, time, glob
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sn
import argparse
from torch.autograd import Variable
from itertools import chain

start_time = time.time()

parser = argparse.ArgumentParser(description='COVID-19 Detection from X-ray Images - SqueezeNet')
parser.add_argument('--test_covid_path', type=str, default='/content/drive/MyDrive/ColabNotebooks/data/val/covid/',
                    help='COVID-19 test samples directory')
parser.add_argument('--test_non_covid_path', type=str, default='/content/drive/MyDrive/ColabNotebooks/data/val/non/',
                    help='Non-COVID test samples directory')
parser.add_argument('--trained_model_path', type=str, default='/content/drive/MyDrive/ColabNotebooks/covid_squeezeNet_epoch50.pt', # TODO
                    help='Path to the trained SqueezeNet model')
parser.add_argument('--cut_off_threshold', type=float, default=0.2,
                    help='Probability threshold to classify as COVID-19')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--num_workers', type=int, default=0)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ['covid', 'non']

#Loading the Trained Model
model = torch.load(args.trained_model_path, map_location=device, weights_only=False)

model.eval()

# Image Preprocessing
imsize = 224
loader = transforms.Compose([
    transforms.Resize(imsize),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def image_loader(image_name):
    image = Image.open(image_name).convert("RGB")
    image = loader(image).float()
    image = Variable(image, requires_grad=False)
    image = image.unsqueeze(0)
    return image.to(device)

sm = torch.nn.Softmax(dim=1)

# Loading the Test Data
covid_files = glob.glob(f"{args.test_covid_path}*")

image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']

non_files = list(chain.from_iterable(
    glob.glob(os.path.join(args.test_non_covid_path, '**', ext), recursive=True)
    for ext in image_extensions
))


covid_pred, covid_prob = [], []
non_pred, non_prob = [], []

# Applying Performance Metrics
for i, path in enumerate(covid_files):
    img = image_loader(path)
    out = model(img)
    prob = sm(out)[0, 0].item()
    pred = out.max(1)[1].item()
    covid_prob.append(prob)
    covid_pred.append(pred)
    print(f"{i:03d} COVID predicted label: {class_names[pred]}")

for i, path in enumerate(non_files):
    img = image_loader(path)
    out = model(img)
    prob = sm(out)[0, 0].item()
    pred = out.max(1)[1].item()
    non_prob.append(prob)
    non_pred.append(pred)
    print(f"{i:03d} Non-COVID predicted label: {class_names[pred]}")

# Sensitivity and Specificity
threshold = args.cut_off_threshold
covid_binary = np.array(covid_prob) > threshold
non_binary = np.array(non_prob) <= threshold

sensitivity = np.mean(covid_binary)
specificity = np.mean(non_binary)
print("\nSensitivity = %.3f, Specificity = %.3f" % (sensitivity, specificity))

# Confusion Matrix
y_true = [1]*len(covid_prob) + [0]*len(non_prob)
y_pred = list(covid_binary.astype(int)) + list((~non_binary).astype(int))

cm = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
plt.figure(figsize=(6,4))
sn.heatmap(df_cm, annot=True, fmt='g', cmap='Blues',
           xticklabels=['Non-COVID', 'COVID'], yticklabels=['Non-COVID', 'COVID'])
plt.title('Confusion Matrix')
plt.savefig('/content/drive/MyDrive/ColabNotebooks/confusion_matrix_squeezenet.png')

# Probability Histogram
plt.figure(figsize=(10,6))
bins = np.linspace(0, 1, 25)
plt.subplot(211)
plt.hist(covid_prob, bins, color='red', label='COVID-19 Probabilities')
plt.legend()
plt.subplot(212)
plt.hist(non_prob, bins, color='green', label='Non-COVID Probabilities')
plt.legend()
plt.savefig('/content/drive/MyDrive/ColabNotebooks/probability_histogram_squeezenet.png')

# ROC Curve
all_probs = covid_prob + non_prob
y_true_bin = [1]*len(covid_prob) + [0]*len(non_prob)
auc = roc_auc_score(y_true_bin, all_probs)
fpr, tpr, _ = roc_curve(y_true_bin, all_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"SqueezeNet AUC = {auc:.3f}", color='purple')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SqueezeNet')
plt.legend(loc='lower right')
plt.savefig('/content/drive/MyDrive/ColabNotebooks/roc_curve_squeezenet.png')

print("\nTotal Inference Time: %.2f seconds" % (time.time() - start_time))

