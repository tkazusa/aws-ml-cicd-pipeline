# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

import argparse
import json
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import os
from PIL import Image
import tarfile
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import transforms as T
from engine import train_one_epoch, evaluate
import utils


print('torch', torch.__version__)
print('torchvision', torchvision.__version__)

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        target["filename"] = self.masks[idx].split('_')[0]
        return img, target

    def __len__(self):
        return len(self.imgs)
    

def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model



def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def evaluate_model(args):
    # evaluate on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset_test = PennFudanDataset(args.data_dir, get_transform(train=False))

    # split the dataset in evaluation set
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define evaluation data loaders
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.test_batch_size, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)
    
    
    model_path = os.path.join(args.model_dir, "model.tar.gz")
    
    print("Extracting model from path: {}".format(model_path))
    with tarfile.open(model_path) as tar:
        tar.extractall(path=args.model_dir)
        
    with open(os.path.join(args.model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
        
    model.to(device)
    model.eval()
    
    coco_evaluator = evaluate(model, data_loader_test, device=device, output_dir=args.output_dir)
    print(coco_evaluator.coco_eval)
    print('Average Precision', coco_evaluator.coco_eval['bbox'].stats[0])
    
    report_dict = {"average_precision": coco_evaluator.coco_eval['bbox'].stats[0]}
    
    return report_dict

def save_model(model, model_dir):
    print("Saving the model.")
    path = os.path.join(model_dir, 'model.pth')
    # recommended way from http://pytorch.org/docs/master/notes/serialization.html
    torch.save(model.cpu().state_dict(), path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                    help='input batch size for testing (default: 4)')
    parser.add_argument('--model-dir', type=str, default='', metavar='N',
                        help='model file path')
    parser.add_argument('--data-dir', type=str, default='', metavar='N',
                    help='data file path')
    parser.add_argument('--output-dir', type=str, default='', metavar='N',
                    help='output file path')
    parser.add_argument('--mlflow-server', type=str, default='', metavar='N',
                    help='MLFlow seer')
    parser.add_argument('--experiment-name', type=str, default='', metavar='N',
                    help='MLFlow experiment name')

    args = parser.parse_args()
    
    experiment_name = args.experiment_name
    mlflow.set_tracking_uri(args.mlflow_server)
    mlflow.set_experiment(experiment_name)
    
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    report = evaluate_model(args)
    
    # get run_id
    with open(os.path.join(args.model_dir, 'metadata.json'), 'r') as f:
        json_load = json.load(f)
        
    run_id = json_load['run_id']
    report["experiment_name"] = experiment_name
    report["experiment_id"] = experiment_id
    report["run_id"] = run_id
    
    client = MlflowClient()
    client.log_metric(run_id, 'average_precision', report['average_precision'])
    # client.log_artifact(run_id, args.output_dir)
    
    evaluation_output_path = os.path.join(args.output_dir, "evaluation.json")
    print("Saving classification report to {}".format(evaluation_output_path))

    with open(evaluation_output_path, "w") as f:
        f.write(json.dumps(report))
