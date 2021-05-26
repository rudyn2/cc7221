import argparse
import torch
from datasets import OrandCarTestDataset
from models import create_model
from torch.utils.data import DataLoader
from tqdm import tqdm


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Test model utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', default='../data/orand-car-with-bbs', type=str, help='Path to dataset folder')
    parser.add_argument('--weights', default='../weights/best_RetinaNet.pth', type=str, help='Path to model weights')

    # model parameters
    parser.add_argument('--backbone', default='resnet50', type=str, help='Type of backbone')

    # runtime parameters
    parser.add_argument('--device', default='cpu', type=str, help='Type of device used for evaluation [cpu, cuda]')
    parser.add_argument('--batch-size', default=8, type=int, help='Batch size used for evaluation')

    # post-processing parameters
    parser.add_argument('--score-thresh', default=0.5, type=float,
                        help="Score threshold used for postprocessing the detections.")
    parser.add_argument('--nms-thresh', default=0.5, type=float,
                        help="NMS threshold used for postprocessing the detections.")
    parser.add_argument('--detections-per-img', default=12, type=int,
                        help="Number of best detections to keep after NMS.")

    args = parser.parse_args()

    print("[*] Initializing model")
    device = args.device
    model = create_model(args.backbone,
                         score_thresh=args.score_thresh,
                         nms_thresh=args.nms_thresh,
                         detections_per_img=args.detections_per_img)
    model.to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(args.device)
    model.eval()
    print("[+] Model ready!")

    print("[*] Initializing dataset")
    test_dataset = OrandCarTestDataset(args.data)
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=0)

    torch.cuda.empty_cache()
    total, tp = 0, 0

    for images, labels in tqdm(test_loader, "Evaluating..."):

        # forward image
        images = images.to(device)
        model_pred = model(images)

        # process detections
        for real_label, detections in zip(labels, model_pred):

            boxes = detections['boxes']
            if len(boxes) > 0:

                # sort detections
                sorted_idxs = torch.argsort(boxes[:, 0], dim=0)  # sort by x coord
                sorted_boxes = boxes[sorted_idxs]
                sorted_labels = detections['labels'][sorted_idxs]
                sorted_scores = detections['scores'][sorted_idxs]

                # get predicted number
                pred_labels = [str(s) for s in list(sorted_labels.detach().cpu().numpy())]
                pred_number = "".join(pred_labels)
                tp += int(pred_number == real_label)
                total += 1

    print(f"Total: {total}, TP: {tp}, Acc: {100 * (tp / total):.2f}%")


