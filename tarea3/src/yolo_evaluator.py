import glob
import re
from pathlib import Path


class YoloEval(object):
    """
    Auxiliary class used to evaluate YOLO object detection results for handwritten digits
    """

    def __init__(self, gt_txt_path: str, pd_folder_path: str):
        """
        :param gt_txt_path: Path to txt containing the expected predictions
                            in the format: <path_image> <expected_number>
        :param pd_folder_path: Path to folder containing all the txt's with predictions.
        """
        self.gt_txt_path = gt_txt_path
        self.pd_folder_path = pd_folder_path

        self.gt = self._load_ground_truth()
        print("")

    def _load_ground_truth(self):
        with open(self.gt_txt_path, "r") as f:
            lines = f.readlines()

        labels = {}
        for line in lines:
            s = re.split(r'\t+', line.rstrip().rstrip('\t'))
            labels[Path(s[0]).stem] = s[1]
        return labels

    @staticmethod
    def read_prediction(file_path: Path) -> int:
        """
        Read a txt containing bounding predictions in the following format:
            <class> <x> ....
        and calculates the predicted number according to the predicted class of each bbox
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        bboxes = []
        for line in lines:
            s = line.split(" ")
            bboxes.append((s[0], s[1]))
        bboxes.sort(key=lambda x: x[1])
        pred_number = int("".join([e[0] for e in bboxes]))
        return pred_number

    def calculate_accuracy(self):
        """
        Read each file from provided prediction's folder and returns the average accuracy w.r.t. the ground truth
        """
        files = glob.glob(self.pd_folder_path + "/*.txt")
        tp, total = 0, 0
        for file in files:
            file_path = Path(file)
            pred_number = self.read_prediction(file_path)
            expected_number = int(self.gt[file_path.stem])
            if pred_number == expected_number:
                tp += 1
            total += 1
        print(f"Accuracy: {100*(tp/total):.2f}%")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Results evaluation utility",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gt', default='../data/orand-car-with-bbs/test/list.txt',
                        type=str, help='Path to txt with ground truth annotations')
    parser.add_argument('--pd', default='../data/exp/labels')
    args = parser.parse_args()

    e = YoloEval(args.gt, args.pd)
    e.calculate_accuracy()