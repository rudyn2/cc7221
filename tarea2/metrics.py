import torch


def simple_accuracy(y_true, y_pred):
    correct_prediction = torch.equal(torch.argmax(y_true, 1), torch.argmax(y_pred, 1))
    acc = torch.mean(correct_prediction, torch.float32)
    return acc


def metric_accuracy_siamese(y_true, cl_a, cl_p, cl_n):
    y_true_a, y_true_p, y_true_n = torch.split(y_true, split_size_or_sections=3, dim=1)
    correct_prediction_a = torch.equal(torch.argmax(torch.squeeze(y_true_a), 1), torch.argmax(torch.squeeze(cl_a), 1))
    correct_prediction_p = torch.equal(torch.argmax(torch.squeeze(y_true_p), 1), torch.argmax(torch.squeeze(cl_p), 1))
    correct_prediction_n = torch.equal(torch.argmax(torch.squeeze(y_true_n), 1), torch.argmax(torch.squeeze(cl_n), 1))
    acc_a = torch.mean(correct_prediction_a).item()
    acc_p = torch.mean(correct_prediction_p).item()
    acc_n = torch.mean(correct_prediction_n).item()
    acc = (acc_a + acc_p + acc_n) / 3.0
    return acc

#def recall(y_true,y_pred) #dudas hacerlo con keras y entradas