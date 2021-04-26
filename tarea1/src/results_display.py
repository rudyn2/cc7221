import torch
from resnext50 import resnext50
from dataset import TestImageDataset
from torch.utils.data import DataLoader


#def calcula_pred(red, dataset, batch_size=100):
    # base = []
#    predicciones = []
#    dataloader = DataLoader(dataset, batch_size)
#    for x, lab in dataloader:
#        out = red.forward(x)
#        _, max_idx = torch.max(out, dim=1)
#        predicciones.append(max_idx)
#    return predicciones


#net = resnext50(img_channel=3, num_classes=19)
#net.load_state_dict(torch.load(
#    r'C:\Users\aleja\Desktop\Tareas\Reconocimiento Virtual con Deep Learning\Tarea1\Trained\best_ResNext_11.pth'))

#test_dataset = TestImageDataset(
#    r'C:\Users\aleja\Desktop\Tareas\Reconocimiento Virtual con Deep Learning\Tarea1\Imagenes\clothing-small', 224, 224)
#true_labels = list(range(19))
#xtick_labels = list(test_dataset.read_mapping().values())


#pr = calcula_pred(red=net, dataset=test_dataset, batch_size=32)

#predictions = [j for i in pr for j in i]
