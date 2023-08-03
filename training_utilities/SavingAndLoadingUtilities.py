import os
import torch


def saveModelToPath(model, path):
    os.makedirs(path, exist_ok=True)
    torch.save(model.state_dict(), f"{path}\\model.pt")


def loadModelFromPath(modelClass,args, PATH):
    model = modelClass(*args )
    model.load_state_dict(torch.load(PATH))
    return model