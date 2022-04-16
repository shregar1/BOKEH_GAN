import os
import torch
import warnings
from testing.tester import Tester
from models.resnet_generator import Net

warnings.filterwarnings("ignore",category=UserWarning)

test_dir = "dataset/test"
outputs_dir = "outputs/results/testing"
weights_dir = "weights"
G_filename = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G = Net(in_channels=3, out_channels=3)
if G_filename is not None:
    G_path = os.path.join(weights_dir,G_filename)
    G.load_state_dict(torch.load(G_path))
G.to(device)

test = Tester(G=G, test_dir=test_dir, outputs_dir=outputs_dir, device=device)
test.fit()