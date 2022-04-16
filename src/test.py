import os
import torch
from src.models.resnet_generator import Net
from src.testing.tester import Tester


test_dir = "test"
outputs_dir = "outputs/results/testing"
weights_dir = "weights"
G_filename = "Generator_model_dped.pth"
encoder_path = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
G_path = os.path.join(weights_dir,G_filename)
G = Net(in_channels=3, out_channels=3, encoder_path=encoder_path)
G.to(device)
G.load_state_dict(torch.load(G_path))

test = Tester(G=G, test_dir=test_dir, outputs_dir=outputs_dir, device=device)
test.fit()