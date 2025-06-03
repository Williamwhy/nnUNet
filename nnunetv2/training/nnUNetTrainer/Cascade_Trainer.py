import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import os


class CascadedTrainer(nnUNetTrainer):
    def __init__(self, *args, pretrained_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pretrained_path = pretrained_path

    def initialize_network(self):
        super().initialize_network()
        if self.pretrained_path is not None:
            self.load_pretrained_weights(self.pretrained_path)

    def load_pretrained_weights(self, pretrained_path):
        if not os.path.isfile(pretrained_path):
            print(f"❌ Pretrained checkpoint not found at: {pretrained_path}")
            return

        print(f"✅ Loading pretrained weights from: {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        pretrained_weights = checkpoint.get('network_weights', checkpoint)

        current_weights = self.network.state_dict()
        matched_weights = {}

        for k, v in pretrained_weights.items():
            if k in current_weights and v.shape == current_weights[k].shape:
                matched_weights[k] = v
            else:
                print(f"Skipping layer: {k} (shape mismatch or not found)")

        current_weights.update(matched_weights)
        self.network.load_state_dict(current_weights)
        print(f"🔁 Loaded {len(matched_weights)} layers from pretrained model")