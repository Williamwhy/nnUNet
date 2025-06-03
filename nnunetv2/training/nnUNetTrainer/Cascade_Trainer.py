import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import os

class Cascade_Trainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device, pretrained_weights=None):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.pretrained_weights = pretrained_weights
        print("✅ Cascade_Trainer initialized!")

    def initialize_network(self):
        super().initialize_network()
        if self.pretrained_weights is not None:
            self.load_pretrained_weights(self.pretrained_weights)

    def load_pretrained_weights(self, pretrained_weights):
        if not os.path.isfile(pretrained_weights):
            print(f"❌ Pretrained checkpoint not found at: {pretrained_weights}")
            return

        print(f"✅ Loading pretrained weights from: {pretrained_weights}")
        checkpoint = torch.load(pretrained_weights, map_location=self.device)
        pretrained_weights = checkpoint.get('network_weights', checkpoint)

        current_weights = self.network.state_dict()
        matched_weights = {}

        for k, v in pretrained_weights.items():
            if k in current_weights and v.shape == current_weights[k].shape:
                matched_weights[k] = v
            else:
                print(f"Skipping layer: {k} (shape mismatch or not found)")

        if not matched_weights:
            print("⚠️ No matching layers found. Make sure architectures are compatible.")

        current_weights.update(matched_weights)
        self.network.load_state_dict(current_weights)
        print(f"🔁 Loaded {len(matched_weights)} layers from pretrained model")
