import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import os


class Cascade_Trainer(nnUNetTrainer):
    def __init__(self, plans, configuration, fold, dataset_json, device, **kwargs):
        print("Cascade_Trainer initialized!")
        super().__init__(plans, configuration, fold, dataset_json, device, **kwargs)

        # Optional: store pretrained weights (use later for loading)
        self.pretrained_weights = kwargs.get("pretrained_weights", None)

    def initialize_network(self):
        super().initialize_network()
        if self.pretrained_weights is not None:
            self.load_pretrained_weights(self.pretrained_weights)

    def load_pretrained_weights(self, pretrained_weights):
        if not os.path.isfile(pretrained_weights):
            print(f"❌ Pretrained checkpoint not found at: {pretrained_weights}")
            return

        print(f"✅ Loading pretrained weights from: {pretrained_weights}")
        checkpoint = torch.load(pretrained_weights, map_location='cpu')
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
