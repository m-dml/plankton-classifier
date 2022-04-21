import onnxruntime as ort
import torch
import torch.nn.functional as F


class EvalWrapper:
    def __init__(self, temperature_file=None, training_distribution_file=None, device="cuda"):
        self.device = device
        self.temperature_file = temperature_file
        self.training_distribution_file = training_distribution_file

        self.training_class_counts = None
        self.temperatures = None

        if self.training_distribution_file:
            self.training_class_counts = torch.load(self.training_distribution_file).to(self.device)

        if self.temperature_file:
            self.temperatures = torch.load(self.temperature_file).to(self.device)

    def __call__(
        self,
        logits: torch.Tensor,
        correct_probabilities_with_training_prior=False,
        correct_probabilities_with_temperature=False,
        return_probabilities=True,
    ):
        is_probability = False
        logits = logits.to(self.device).detach()
        outputs = logits
        if correct_probabilities_with_temperature:
            if self.temperatures is None:
                raise ValueError("Temperature File has not been provided")
            outputs = self._correct_probabilities_with_temperature(logits)

        if correct_probabilities_with_training_prior:
            if self.training_class_counts is None:
                raise ValueError("training_distribution_file has not been provided")
            outputs = self._correct_probabilities_with_training_prior(logits)
            is_probability = True

        if return_probabilities and (not is_probability):
            outputs = F.softmax(logits, dim=1)

        return outputs

    def _correct_probabilities_with_training_prior(self, logits):
        probabilities = F.softmax(logits)

        p_balanced_per_class = 1 / (len(self.training_class_counts))
        p_corrected_per_class = self.training_class_counts / torch.sum(self.training_class_counts)

        corrected_enumerator = (p_corrected_per_class / p_balanced_per_class) * probabilities
        corrected_denominator = corrected_enumerator + (
            ((1 - p_corrected_per_class) / (1 - p_balanced_per_class)) * (1 - probabilities)
        )
        return corrected_enumerator / corrected_denominator

    def _correct_probabilities_with_temperature(self, logits):
        if len(torch.unique(self.temperatures)) == 1:
            return logits / torch.unique(self.temperatures).to(self.device)
        else:
            raise NotImplementedError("Temperatures are not the same for every output!")
