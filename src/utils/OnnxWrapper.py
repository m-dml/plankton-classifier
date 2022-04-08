import onnxruntime as ort
import torch
import torch.nn.functional as F


class OnnxWrapper:
    def __init__(self, onnx_file, temperature_file=None, training_distribution_file=None, device="cuda"):
        self.device = device
        self.onnx_file = onnx_file
        self.temperature_file = temperature_file
        self.training_distribution_file = training_distribution_file

        self.ort_sess = None
        self.training_class_counts = None
        self.temperatures = None

        if self.training_distribution_file:
            self.training_class_counts = torch.load(self.training_distribution_file)

        if self.temperature_file:
            self.temperatures = torch.load(self.temperature_file)

        self.init_ort()

    def init_ort(self):
        self.ort_sess = ort.InferenceSession(self.onnx_file)

    def __call__(
        self,
        input_image: torch.Tensor,
        correct_probabilities_with_training_prior=False,
        correct_probabilties_with_temperature=False,
        return_probabilities=True,
    ):

        outputs = self.ort_sess.run(None, {"input": input_image.cpu().numpy()})
        outputs = torch.tensor(outputs).to(self.device)

        if correct_probabilties_with_temperature:
            if self.temperatures is None:
                raise ValueError("Temperature File has not been provided")
            outputs = self._correct_probabilities_with_temperature(outputs)

        if correct_probabilities_with_training_prior:
            if self.training_class_counts is None:
                raise ValueError("training_distribution_file has not been provided")
            outputs = self._correct_probabilities_with_training_prior(outputs)

        return outputs

    def _correct_probabilities_with_training_prior(self, logits):
        probabilities = F.softmax(logits)

        p_balanced_per_class = 1 / (len(self.training_class_counts))
        p_corrected_per_class = self.training_class_counts / torch.sum(self.training_class_counts)

        corrected_enumerator = (p_corrected_per_class / p_balanced_per_class) * probabilities
        corrected_denominator = corrected_enumerator + (
            ((1 - p_corrected_per_class) / (1 - p_balanced_per_class)) * (1 - probabilities)
        )
        return F.softmax(corrected_enumerator / corrected_denominator)

    def _correct_probabilities_with_temperature(self, logits):
        temperature = self.temperatures.unsqueeze(1).expand(logits.size(0), logits.size(1)).to(self.device)
        return logits / temperature
