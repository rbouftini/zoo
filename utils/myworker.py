import torch

class MyWorker:
    def perturb_params(self, mu, seed):
        with torch.no_grad():
            for j, (name, p) in enumerate(self.model_runner.model.named_parameters()):
                g = torch.Generator(device=p.device).manual_seed(seed)
                v = torch.randn(size=p.shape, device=p.device, generator=g, dtype=p.dtype)
                p.add_(v, alpha=mu)