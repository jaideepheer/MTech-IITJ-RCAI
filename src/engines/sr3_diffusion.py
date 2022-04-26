from typing import Dict, List
from ignite.engine.engine import Engine
from ignite.engine.events import Events
import numpy as np
import torch.nn as nn
import torch
from src.datamodules.utils import de_normalize_image
from rich.progress import track
from src.engines.interface import EngineInterface
from src.torchmodules.mixins import MeasurableModuleMixin

from src.utils.utils import get_module_device


class SR3DiffusionTrainingEngine(EngineInterface):
    def __init__(
        self,
        *_,
        denoise_model: MeasurableModuleMixin,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module = nn.MSELoss(reduction="sum"),
        conditional: bool = True,
        independent_train_t_sample: bool = True,
        beta_variance_schedule: List[float],
        # weights of metrics to add to the loss value
        metric_loss_weights: Dict[str, float] = {
            "flops": 1e-15,
            "latency": 1e-3,
        },
        **kwargs,
    ):
        """
        Parameters
        ---
        denoise_model: MeasurableModuleMixin
            The learnable child module used for denoising.
            This module must take two arguments as input,
                y_t: torch.Tensor
                    The noise tensor to denoise.
                    Shape: (b, ...)
                gamma: torch.Tensor
                    A tensor representing the noise level at the current time step t.
                    Shape: (b,)
            It should return the noise in y_t to be removed from tensor y_{t-1}.
        optimizer: torch.optim.Optimizer
            The optimizer used to train the model.
        loss_fn: nn.Module = nn.MSELoss()
            The loss function to use for denoise_model output and epsilon_noise.
            This loss function must be a p-norm function.
        conditional: bool = True
            If True, provides the input image to the denoise_model during training.
        independent_train_t_sample: bool = True
            If True, the no. of steps t ~ U(T) is different for each batch element.
        beta_variance_schedule: torch.Tensor
            A 1-D tensor of variance beta values, beta_1 to beta_T.
        metric_loss_weights: Dict[str, float]
            A dict specifying which metric to add to loss after how much scaling.
        """
        self.metric_loss_weights = metric_loss_weights or {}
        self.denoise_model = denoise_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.conditional = conditional
        self.independent_train_t_sample = independent_train_t_sample
        self.beta_variance_schedule = beta_variance_schedule
        super().__init__(self.train_step, **kwargs)
        self.add_event_handler(Events.EPOCH_STARTED, self.set_variance_schedule)

    @torch.no_grad()
    def set_variance_schedule(self):
        device = get_module_device(self.denoise_model)
        schedule = torch.FloatTensor(self.beta_variance_schedule).to(device=device)
        # calc. values
        v = {}
        v["beta"] = schedule
        v["alpha"] = 1.0 - v["beta"]
        v["gamma"] = torch.cumprod(v["alpha"], dim=0)
        v["gamma_prev"] = torch.cat([torch.ones((1,), device=device), v["gamma"]])[:-1]
        v["root_one_minus_gamma"] = (1.0 - v["gamma"]).sqrt()
        v["inv_root_gamma"] = (1.0 / v["gamma"]).sqrt()
        v["q_posterior_mean_coef_1"] = (v["gamma_prev"].sqrt() * v["beta"]) / (
            1.0 - v["gamma"]
        )
        v["q_posterior_mean_coef_2"] = (v["alpha"].sqrt() * (1.0 - v["gamma_prev"])) / (
            1.0 - v["gamma"]
        )
        # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        # See: https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/blob/b03c8eff4140e29e30bc4b8ab1fa75dfba630bc6/model/sr3_modules/diffusion.py#L133
        v["q_posterior_stddev_coef"] = torch.clamp(
            (1.0 - v["gamma_prev"]) * v["beta"] / (1.0 - v["gamma"]),
            max=1e-20,
        ).sqrt()
        try:
            # register buffers
            for k, v in v.items():
                self.denoise_model.register_buffer(k, v)
        except Exception:
            for k, v in v.items():
                setattr(self.denoise_model, k, v)

    @property
    def T(self) -> int:
        """
        Returns
        -------
        int
            The number of timesteps T according to the noise schedule beta.
        """
        return int(self.denoise_model.beta.shape[0])

    def predict_noise(
        self,
        *_,
        noisy: torch.Tensor,
        t: int = None,
        gamma_noise_level: torch.Tensor = None,
        condition: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Predict the noise in input noisy image y_t using the given conditional tensor and time step t or gamma noise level.

        Parameters
        ----------
        noise : torch.Tensor
            The noisy tensor to denoise.
        t : int, optional
            The time-step for the noisy tensor for predicting noise in y_{t-1}, if None then gamma_noise_level is used instead.
            By default None.
        gamma_noise_level : torch.Tensor, optional
            Noise level hint gamma passed to the model, if None then t is used instead.
            By default None.
        condition : torch.Tensor, optional
            The conditioning tensor. If provided this tensor is concatenated with the noise tensor at `dim=-3`.
            By default None

        Returns
        -------
        torch.Tensor
            The noise tensor f(x, y_t, gamma) predicted by the denoise_model.

        Raises
        ------
        Exception
            If none or both of `t` or `gamma_noise_level` are provided.
            Exactly one of the args is allowed.
        """
        if not ((gamma_noise_level is None) ^ (t is None)):
            raise Exception("Only one of gamma or t is allowed.")
        if gamma_noise_level is None:
            # use gamma for time step t for all batch elements
            gamma_noise_level = (
                self.denoise_model.gamma[t].expand(noisy.shape[0]).unsqueeze(1)
            )
        if self.conditional is True and condition is not None:
            # concat along channel dim.
            noisy = torch.cat([condition, noisy], dim=-3)
        rt = self.denoise_model(noisy, gamma_noise_level)
        return rt

    def sample_gamma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Samples gamma for each element given the no. of steps t for each element.
        Returns gamma ~ p(gamma) = U(gamma_{t-1}, gamma_t), for each sample according to its given t.
        See: Section 2.4 Para 2 of https://arxiv.org/abs/2104.07636v2

        Parameters
        ---
        t: torch.Tensor
            The tensor having shape (b,) or scalar tensor containing time-steps to sample for each element in the batch.

        Returns
        ---
        gamma: torch.Tensor
            gamma ~ p(gamma) = U(gamma_{t-1}, gamma_t), for each sample according to its give t.
            Shape: (b, 1)
        """
        # select gamma_t and gamma_{t-1}
        # gamma_t shape: (b,) or (1,)
        gamma_t = self.denoise_model.gamma.gather(0, t)
        gamma_t_prev = self.denoise_model.gamma_prev.gather(0, t)
        # take uniform dist. samples for each element
        uniform_samples = torch.rand_like(gamma_t)
        uniform_samples = gamma_t_prev + (gamma_t - gamma_t_prev) * uniform_samples
        return uniform_samples.unsqueeze(1)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        eps_noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Given the source image at t=0, this returns the noisy image after t steps of noise adding.
        See: Eq. 5 in https://arxiv.org/abs/2104.07636v2

        Parameters
        ---
        x0: torch.Tensor
            The initial input image.
        t: torch.Tensor
            The tensor having shape (b,) or (1,) containing time-steps to sample for each element in the batch.
        eps_noise: torch.Tensor
            The epsilon noise which we want our denoiser to predict during training. Shape must match `x0`.

        Returns
        ---
        torch.Tensor: Noisy image after `t` iterations of noise adding.
        torch.Tensor: The tensor of gamma samples.
        """
        gamma_samples = self.sample_gamma(t)
        if self.independent_train_t_sample is False:
            gamma_samples = gamma_samples.expand((x0.shape[0], -1))
        # fix gamma samples shape
        batch = x0.shape[0]
        view = [batch] + [1] * (len(x0.shape) - 1)
        orig_shape = gamma_samples.shape
        gamma_samples = gamma_samples.view(*view)
        # Eq. 5
        noisy = (gamma_samples.sqrt() * x0) + (eps_noise * (1.0 - gamma_samples).sqrt())
        return noisy, gamma_samples.view(orig_shape)

    def train_step(self, engine: Engine, batch: Dict[str, np.ndarray]):
        # prep data and opti
        self.denoise_model.train()
        device = get_module_device(self.denoise_model)
        self.optimizer.zero_grad(set_to_none=True)
        x, y = [torch.as_tensor(i, device=device) for i in [batch["lr"], batch["hr"]]]
        # random t steps, up to T
        t = torch.randint(
            low=0,
            high=self.T,
            size=((x.shape[0],) if self.independent_train_t_sample is True else (1,)),
            device=x.device,
        )
        # sample epsilon noise
        eps_noise = torch.randn_like(y)
        # generate noisy target
        y_noisy, gamma_samples = self.q_sample(y, t, eps_noise)
        # ask model to predict added noise
        noise_pred = self.predict_noise(
            noisy=y_noisy, gamma_noise_level=gamma_samples, condition=x
        )
        # calc loss and back step
        loss: torch.Tensor = self.loss_fn(eps_noise, noise_pred)
        # add metric values to loss
        rt = {}
        if len(self.metric_loss_weights) > 0:
            metric_losses = {
                k: torch.zeros((1), device=device)
                for k in self.metric_loss_weights.keys()
            }
            metric_losses = self.denoise_model.forward_measurements(metric_losses)
            # store data in return variable for logging
            rt = {f"loss/{k}": v.item() for k, v in metric_losses.items()}
            # rescale metric measurements
            metric_losses = {
                k: (self.metric_loss_weights[k] * v) for k, v in metric_losses.items()
            }
            rt |= {f"loss/scaled_{k}": v.item() for k, v in metric_losses.items()}
            # add everything to loss
            for v in metric_losses.values():
                loss = loss + v
        # backward and step
        loss.backward()
        self.optimizer.step()
        # log
        return {
            **rt,
            "loss": loss.item(),
            "x_original": de_normalize_image(x).detach(),
            "y_original": de_normalize_image(y).detach(),
            "y_noisy": de_normalize_image(y_noisy).detach(),
            "gamma_noise_level": gamma_samples.detach(),
        }


class SR3DiffusionInferenceEngine(SR3DiffusionTrainingEngine):
    def __init__(
        self,
        *_,
        denoise_model: nn.Module,
        conditional: bool = True,
        beta_variance_schedule: List[float],
        steps_to_record: List[int] = [],
        clip_after_denoise: bool = True,
        **kwargs,
    ):
        self.denoise_model = denoise_model
        self.conditional = conditional
        self.beta_variance_schedule = beta_variance_schedule
        self.steps_to_record = steps_to_record
        self.clamp_after_denoise = clip_after_denoise
        EngineInterface.__init__(self, self.infer, **kwargs)
        self.add_event_handler(Events.EPOCH_STARTED, self.set_variance_schedule)

    @torch.no_grad()
    def infer(self, engine: Engine, batch: Dict[str, torch.Tensor]):
        """
        Performs iterative generation using the trained denoiser.

        Parameters
        ---
        x: batch: Dict[str, torch.Tensor]
            Batch containing a key 'lr' for the low resolution tensor.
        t: Iterable[int] = (0,)
            A list of int specifying the time-steps to include in the returned tensor.
            t=0 is the the reconstruction of original image y_0.
            If len(t) == 1, then the returned tensor has shape (b, ...)

        Returns
        ---
        Dict[int, torch.Tensor]: A dict with int keys for each timestep and tensor values for the denoised image at that timestep.
        """
        self.denoise_model.eval()
        device = get_module_device(self.denoise_model)
        x = torch.as_tensor(batch["lr"], device=device)
        ret = {"lr": de_normalize_image(x).detach()}
        if "hr" in batch:
            ret["hr"] = de_normalize_image(
                torch.as_tensor(batch["hr"], device=device)
            ).detach()
        # start with random image
        image = torch.randn_like(x)
        # iteratively refine image
        steps = list(reversed(range(0, self.T)))
        for i in track(steps, f"[{self.T}] Batch inference...", transient=True):
            image = self.p_sample(t=i, y_noisy=image, condition_x=x)
            if i in self.steps_to_record:
                # save image
                ret[i] = de_normalize_image(image).detach()
        return ret

    @torch.no_grad()
    def p_sample(self, t: int, y_noisy: torch.Tensor, condition_x: torch.Tensor):
        """
        Given y_t and conditional tensor x, this returns p(y_{t-1} | y_t, x).

        Parameters
        ---
        t: int
            The current time-step for all elements in this batch.
        y: torch.Tensor
            The noisy tensor to denoise. Shape must be batched (b, ...).
        condition_x: torch.Tensor
            The conditioning tensor `x`.
        """
        # apply denoise model to predict noise
        noise_pred = self.predict_noise(noisy=y_noisy, t=t, condition=condition_x)
        # clamp predicted noise
        if self.clamp_after_denoise:
            noise_pred = torch.clamp(noise_pred, -1.0, 1.0)
        # calc. denoise mean
        # See: Eq. 11 in https://arxiv.org/abs/2104.07636v2
        mean = (
            y_noisy - self.denoise_model.root_one_minus_gamma[t] * noise_pred
        ) / self.denoise_model.inv_root_gamma[t]
        # no Z_noise for last inference step
        if t == 0:
            return mean
        # sample normal noise
        Z_noise = torch.randn_like(mean)
        # calc variance
        stddev = self.denoise_model.q_posterior_stddev_coef[t]
        # calc. y_{t-1}
        return mean + stddev * Z_noise


class SubnetSR3DiffusionInferenceEngine(SR3DiffusionInferenceEngine):
    def infer(self, engine: Engine, batch: Dict[str, torch.Tensor]):
        with self.denoise_model.subnet_masked():
            return super().infer(engine, batch)