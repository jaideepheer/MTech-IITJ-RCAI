from typing import Dict, List, Tuple
from ignite.engine.engine import Engine
from ignite.engine.events import EventEnum
import torch.nn as nn
import torch

from src.utils.utils import get_module_device


class SR3DiffusionEngine(Engine):
    def __init__(
        self,
        *_,
        device: str = "cpu",
        denoise_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: nn.Module = nn.MSELoss(reduction="sum"),
        conditional: bool = True,
        independent_train_t_sample: bool = True,
        beta_variance_schedule: List[float],
    ):
        """
        Parameters
        ---
        denoise_model: nn.Module
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
        """
        self.denoise_model = denoise_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.conditional = conditional
        self.independent_train_t_sample = independent_train_t_sample
        self.set_variance_schedule(torch.FloatTensor(beta_variance_schedule))
        super().__init__(self.train_step)

    @torch.no_grad()
    def set_variance_schedule(self, schedule: torch.FloatTensor):
        device = get_module_device(self.denoise_model)
        schedule = schedule.to(device=device)
        try:
            self.denoise_model.beta = schedule
            self.denoise_model.alpha = 1.0 - self.denoise_model.beta
            self.denoise_model.gamma = torch.cumprod(self.denoise_model.alpha, dim=0)
            self.denoise_model.gamma_prev = torch.cat(
                [
                    torch.ones((1,), device=device),
                    self.denoise_model.gamma,
                ]
            )
        except Exception:
            # register buffers
            self.denoise_model.register_buffer("beta", schedule)
            # add beta derived buffers
            self.denoise_model.register_buffer("alpha", 1.0 - self.denoise_model.beta)
            self.denoise_model.register_buffer(
                "gamma", torch.cumprod(self.denoise_model.alpha, dim=0)
            )
            # gamma right shifted by appending 1.0 to the left
            self.denoise_model.register_buffer(
                "gamma_prev", torch.cat([torch.ones((1,)), self.denoise_model.gamma])
            )

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

    @torch.no_grad()
    def p_sample(self, t: int, y: torch.Tensor, condition_x: torch.Tensor):
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
        alpha, beta, gamma = (
            self.denoise_model.alpha[t],
            self.denoise_model.beta[t],
            self.denoise_model.gamma[t],
        )
        # apply denoise model to predict noise
        y_pred = self.predict_noise(noisy=y, t=t, condition=condition_x)
        # calc. denoise mean
        # See: Eq. 11 in https://arxiv.org/abs/2104.07636v2
        mean = (y - (beta / (1.0 - gamma).sqrt()) * y_pred) / alpha.sqrt()
        # eps for last inference step
        if t == 0:
            return mean
        # sample normal noise
        eps = torch.randn_like(mean)
        # calc. y_{t-1}
        y_prev = mean + beta.sqrt() * eps
        return y_prev

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

    def train_step(self, engine: Engine, batch: Dict[str, torch.Tensor]):
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
        y_noise_pred = self.predict_noise(
            noisy=y_noisy, gamma_noise_level=gamma_samples, condition=x
        )
        # calc loss and back step
        loss = self.loss_fn(eps_noise, y_noise_pred)
        loss.backward()
        self.optimizer.step()
        # log
        return {
            "loss": loss.item(),
            "y_original": y.detach(),
            "y_noisy": y_noisy.detach(),
            "gamma_noise_level": gamma_samples.detach(),
            "pred_noise": y_noise_pred.detach(),
        }

    # def __progbar_generator(self, iterable, description: str, total: int):
    #     if self.trainer is not None:
    #         bar = self.trainer.progress_bar_callback
    #         if isinstance(bar, RichProgressBar) and bar.progress is not None:
    #             prog = bar.progress
    #             task = prog.add_task(description, total=total)

    #             def _gen():
    #                 for i in iterable:
    #                     yield i
    #                     prog.update(task, advance=1)
    #                 prog.remove_task(task)

    #             return _gen()
    #     return track(iterable, description=description, total=total)

    # def forward(self, x: torch.Tensor, t: Iterable[int] = (0,), progbar: bool = True):
    #     """
    #     Performs iterative generation using the trained denoiser.

    #     Parameters
    #     ---
    #     x: torch.Tensor
    #         Conditional tensor for conditional generation.
    #     t: Iterable[int] = (0,)
    #         A list of int specifying the time-steps to include in the returned tensor.
    #         t=0 is the the reconstruction of original image y_0.
    #         If len(t) == 1, then the returned tensor has shape (b, ...)
    #     progbar: bool = False
    #         If True, displays a progress bar for steps completed.

    #     Returns
    #     ---
    #     torch.Tensor: A tensor of shape (len(t), b, ...) or (b, ...) containing iteratively denoised data.
    #     """
    #     ret = []
    #     # start with random image
    #     img = torch.randn_like(x)
    #     # iteratively refine image
    #     steps = list(reversed(range(0, self.T)))
    #     if progbar is True:
    #         steps = self.__progbar_generator(
    #             steps,
    #             description=f"Forward diffusion for tensor {tuple(x.shape)}",
    #             total=self.T,
    #         )
    #     for i in steps:
    #         img = self.p_sample(t=i, y=img, condition_x=x)
    #         if i in t:
    #             ret.append(img.detach())
    #     # return refined result
    #     if len(ret) > 1:
    #         return torch.stack(ret)
    #     else:
    #         return ret[0]

    # def init_params(self, x: torch.Tensor):
    #     return self.p_sample(t=self.T - 1, y=torch.randn_like(x), condition_x=x)
