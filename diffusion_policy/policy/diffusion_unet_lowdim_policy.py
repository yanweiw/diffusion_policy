from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy.common.pytorch_util import dict_apply
import time
import numpy as np

micro_prompt = np.array([
    [ 0.2  , -0.027,  0.841,  0.887, -0.03 ,  0.461, -0.011,  1.   ],
    [ 0.202, -0.047,  0.837,  0.888, -0.042,  0.457, -0.019,  1.   ],
    [ 0.206, -0.068,  0.834,  0.889, -0.047,  0.456, -0.024,  1.   ],
    [ 0.213, -0.086,  0.835,  0.883, -0.052,  0.465, -0.029,  1.   ],
    [ 0.223, -0.103,  0.839,  0.874, -0.061,  0.481, -0.034,  1.   ],
    [ 0.235, -0.118,  0.841,  0.864, -0.068,  0.498, -0.037,  1.   ],
    [ 0.251, -0.131,  0.84 ,  0.857, -0.07 ,  0.51 , -0.039,  1.   ],
    [ 0.269, -0.139,  0.835,  0.854, -0.069,  0.515, -0.039,  1.   ],
    [ 0.287, -0.144,  0.828,  0.853, -0.067,  0.516, -0.038,  1.   ],
    [ 0.307, -0.148,  0.82 ,  0.853, -0.065,  0.516, -0.035,  1.   ],
    [ 0.326, -0.15 ,  0.814,  0.852, -0.064,  0.518, -0.032,  1.   ],
    [ 0.348, -0.151,  0.809,  0.85 , -0.063,  0.523, -0.027,  1.   ],
    [ 0.37 , -0.152,  0.805,  0.846, -0.065,  0.529, -0.021,  1.   ],
    [ 0.391, -0.154,  0.802,  0.842, -0.067,  0.534, -0.014,  1.   ],
    [ 0.412, -0.154,  0.797,  0.84 , -0.067,  0.538, -0.008,  1.   ],
    [ 0.432, -0.151,  0.791,  0.838, -0.065,  0.542, -0.002,  1.   ],
    [ 0.451, -0.148,  0.785,  0.837, -0.063,  0.544,  0.002,  1.   ],
    [ 0.47 , -0.145,  0.778,  0.834, -0.061,  0.548,  0.007,  1.   ],
    [ 0.49 , -0.14 ,  0.772,  0.829, -0.057,  0.556,  0.013,  1.   ],
    [ 0.509, -0.133,  0.767,  0.823, -0.05 ,  0.565,  0.018,  1.   ],
    [ 0.527, -0.124,  0.762,  0.817, -0.04 ,  0.575,  0.023,  1.   ],
    [ 0.544, -0.115,  0.756,  0.811, -0.031,  0.584,  0.029,  1.   ],
    [ 0.562, -0.107,  0.748,  0.804, -0.024,  0.594,  0.036,  1.   ],
    [ 0.579, -0.1  ,  0.739,  0.797, -0.014,  0.602,  0.044,  1.   ],
    [ 0.595, -0.092,  0.73 ,  0.789, -0.006,  0.612,  0.051,  1.   ],
    [ 0.611, -0.085,  0.719,  0.782,  0.001,  0.621,  0.056,  1.   ],
    [ 0.628, -0.081,  0.708,  0.777,  0.005,  0.626,  0.061,  1.   ],
    [ 0.628, -0.081,  0.708,  0.777,  0.005,  0.626,  0.061,  1.   ]])
bowl1_prompt = np.array([
    [ 0.198,  0.008,  0.858,  0.851,  0.026,  0.522, -0.046,  1.   ],
    [ 0.202,  0.027,  0.857,  0.854,  0.021,  0.516, -0.067,  1.   ],
    [ 0.208,  0.045,  0.853,  0.858,  0.024,  0.506, -0.08 ,  1.   ],
    [ 0.218,  0.063,  0.848,  0.864,  0.034,  0.495, -0.086,  1.   ],
    [ 0.23 ,  0.078,  0.841,  0.871,  0.046,  0.482, -0.086,  1.   ],
    [ 0.246,  0.091,  0.834,  0.876,  0.055,  0.472, -0.082,  1.   ],
    [ 0.264,  0.102,  0.832,  0.877,  0.06 ,  0.47 , -0.081,  1.   ],
    [ 0.282,  0.112,  0.831,  0.875,  0.065,  0.473, -0.081,  1.   ],
    [ 0.3  ,  0.121,  0.83 ,  0.873,  0.068,  0.476, -0.082,  1.   ],
    [ 0.318,  0.132,  0.826,  0.872,  0.071,  0.478, -0.083,  1.   ],
    [ 0.335,  0.145,  0.823,  0.871,  0.074,  0.479, -0.086,  1.   ],
    [ 0.35 ,  0.159,  0.819,  0.87 ,  0.078,  0.479, -0.088,  1.   ],
    [ 0.366,  0.174,  0.815,  0.87 ,  0.082,  0.477, -0.092,  1.   ],
    [ 0.379,  0.19 ,  0.811,  0.874,  0.085,  0.469, -0.095,  1.   ],
    [ 0.393,  0.206,  0.807,  0.878,  0.085,  0.46 , -0.099,  1.   ],
    [ 0.407,  0.221,  0.804,  0.882,  0.083,  0.452, -0.105,  1.   ],
    [ 0.423,  0.233,  0.8  ,  0.886,  0.082,  0.444, -0.108,  1.   ],
    [ 0.44 ,  0.242,  0.794,  0.89 ,  0.085,  0.436, -0.107,  1.   ],
    [ 0.457,  0.248,  0.784,  0.895,  0.086,  0.424, -0.107,  1.   ],
    [ 0.474,  0.254,  0.771,  0.901,  0.088,  0.411, -0.108,  1.   ],
    [ 0.489,  0.259,  0.758,  0.906,  0.089,  0.398, -0.11 ,  1.   ],
    [ 0.503,  0.264,  0.744,  0.912,  0.091,  0.385, -0.112,  1.   ],
    [ 0.517,  0.269,  0.73 ,  0.917,  0.092,  0.371, -0.115,  1.   ],
    [ 0.532,  0.272,  0.716,  0.921,  0.092,  0.358, -0.119,  1.   ],
    [ 0.546,  0.275,  0.701,  0.926,  0.091,  0.345, -0.123,  1.   ],
    [ 0.56 ,  0.277,  0.687,  0.93 ,  0.09 ,  0.333, -0.127,  1.   ],
    [ 0.576,  0.278,  0.672,  0.934,  0.085,  0.32 , -0.132,  1.   ],
    [ 0.591,  0.279,  0.657,  0.939,  0.075,  0.306, -0.138,  1.   ],
    [ 0.604,  0.278,  0.641,  0.943,  0.071,  0.295, -0.14 ,  1.   ],
    [ 0.615,  0.277,  0.624,  0.946,  0.068,  0.282, -0.141,  1.   ],
    [ 0.626,  0.276,  0.605,  0.951,  0.065,  0.269, -0.141,  1.   ],
    [ 0.636,  0.27 ,  0.589,  0.954,  0.06 ,  0.258, -0.138,  1.   ],
    [ 0.646,  0.257,  0.577,  0.957,  0.049,  0.252, -0.134,  1.   ],
    [ 0.646,  0.257,  0.577,  0.957,  0.049,  0.252, -0.134,  1.   ]])
bowl2_prompt = np.array([
    [ 0.195, -0.004,  0.81 ,  0.877, -0.01 ,  0.476, -0.058,  1.   ],
    [ 0.211,  0.001,  0.798,  0.887, -0.005,  0.459, -0.059,  1.   ],
    [ 0.228,  0.004,  0.787,  0.895, -0.002,  0.442, -0.058,  1.   ],
    [ 0.246,  0.009,  0.776,  0.905,  0.001,  0.422, -0.057,  1.   ],
    [ 0.263,  0.012,  0.766,  0.913,  0.004,  0.404, -0.056,  1.   ],
    [ 0.281,  0.016,  0.756,  0.921,  0.006,  0.386, -0.054,  1.   ],
    [ 0.3  ,  0.019,  0.745,  0.927,  0.007,  0.371, -0.053,  1.   ],
    [ 0.32 ,  0.023,  0.737,  0.931,  0.008,  0.362, -0.051,  1.   ],
    [ 0.339,  0.026,  0.73 ,  0.932,  0.009,  0.358, -0.05 ,  1.   ],
    [ 0.359,  0.03 ,  0.723,  0.933,  0.01 ,  0.356, -0.05 ,  1.   ],
    [ 0.378,  0.035,  0.717,  0.933,  0.012,  0.355, -0.049,  1.   ],
    [ 0.397,  0.04 ,  0.711,  0.933,  0.014,  0.356, -0.048,  1.   ],
    [ 0.415,  0.046,  0.705,  0.933,  0.017,  0.356, -0.047,  1.   ],
    [ 0.433,  0.053,  0.698,  0.934,  0.021,  0.353, -0.044,  1.   ],
    [ 0.451,  0.061,  0.69 ,  0.937,  0.025,  0.347, -0.042,  1.   ],
    [ 0.468,  0.068,  0.681,  0.94 ,  0.029,  0.338, -0.04 ,  1.   ],
    [ 0.484,  0.075,  0.672,  0.943,  0.032,  0.328, -0.038,  1.   ],
    [ 0.502,  0.081,  0.663,  0.947,  0.034,  0.318, -0.037,  1.   ],
    [ 0.522,  0.087,  0.656,  0.95 ,  0.029,  0.308, -0.038,  1.   ],
    [ 0.542,  0.091,  0.65 ,  0.952,  0.02 ,  0.302, -0.04 ,  1.   ],
    [ 0.562,  0.093,  0.646,  0.954,  0.011,  0.297, -0.042,  1.   ],
    [ 0.583,  0.094,  0.643,  0.955,  0.008,  0.295, -0.041,  1.   ],
    [ 0.602,  0.095,  0.64 ,  0.956,  0.006,  0.29 , -0.039,  1.   ],
    [ 0.622,  0.099,  0.632,  0.96 ,  0.007,  0.277, -0.038,  1.   ],
    [ 0.634,  0.1  ,  0.62 ,  0.966,  0.007,  0.257, -0.035,  1.   ],
    [ 0.645,  0.102,  0.603,  0.97 ,  0.007,  0.242, -0.033,  1.   ],
    [ 0.653,  0.104,  0.584,  0.973,  0.008,  0.228, -0.031,  1.   ],
    [ 0.656,  0.104,  0.566,  0.977,  0.008,  0.209, -0.028,  1.   ],
    [ 0.65 ,  0.106,  0.558,  0.981,  0.01 ,  0.19 , -0.025,  1.   ],
    [ 0.65 ,  0.106,  0.558,  0.981,  0.01 ,  0.19 , -0.025,  1.   ]])

class DiffusionUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self, 
            model: ConditionalUnet1D,
            noise_scheduler: DDPMScheduler,
            horizon, 
            obs_dim, 
            action_dim, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            past_action_visible=False,
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=past_action_visible
        )
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.past_action_visible = past_action_visible
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None, guide=None, visualizer=None, guide_visualizer=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
        
        # # use micro prompt as base trajectory and add guassian noise
        # # first make micro prompt the same shape as trajectory
        # prompt = torch.from_numpy(micro_prompt).float().cuda()
        # indices = torch.linspace(0, prompt.shape[0]-1, trajectory.shape[1], dtype=int)
        # prompt = torch.unsqueeze(prompt[indices], dim=0) # (1, pred_horizon, guide_dim)
        # prompt = self.normalizer['action'].normalize(prompt)

        # # add gaussian noise
        # std = 0.00
        # trajectory = std * trajectory + prompt

        if guide is not None:
            # normalize guide
            # guide_original = guide.clone()
            guide = self.normalizer['action'].normalize(guide)

        # if guide is not None and guide_visualizer is not None:
        #     guide_markers = self.normalizer['action'].unnormalize(guide)
        #     visualizer.viz_guide(guide_markers)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        MCMC_steps = 1
        if guide is not None:
            MCMC_steps = 5
        clean_sample = None
        dist = None
        for t in scheduler.timesteps:
            for i in range(MCMC_steps):
                # 1. apply conditioning
                trajectory[condition_mask] = condition_data[condition_mask]

                # 2. predict model output
                model_output = model(trajectory, t, 
                    local_cond=local_cond, global_cond=global_cond)
                
                # 3. add interaction gradient
                if guide is not None and t > 0: # stop adding noise as it will distract the plan
                    grad, dist = self.guide_gradient_by_pixel(trajectory, guide, t)
                    # guide_ratio = 20 * 0.98**(self.num_inference_steps - t)
                    guide_ratio = 300
                    # print('model_output norm and grad norm:', torch.linalg.matrix_norm(model_output).mean(), torch.linalg.matrix_norm(grad).mean())
                    assert model_output.shape == grad.shape                
                    model_output = model_output + guide_ratio * grad

                # 4. compute previous image: x_t -> x_t-1
                scheduler_output = scheduler.step(
                        model_output, t, trajectory, 
                        generator=generator,
                        **kwargs)
                prev_sample = scheduler_output.prev_sample
                clean_sample = scheduler_output.pred_original_sample    

                if i < MCMC_steps - 1:
                    # print('mcmc step i: ', i, 'at t: ', t)
                    std = 1
                    noise = std * torch.randn(clean_sample.shape, device=clean_sample.device)
                    trajectory = self.noise_scheduler.add_noise(clean_sample, noise, t)
                else:
                    # print('final mcmc step at t:', t)
                    trajectory = prev_sample

                # # 5. visualize
                # if visualizer is not None:
                #     action = self.normalizer['action'].unnormalize(clean_sample)
                #     action = action.detach().cpu().numpy()
                #     action_marker = action.reshape(-1, 8)
                #     visualizer.viz_traj(traj_ee=action_marker, scores=None)  
                #     print('timestep:', t)
                #     time.sleep(0.1)

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        
        # time.sleep(1)

        return trajectory, dist

    def guide_gradient_by_pixel(self, naction, guide, t):
        # guide = torch.tensor([0.628, -0.067, 0.694]).cuda().unsqueeze(0)
        # naction: (B, pred_horizon, action_dim);
        # guide: (1, guide_dim)
        # assert naction.shape[2] == 8 and guide.shape == (1, 3) # guide is only 3D point
        # print('guide pixel:', guide)
        # start_to_goal = [((1 - t) * naction[:, 0, :3] + t * guide) for t in torch.linspace(0, 1, naction.shape[1])]
        # start_to_goal = torch.stack(start_to_goal, dim=1) # (B, pred_horizon, 3)
        indices = torch.linspace(0, guide.shape[0]-1, naction.shape[1], dtype=int)
        guide = torch.unsqueeze(guide[indices], dim=0) # (1, pred_horizon, guide_dim)

        assert guide.shape == (1, naction.shape[1], 8)
        # assert guide.shape[1:] == (8,)
        # guide = torch.unsqueeze(guide, dim=0) # (1, guide_horizon, guide_dim)
        # indices = torch.linspace(0, guide.shape[0]-1, naction.shape[1], dtype=int)
        # guide = torch.unsqueeze(guide[indices], dim=0) # (1, guide_horizon, guide_dim)
        with torch.enable_grad():
            naction = naction.clone().detach().requires_grad_(True)
            # dist = torch.linalg.norm(naction[:, :, :3] - guide, dim=2)[:, (naction.shape[1]//2):].mean(dim=1) # (B,)
            # dist = torch.min(torch.linalg.norm(naction[:, :, :3] - start_to_goal, dim=2), dim=1)[0] # (B,)
            dist = torch.linalg.norm(naction[:, :, :3] - guide[:, :, :3], dim=2, ord=2)**2 # (B, pred_horizon)
            dist = dist.mean(dim=1) # (B,)
            # dist_min = torch.min(dist, dim=1)[0] # (B,)
            # print('dist:', dist_mean, dist_min)
            # if t > 50:
                # dist = 0.1*dist_mean
            # else:
                # dist = 0.1*dist_mean + 10*dist_min
            grad = torch.autograd.grad(dist, naction, grad_outputs=torch.ones_like(dist), create_graph=False)[0]
            # naction.detach()
        return grad, dist.clone().detach()

    # def guide_gradient_by_pixel(self, naction, guide, t):
    #     # guide = torch.tensor([0.628, -0.067, 0.694]).cuda().unsqueeze(0)
    #     # naction: (B, pred_horizon, action_dim);
    #     # guide: (1, guide_dim)
    #     # assert naction.shape[2] == 8 and guide.shape == (1, 3) # guide is only 3D point
    #     # print('guide pixel:', guide)
    #     # start_to_goal = [((1 - t) * naction[:, 0, :3] + t * guide) for t in torch.linspace(0, 1, naction.shape[1])]
    #     # start_to_goal = torch.stack(start_to_goal, dim=1) # (B, pred_horizon, 3)
    #     # indices = torch.linspace(0, guide.shape[0]-1, naction.shape[1], dtype=int)
    #     # guide = torch.unsqueeze(guide[indices], dim=0) # (1, pred_horizon, guide_dim)

    #     # assert guide.shape == (1, naction.shape[1], 3)
    #     # guide = torch.unsqueeze(guide, dim=0) # (1, pred_horizon, guide_dim)
    #     assert guide.shape[1:] == (8,)
    #     # guide = torch.unsqueeze(guide, dim=0) # (1, guide_horizon, guide_dim)
    #     indices = torch.linspace(0, guide.shape[0]-1, naction.shape[1], dtype=int)
    #     guide = torch.unsqueeze(guide[indices], dim=0) # (1, guide_horizon, guide_dim)
    #     with torch.enable_grad():
    #         naction = naction.clone().detach().requires_grad_(True)
    #         # dist = torch.linalg.norm(naction[:, :, :3] - guide, dim=2)[:, (naction.shape[1]//2):].mean(dim=1) # (B,)
    #         # dist = torch.min(torch.linalg.norm(naction[:, :, :3] - start_to_goal, dim=2), dim=1)[0] # (B,)
    #         dist = torch.linalg.norm(naction[:, :, :3] - guide[:, :, :3], dim=2, ord=2)**2 # (B, pred_horizon)
    #         dist = dist.mean(dim=1) # (B,)
    #         # dist_min = torch.min(dist, dim=1)[0] # (B,)
    #         # print('dist:', dist_mean, dist_min)
    #         # if t > 50:
    #             # dist = 0.1*dist_mean
    #         # else:
    #             # dist = 0.1*dist_mean + 10*dist_min
    #         grad = torch.autograd.grad(dist, naction, grad_outputs=torch.ones_like(dist), create_graph=False)[0]
    #         # naction.detach()
    #     return grad   


    def predict_action(self, obs_dict: Dict[str, torch.Tensor], guide=None, visualizer=None, guide_visualizer=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict # not implemented yet
        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.action_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B,T,Do), device=device, dtype=dtype)
            local_cond[:,:To] = nobs[:,:To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # for each batch element, add the same gaussian noise across all timesteps
            # std = 0.1
            # noise = std * torch.randn(nobs.shape[0], 1, nobs.shape[2], device=device, dtype=dtype)
            # nobs = nobs + noise

            # condition throught global feature
            global_cond = nobs[:,:To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            if self.past_action_visible:
                cond_data[:,:To-1,:] = nobs[:,1:self.n_obs_steps,:]
                cond_mask[:,:To-1,:] = True            
        else:
            # condition through impainting
            shape = (B, T, Da+Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs[:,:To]
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample, dist = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            guide=guide,
            visualizer=visualizer,
            guide_visualizer=guide_visualizer,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:,start:end]
        
        result = {
            'action': action,
            'action_pred': action_pred
        }
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[...,Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:,start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred
        return result, dist

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:,self.n_obs_steps:,:] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:,:self.n_obs_steps,:].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:,start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask
        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = trajectory[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
