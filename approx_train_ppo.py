# approx_train_ppo.py - PPO-based learning for 5-parameter INT8 approximation
import os
import json
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical
import torch.nn.functional as F

# Re-use components from the original script
from model_vgg16_tap_quant import VGG16TapQuant
from model_vgg19_tap_quant import VGG19TapQuant, get_vgg19_layer_names
from model_alexnet_tap_quant import AlexNetTapQuant, get_alexnet_layer_names
from model_lenet5_tap_quant import LeNet5TapQuant, get_lenet5_layer_names
from utils import get_dataloaders, ensure_dir
from recorder import ActivationRecorder
from torch.ao.quantization import prepare_qat, get_default_qat_qconfig, convert
from approx_train import (_build_teacher_int8, _build_student_fq, teacher_forward_with_scales, 
                          CONV_LAYERS, VGG19_CONV_LAYERS, ALEXNET_CONV_LAYERS, LENET5_LAYERS, NIN_LAYERS, MOBILENETV1_LAYERS,
                          _build_teacher_int8_vgg19, _build_student_fq_vgg19,
                          _build_teacher_int8_alexnet, _build_student_fq_alexnet,
                          _build_teacher_int8_lenet5, _build_student_fq_lenet5,
                          _build_teacher_int8_nin, _build_student_fq_nin,
                          _build_teacher_int8_mobilenetv1, _build_student_fq_mobilenetv1,
                          EditOnlyRecorder, _collect_histograms, _q_nonzero_from_hist)

# Import new modules for state encoding and attention transfer
from state_encoder_simple import SimpleStateEncoder
from attention_transfer import AttentionTransfer, register_feature_hooks

# ==================================
#      PPO Core Components
# ==================================

class RolloutBuffer:
    """Buffer to store trajectories for PPO."""
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.tmax_codes = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.tmax_codes[:]

class ActorCritic(nn.Module):
    """
    PPO Actor-Critic network with 4D state input.
    Learns a policy to select 4 INT8 code parameters for each layer (t1, v1, t2, v2).
    tmax is fixed based on histogram statistics.
    
    State: 4D vector [layer_depth_ratio, act_mean, act_std, p90]
    """
    def __init__(self, state_dim=4, hidden_dim=64, action_dim=256):
        super(ActorCritic, self).__init__()

        # State encoder: 2-layer MLP to process 4D state vector
        # Replaces the original layer_embeddings (Embedding layer)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Actor: output raw logits over INT8 codes [0..255] for each parameter head
        # We have 4 separate heads, one for each parameter (t1, v1, t2, v2)
        self.actor_t1 = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.actor_v1 = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.actor_t2 = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.actor_v2 = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        
        # Critic: evaluates the state
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.action_heads = [self.actor_t1, self.actor_v1, self.actor_t2, self.actor_v2]

    def initialize_from_histograms(self, layer_init_params: Dict[str, Tuple[int, int, int, int]]):
        """
        Optionally initialize/bias embeddings based on histogram-derived init values.
        Note: A robust per-layer bias would require modifying embeddings/weights jointly; for now we keep
        the default initialization to avoid introducing global biases that affect all layers.
        """
        return

    def forward(self, state):
        raise NotImplementedError

    def act(self, state, tmax_code):
        """
        Sample an action (t1, t2, v1, v2) using masked categorical distributions over logits.
        Enforce constraints under fixed tmax_code:
          - t1 in [2, min(10, tmax-8)]
          - t2 in [t1+4, tmax-4]
          - v1 in [t1+1, t2-1]
          - v2 in [t2+1, tmax]
        And ensure overall ordering: 1 <= t1 < v1 < t2 < v2 <= tmax_code
        
        Args:
            state: 4D state vector, shape [1, 4] or [4]
            tmax_code: fixed tmax value (int)
        """
        # Ensure state is 2D: [1, 4]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        state_embedding = self.state_encoder(state)  # shape [1, hidden_dim]

        def masked_sample(head_logits: torch.Tensor, low: int, high: int):
            # head_logits: [1, 256]
            logits = head_logits.clone()
            mask = torch.zeros_like(logits)
            # valid indices [low, high]
            low_c = max(0, int(low))
            high_c = min(int(high), logits.size(-1) - 1)
            if high_c < low_c:
                high_c = low_c
            mask[..., low_c:high_c + 1] = 1.0
            masked_logits = logits + (1.0 - mask) * (-1e9)
            dist = Categorical(logits=masked_logits)
            a = dist.sample()
            lp = dist.log_prob(a)
            return a, lp

        # 1) t1 in [2, min(10, tmax-8)] to ensure room for t2<=tmax-4 and t2>=t1+4
        t1_logits = self.actor_t1(state_embedding)
        t1_low = 2
        t1_high = min(10, max(t1_low, tmax_code - 8))
        t1, lp1 = masked_sample(t1_logits, t1_low, t1_high)

        # 2) t2 in [t1+4, tmax-4]
        t2_logits = self.actor_t2(state_embedding)
        t1_val = int(t1.item())
        t2_low = t1_val + 4
        t2_high = max(t2_low, tmax_code - 4)
        t2, lp2 = masked_sample(t2_logits, t2_low, t2_high)

        # 3) v1 in [t1+1, t2-1]
        v1_logits = self.actor_v1(state_embedding)
        t2_val = int(t2.item())
        v1_low = t1_val + 1
        v1_high = max(v1_low, t2_val - 1)
        v1, lp3 = masked_sample(v1_logits, v1_low, v1_high)

        # 4) v2 in [t2+1, tmax]
        v2_logits = self.actor_v2(state_embedding)
        v2_low = t2_val + 1
        v2_high = tmax_code
        v2, lp4 = masked_sample(v2_logits, v2_low, v2_high)

        # Return in canonical order [t1, v1, t2, v2]
        actions = [t1, v1, t2, v2]
        logps = [lp1, lp3, lp2, lp4]
        return torch.stack(actions), torch.stack(logps).sum()

    def evaluate(self, state, action, tmax_codes):
        """
        Compute log_prob and entropy under masked logits for a batch of (state, action, tmax).
        
        Args:
            state: [B, 4] - batch of 4D state vectors
            action: [B, 4] - batch of actions
            tmax_codes: [B] - batch of tmax values
        """
        # state is now [B, 4], directly encode it
        state_embedding = self.state_encoder(state)  # [B, hidden_dim]

        B = state_embedding.size(0)
        logprobs_sum = []
        entropies_sum = []

        for b in range(B):
            emb = state_embedding[b:b+1]  # keep batch dim
            tmax_code = int(tmax_codes[b].item())

            # t1 in [2, min(10, tmax-8)]
            t1_logits = self.actor_t1(emb)
            t1_low = 2
            t1_high = min(10, max(2, tmax_code - 8))
            mask = torch.zeros_like(t1_logits)
            mask[..., t1_low:t1_high + 1] = 1.0
            dist_t1 = Categorical(logits=t1_logits + (1.0 - mask) * (-1e9))
            a_t1 = action[b, 0]  # index 0 is t1
            lp1 = dist_t1.log_prob(a_t1)
            ent1 = dist_t1.entropy()

            # t2 in [t1+4, tmax-4]
            t2_logits = self.actor_t2(emb)
            t1_val = int(a_t1.item())
            t2_low = t1_val + 4
            t2_high = max(t2_low, tmax_code - 4)
            mask = torch.zeros_like(t2_logits)
            mask[..., t2_low:t2_high + 1] = 1.0
            dist_t2 = Categorical(logits=t2_logits + (1.0 - mask) * (-1e9))
            a_t2 = action[b, 2]  # index 2 is t2
            lp3 = dist_t2.log_prob(a_t2)
            ent3 = dist_t2.entropy()

            # v1 in [t1+1, t2-1]
            v1_logits = self.actor_v1(emb)
            t2_val = int(a_t2.item())
            v1_low = t1_val + 1
            v1_high = max(v1_low, t2_val - 1)
            mask = torch.zeros_like(v1_logits)
            mask[..., v1_low:v1_high + 1] = 1.0
            dist_v1 = Categorical(logits=v1_logits + (1.0 - mask) * (-1e9))
            a_v1 = action[b, 1]  # index 1 is v1
            lp2 = dist_v1.log_prob(a_v1)
            ent2 = dist_v1.entropy()

            # v2 in [t2+1, tmax]
            v2_logits = self.actor_v2(emb)
            v2_low = t2_val + 1
            v2_high = tmax_code
            mask = torch.zeros_like(v2_logits)
            mask[..., v2_low:v2_high + 1] = 1.0
            dist_v2 = Categorical(logits=v2_logits + (1.0 - mask) * (-1e9))
            a_v2 = action[b, 3]  # index 3 is v2
            lp4 = dist_v2.log_prob(a_v2)
            ent4 = dist_v2.entropy()

            logprobs_sum.append((lp1 + lp2 + lp3 + lp4).squeeze(0))
            entropies_sum.append((ent1 + ent2 + ent3 + ent4).squeeze(0))

        logprobs_sum = torch.stack(logprobs_sum, dim=0)
        entropies_sum = torch.stack(entropies_sum, dim=0)
        state_values = self.critic(state_embedding)

        return logprobs_sum, state_values, entropies_sum

# ==================================
#   PPO-Compatible Approx Function
# ==================================

def TriApproxINT8_PPO(x: torch.Tensor, s: float, z: int, codes: torch.Tensor, tmax_code: int):
    """
    A simple, non-nn.Module function to apply the approximation.
    'codes' is a tensor of 4 integer codes [t1, v1, t2, v2] from the PPO agent.
    'tmax_code' is the fixed maximum code for this layer.
    Note: codes already satisfy constraints t1 < v1 < t2 < v2 < tmax
    """
    x = x.clamp_min(0)
    
    # Move codes to the same device as x
    codes = codes.to(x.device)

    # Extract codes (already constrained during sampling)
    t1_code = codes[0]
    v1_code = codes[1]
    t2_code = codes[2]
    v2_code = codes[3]

    # Dequantize codes to float thresholds
    t1   = s * (t1_code   - z)
    t2   = s * (t2_code   - z)
    tmax = s * (tmax_code - z)
    v1   = s * (v1_code   - z)
    v2   = s * (v2_code   - z)

    # Apply hard approximation
    m1 = (x > 0)  & (x <= t1)
    m2 = (x > t1) & (x <= t2)
    m3 = (x > t2) & (x <= tmax)
    y = torch.where(m1, torch.zeros_like(x),
            torch.where(m2, v1,
            torch.where(m3, v2, x)))
    
    return y

# ==================================
#      PPO Training Agent
# ==================================

class PPO:
    """PPO Agent: takes collected trajectories and updates the policy."""
    def __init__(self,
                 policy: ActorCritic,
                 lr: float,
                 betas: Tuple[float, float],
                 gamma: float,
                 K_epochs: int,
                 eps_clip: float,
                 v_loss_coef: float = 0.5,
                 ent_coef: float = 0.01):
        
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.v_loss_coef = v_loss_coef
        self.ent_coef = ent_coef

        self.policy = policy
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.state_encoder.parameters(), 'lr': lr},
            {'params': self.policy.actor_t1.parameters(), 'lr': lr},
            {'params': self.policy.actor_v1.parameters(), 'lr': lr},
            {'params': self.policy.actor_t2.parameters(), 'lr': lr},
            {'params': self.policy.actor_v2.parameters(), 'lr': lr},
            {'params': self.policy.critic.parameters(), 'lr': lr}
        ])
        self.MseLoss = nn.MSELoss()

    def update(self, rollout_buffer: RolloutBuffer):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rollout_buffer.rewards), reversed(rollout_buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.policy.state_encoder[0].weight.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(rollout_buffer.states, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(rollout_buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(rollout_buffer.logprobs, dim=0)).detach()
        tmax_codes_tensor = torch.tensor(rollout_buffer.tmax_codes, dtype=torch.long).to(old_states.device)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions, tmax_codes_tensor)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + self.v_loss_coef * self.MseLoss(state_values, rewards) - self.ent_coef * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Clear the buffer
        rollout_buffer.clear()

# ==================================
#      Main Training Loop
# ==================================

def train_ppo(args):
    """Main PPO training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_dataloaders(args.data, args.batch_size, args.workers)

    # 选择模型架构
    model_type = getattr(args, "model", "vgg16")  # 默认VGG16
    
    if model_type == "vgg19":
        # VGG19流程（16个卷积层 + 1个FC层）
        teacher = _build_teacher_int8_vgg19(args)
        student = _build_student_fq_vgg19(args).to(device)
        CONV_LAYERS_USED = get_vgg19_layer_names()
        attention_layers_default = ['conv3_2', 'conv4_2', 'conv5_2']  # VGG19的关键层
        print(f"[PPO] Using VGG19 with {len(CONV_LAYERS_USED)} layers (16 conv + 1 fc)")
    elif model_type == "alexnet":
        # AlexNet流程（5个卷积层 + 2个FC层 = 7层）
        teacher = _build_teacher_int8_alexnet(args)
        student = _build_student_fq_alexnet(args).to(device)
        CONV_LAYERS_USED = ALEXNET_CONV_LAYERS  # ['conv1'...'conv5', 'fc1', 'fc2']
        attention_layers_default = ['conv3', 'conv4', 'conv5']  # AlexNet只用卷积层（Attention Transfer不支持FC层）
        print(f"[PPO] Using AlexNet with {len(CONV_LAYERS_USED)} layers (5 conv + 2 fc)")
    elif model_type == "lenet5":
        # LeNet-5流程（2个卷积层 + 2个FC层 = 4层）
        teacher = _build_teacher_int8_lenet5(args)
        student = _build_student_fq_lenet5(args).to(device)
        CONV_LAYERS_USED = LENET5_LAYERS  # ['conv1', 'conv2', 'fc1', 'fc2']
        attention_layers_default = ['conv1', 'conv2']  # LeNet-5只有2个卷积层（Attention Transfer不支持FC层）
        print(f"[PPO] Using LeNet-5 with {len(CONV_LAYERS_USED)} layers (2 conv + 2 fc)")
    elif model_type == "nin":
        # NiN流程
        teacher = _build_teacher_int8_nin(args)
        student = _build_student_fq_nin(args).to(device)
        CONV_LAYERS_USED = NIN_LAYERS
        attention_layers_default = ['conv1', 'conv2', 'conv3']
        print(f"[PPO] Using NiN with {len(CONV_LAYERS_USED)} layers")
    elif model_type == "mobilenetv1":
        # MobileNetV1流程
        teacher = _build_teacher_int8_mobilenetv1(args)
        student = _build_student_fq_mobilenetv1(args).to(device)
        CONV_LAYERS_USED = MOBILENETV1_LAYERS
        attention_layers_default = ['dw4', 'dw8', 'dw12'] # 选取一些中间层
        print(f"[PPO] Using MobileNetV1 with {len(CONV_LAYERS_USED)} layers")
    else:
        # VGG16流程（原有代码）
        teacher = _build_teacher_int8(args)
        student = _build_student_fq(args).to(device)
        CONV_LAYERS_USED = CONV_LAYERS
        attention_layers_default = ['conv3_1', 'conv4_1', 'conv5_1']
        print(f"[PPO] Using VGG16 with {len(CONV_LAYERS_USED)} conv layers")

    print(f"[PPO] Teacher and Student models loaded.")
    
    # Collect histograms for initialization
    print(f"[PPO] Collecting histograms for initialization...")
    calib_batches = int(getattr(args, "calib_batches", 8))
    # Fix: Use train_loader instead of test_loader to avoid data leakage
    # 根据模型类型收集直方图
    if model_type in ["alexnet", "lenet5"]:
        # AlexNet/LeNet-5: 卷积层用 block_output.{lyr}, FC层用 classifier.{lyr}.out
        hists = {}
        for lyr in CONV_LAYERS_USED:
            if lyr.startswith('conv'):
                hists[f"block_output.{lyr}"] = torch.zeros(256, dtype=torch.long)
            elif lyr.startswith('fc'):
                hists[f"classifier.{lyr}.out"] = torch.zeros(256, dtype=torch.long)
        
        it = 0
        for images, _ in train_loader:
            rec = ActivationRecorder(store_cpu=True)
            _ = teacher(images, recorder=rec)
            for lyr in CONV_LAYERS_USED:
                if lyr.startswith('conv'):
                    k = f"block_output.{lyr}"
                elif lyr.startswith('fc'):
                    k = f"classifier.{lyr}.out"
                
                if k in rec.storage:
                    v = rec.storage[k].flatten()
                    hists[k] += torch.bincount(v, minlength=256).to(torch.long)
            it += 1
            if it >= calib_batches:
                break
    else:
        hists = _collect_histograms(teacher, train_loader, num_batches=calib_batches, layers=CONV_LAYERS_USED)
    
    # Compute initial parameters and fixed tmax for each layer based on histograms
    layer_init_params = {}
    layer_tmax_codes = {}
    
    # Print tmax configuration
    if args.tmax_mode == "fixed":
        print(f"[PPO] tmax mode: FIXED (value={args.tmax_fixed})")
    else:
        print(f"[PPO] tmax mode: PERCENTILE (p{int(args.tmax_percentile)} + {args.tmax_offset})")
    
    # Verify Student Model Baseline Accuracy
    print("[PPO] Verifying Student Model Baseline Accuracy (No Approximation)...")
    student.eval()
    correct_s = 0
    total_s = 0
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = student(images)  # No recorder
            pred = logits.argmax(1)
            correct_s += (pred == targets).sum().item()
            total_s += targets.size(0)
            if total_s >= 1000: break # Check first 1000 samples
    print(f"[PPO] Student Baseline Accuracy: {100.0 * correct_s / total_s:.2f}%")
    student.train()

    for layer_name in CONV_LAYERS_USED:
        # Compute tmax from histogram: 90% percentile of non-zero data + 10
        # 卷积层用 block_output.{layer_name}, FC层用 classifier.{layer_name}.out
        if layer_name.startswith('conv'):
            hist_key = f"block_output.{layer_name}"
        elif layer_name == 'fc':
            hist_key = "fc.out"
        elif layer_name.startswith('fc'):
            hist_key = f"classifier.{layer_name}.out"
        else:
            hist_key = f"block_output.{layer_name}"  # fallback
        
        if hist_key not in hists:
            hist_key = layer_name  # fallback to layer_name directly
        
        # Calculate tmax based on mode
        if args.tmax_mode == "fixed":
            # Fixed tmax value
            tmax_code = args.tmax_fixed
        else:
            # Percentile + offset mode (default)
            if hist_key in hists:
                hist = hists[hist_key]
                # Get specified percentile code
                percentile = args.tmax_percentile / 100.0  # Convert to 0-1 range
                pN_code = _q_nonzero_from_hist(hist, percentile)
                tmax_code = pN_code + args.tmax_offset
                # Ensure tmax is at least 15 for reasonable range
                tmax_code = max(15, min(255, tmax_code))
            else:
                # Fallback if histogram not found
                tmax_code = 32
        
        layer_tmax_codes[layer_name] = tmax_code
        
        # t1 = 0.35 * tmax
        t1_code = int(0.35 * tmax_code)
        t1_code = max(2, min(tmax_code - 8, t1_code))
        
        # t2 = 0.7 * tmax
        t2_code = int(0.7 * tmax_code)
        t2_code = max(t1_code + 4, min(tmax_code - 4, t2_code))
        
        # v1 = midpoint between t1 and t2
        v1_code = (t1_code + t2_code) // 2
        v1_code = max(t1_code + 1, min(t2_code - 1, v1_code))
        
        # v2 = midpoint between t2 and tmax
        v2_code = (t2_code + tmax_code) // 2
        v2_code = max(t2_code + 1, min(tmax_code, v2_code))
        
        layer_init_params[layer_name] = (t1_code, v1_code, t2_code, v2_code)
        
        # Display tmax calculation method
        if args.tmax_mode == "fixed":
            tmax_info = f"fixed={args.tmax_fixed}"
        else:
            tmax_info = f"p{int(args.tmax_percentile)}+{args.tmax_offset}"
        print(f"  [init] {layer_name}: t1={t1_code}, v1={v1_code}, t2={t2_code}, v2={v2_code}, tmax={tmax_code} ({tmax_info})")
    
    # Initialize State Encoder (4D state vector)
    print(f"[PPO] Initializing 4D state encoder...")
    state_encoder = SimpleStateEncoder(histograms=hists, conv_layers=CONV_LAYERS_USED)
    state_encoder.print_statistics()
    
    # Initialize Attention Transfer (3 key layers for cost-efficiency)
    print(f"[PPO] Initializing Attention Transfer module...")
    attention_layers = attention_layers_default
    attention_transfer = AttentionTransfer(layer_names=attention_layers).to(device)
    attention_weight = 100.0  # Weight for attention loss in reward (increased from 0.15 to make it effective)
    use_attention_transfer = True  # Enable attention transfer for multi-layer feature alignment
    print(f"[PPO] Attention Transfer: {len(attention_layers)} layers, weight={attention_weight}")
    
    # Initialize PPO agent with 4D state input
    state_dim = 4  # Fixed: 4D state vector
    hidden_dim = 64
    policy = ActorCritic(state_dim=state_dim, hidden_dim=hidden_dim).to(device)
    print(f"[PPO] ActorCritic initialized with state_dim={state_dim}, hidden_dim={hidden_dim}")
    
    # PPO hyperparameters
    lr_ppo = float(getattr(args, "lr", 3e-4))
    gamma = 0.99
    K_epochs = 4
    eps_clip = 0.2
    
    ppo_agent = PPO(
        policy=policy,
        lr=lr_ppo,
        betas=(0.9, 0.999),
        gamma=gamma,
        K_epochs=K_epochs,
        eps_clip=eps_clip,
        v_loss_coef=0.5,
        ent_coef=0.01
    )
    
    rollout_buffer = RolloutBuffer()

    # Training settings
    max_episodes = int(getattr(args, "episodes", 500))
    update_timestep = int(getattr(args, "update_timestep", 50))
    eval_every = int(getattr(args, "eval_every", 50))  # Evaluate every N episodes
    T = float(getattr(args, "kd_T", 2.0))
    max_acc_drop = float(getattr(args, "max_acc_drop", 2.0))
    
    print(f"[PPO] Starting training for {max_episodes} episodes...")
    print(f"[PPO] lr={lr_ppo}, gamma={gamma}, K_epochs={K_epochs}, eps_clip={eps_clip}")
    
    timestep = 0
    # Track top-30 configurations by reward (smallest absolute value = best)
    # Each entry: {'reward': float, 'params': dict, 'episode': int}
    top_configs = []  # Will maintain top 30 configs sorted by |reward| (ascending)
    max_top_configs = 30
    
    # Get scale cache for all layers (including FC layers for AlexNet)
    for images, _ in train_loader:
        _, scale_cache = teacher_forward_with_scales(teacher, images.cpu())
        break
    
    # For AlexNet and VGG19 FC layers, ensure we have scale information
    if model_type == "alexnet":
        # FC层的scale需要特殊处理，因为teacher_forward_with_scales可能只返回conv层
        # 如果没有fc层的scale，使用默认值
        for lyr in ['fc1', 'fc2']:
            if lyr not in scale_cache:
                scale_cache[lyr] = (1.0, 0.0)  # 默认scale
    elif model_type == "vgg19":
        # VGG19现在只有一个fc层
        if 'fc' not in scale_cache:
            scale_cache['fc'] = (1.0, 0.0)
    
    # Create a reusable data iterator
    train_iter = iter(train_loader)
    
    # Helper function to evaluate model accuracy
    @torch.no_grad()
    def evaluate_accuracy(layer_actions_dict, max_batches=20):
        """Evaluate both teacher and student (with approximations) accuracy."""
        student.eval()
        total, correct_s, correct_t = 0, 0, 0
        
        for batch_idx, (images, targets) in enumerate(test_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
                
            images_cpu = images
            targets = targets.to(device)
            images = images.to(device)
            
            # Teacher forward
            t_logits_cpu, scales = teacher_forward_with_scales(teacher, images_cpu)
            t_logits = t_logits_cpu.to(device)
            pred_t = t_logits.argmax(1)
            correct_t += (pred_t == targets).sum().item()
            
            # Student forward with approximations
            edits = {}
            for layer_name, action_codes in layer_actions_dict.items():
                s, z = scales.get(layer_name, (1.0, 0.0))
                tmax_code = layer_tmax_codes[layer_name]
                # 卷积层用 block_output.{layer_name}, FC层用 classifier.{layer_name}.out
                if layer_name.startswith('conv'):
                    key = f"block_output.{layer_name}"
                elif layer_name == 'fc':
                    key = "fc.out"
                elif layer_name.startswith('fc'):
                    key = f"classifier.{layer_name}.out"
                else:
                    key = f"block_output.{layer_name}"
                
                def make_edit(codes, s_, z_, tmax_):
                    def _fn(x):
                        return TriApproxINT8_PPO(x, s_, z_, codes, tmax_)
                    return _fn
                
                edits[key] = make_edit(action_codes.to(device), s, z, tmax_code)
            
            rec = EditOnlyRecorder(edits=edits)
            s_logits = student(images, recorder=rec)
            pred_s = s_logits.argmax(1)
            correct_s += (pred_s == targets).sum().item()
            
            total += targets.size(0)
        
        student.train()
        acc_t = 100.0 * correct_t / total
        acc_s = 100.0 * correct_s / total
        return acc_t, acc_s
    
    for episode in range(1, max_episodes + 1):
        # Sample a batch of images
        try:
            images, targets = next(train_iter)
        except StopIteration:
            # Restart iterator when exhausted
            train_iter = iter(train_loader)
            images, targets = next(train_iter)
            
        images = images.to(device)
        targets = targets.to(device)
        
        # Get teacher's output
        with torch.no_grad():
            t_logits_cpu, scales = teacher_forward_with_scales(teacher, images.cpu())
        t_logits = t_logits_cpu.to(device)
        
        # For each layer, sample actions (4 INT8 codes) from the policy
        layer_actions = {}
        episode_reward_sum = 0.0
        
        for layer_idx, layer_name in enumerate(CONV_LAYERS_USED):
            # Get 4D state vector from state encoder
            state_vector = state_encoder.get_state(layer_name).to(device)  # [4]
            state_vector = state_vector.unsqueeze(0)  # [1, 4] for batch processing
            tmax_code = layer_tmax_codes[layer_name]
            
            # Sample action from policy with fixed tmax
            action, action_logprob = policy.act(state_vector, tmax_code)
            
            # Store state, action, logprob
            rollout_buffer.states.append(state_vector.squeeze(0))  # Store as [4]
            rollout_buffer.actions.append(action)
            rollout_buffer.logprobs.append(action_logprob)
            rollout_buffer.tmax_codes.append(tmax_code)
            
            # Convert action to INT8 codes
            layer_actions[layer_name] = action.cpu()
            
            # Debug: print first layer's codes in first episode
            if episode == 1 and layer_idx == 0:
                print(f"  [Debug ep1] {layer_name}: tmax={tmax_code}, codes={action.cpu().tolist()}")
            
        # Build edits using sampled actions
        edits = {}
        for layer_name, action_codes in layer_actions.items():
            s, z = scales.get(layer_name, (1.0, 0.0))
            tmax_code = layer_tmax_codes[layer_name]
            # 卷积层用 block_output.{layer_name}, FC层用 classifier.{layer_name}.out
            if layer_name.startswith('conv'):
                key = f"block_output.{layer_name}"
            elif layer_name == 'fc':
                key = "fc.out"
            elif layer_name.startswith('fc'):
                key = f"classifier.{layer_name}.out"
            else:
                key = f"block_output.{layer_name}"
            
            def make_edit(codes, s_, z_, tmax_):
                def _fn(x):
                    return TriApproxINT8_PPO(x, s_, z_, codes, tmax_)
                return _fn
            
            edits[key] = make_edit(action_codes, s, z, tmax_code)
        
        # Compute attention transfer loss (if enabled)
        if use_attention_transfer:
            # Register hooks to extract features for attention transfer
            student_features = {}
            teacher_features = {}
            
            def make_hook(feature_dict, layer_name, is_teacher=False):
                def hook(module, input, output):
                    # Teacher outputs quantized tensors on CPU, need to dequantize and move to CUDA
                    if is_teacher:
                        feature_dict[layer_name] = output.dequantize().cuda()
                    else:
                        feature_dict[layer_name] = output
                return hook
            
            # Register forward hooks BEFORE forward pass
            student_hooks = []
            teacher_hooks = []
            for layer_name in attention_layers:
                # Access layer directly as attribute (e.g., 'conv3_1' -> student.conv3_1)
                if hasattr(student, layer_name):
                    student_layer = getattr(student, layer_name)
                    teacher_layer = getattr(teacher, layer_name)
                    
                    student_hooks.append(
                        student_layer.register_forward_hook(
                            make_hook(student_features, layer_name, is_teacher=False)
                        )
                    )
                    teacher_hooks.append(
                        teacher_layer.register_forward_hook(
                            make_hook(teacher_features, layer_name, is_teacher=True)
                        )
                    )
        
        # Forward pass with approximations (Student)
        rec = EditOnlyRecorder(edits=edits)
        s_logits = student(images, recorder=rec)
        
        # Teacher forward (to get features for attention transfer if enabled)
        if use_attention_transfer:
            # Teacher is on CPU, so use CPU images
            with torch.no_grad():
                _ = teacher(images.cpu())
        
        # Compute KL divergence as the measure of distance
        logp = F.log_softmax(s_logits / T, dim=1)
        q = F.softmax(t_logits / T, dim=1)
        kd_loss = F.kl_div(logp, q, reduction="batchmean") * (T * T)
        
        # Compute attention transfer loss (if enabled)
        if use_attention_transfer:
            loss_at = attention_transfer(student_features, teacher_features)
            
            # Remove hooks after this batch
            for h in student_hooks + teacher_hooks:
                h.remove()
            
            # Clear feature dictionaries
            student_features.clear()
            teacher_features.clear()
            
            # Reward: negative KL divergence + weighted attention loss (both should be minimized)
            reward = -kd_loss.item() - attention_weight * loss_at.item()
        else:
            # Reward: negative KL divergence only
            reward = -kd_loss.item()
        
        episode_reward_sum += reward
        
        # Store rewards for each layer (all layers share the same reward in this episode)
        for _ in range(len(CONV_LAYERS_USED)):
            rollout_buffer.rewards.append(reward / len(CONV_LAYERS_USED))
            rollout_buffer.is_terminals.append(False)
        
        # Mark the last layer's transition as terminal
        rollout_buffer.is_terminals[-1] = True
        
        timestep += len(CONV_LAYERS_USED)
        
        # Update policy
        if timestep % update_timestep == 0:
            ppo_agent.update(rollout_buffer)
        
        # Track top-30 configurations by reward (smaller |reward| is better)
        current_config = {
            'reward': episode_reward_sum,
            'params': {lyr: [int(x) for x in layer_actions[lyr].view(-1).tolist()] for lyr in CONV_LAYERS_USED},
            'episode': episode
        }
        
        # Insert into top_configs (sorted by |reward| ascending)
        abs_reward = abs(episode_reward_sum)
        inserted = False
        for i, cfg in enumerate(top_configs):
            if abs_reward < abs(cfg['reward']):
                top_configs.insert(i, current_config)
                inserted = True
                break
        if not inserted:
            top_configs.append(current_config)
        
        # Keep only top 30
        if len(top_configs) > max_top_configs:
            top_configs = top_configs[:max_top_configs]
        
        # Log if this enters top-30
        if len(top_configs) <= max_top_configs and current_config in top_configs:
            rank = top_configs.index(current_config) + 1
            print(f"  [Top-30] rank={rank}, reward={episode_reward_sum:.4f}, |reward|={abs_reward:.4f}")
        
        # Logging
        if episode % 10 == 0:
            top1_reward = top_configs[0]['reward'] if top_configs else float('inf')
            if use_attention_transfer:
                print(f"[Episode {episode}] reward={episode_reward_sum:.4f}, kd_loss={kd_loss.item():.4f}, "
                      f"attn_loss={loss_at.item():.4f}, top1_reward={top1_reward:.4f}")
            else:
                print(f"[Episode {episode}] reward={episode_reward_sum:.4f}, kd_loss={kd_loss.item():.4f}, "
                      f"top1_reward={top1_reward:.4f}")
            
            # Print sampled parameters for the first layer
            if len(CONV_LAYERS_USED) > 0:
                first_layer = CONV_LAYERS_USED[0]
                codes = layer_actions[first_layer]
                tmax = layer_tmax_codes[first_layer]
                print(f"  {first_layer}: t1={codes[0]}, v1={codes[1]}, t2={codes[2]}, v2={codes[3]}, tmax={tmax} (fixed)")
        
        # Periodic evaluation: evaluate accuracy every eval_every episodes
        if episode % eval_every == 0:
            print(f"\n[Eval] Episode {episode}/{max_episodes}")
            if len(top_configs) > 0:
                best_abs_reward = abs(top_configs[0]['reward'])
                print(f"  Progress: Top-1 |reward|={best_abs_reward:.4f}, Top-30 size={len(top_configs)}")
                
                # Evaluate current episode's configuration
                layer_actions_eval = {lyr: layer_actions[lyr] for lyr in CONV_LAYERS_USED}
                acc_t, acc_s = evaluate_accuracy(layer_actions_eval)
                acc_drop = acc_t - acc_s
                print(f"  Current: Teacher={acc_t:.2f}%, Student={acc_s:.2f}%, Drop={acc_drop:.2f}%")
                
                # Also evaluate top-1 configuration for comparison
                top1_config = top_configs[0]
                layer_actions_top1 = {lyr: torch.tensor(top1_config['params'][lyr], dtype=torch.long) for lyr in CONV_LAYERS_USED}
                acc_t_top1, acc_s_top1 = evaluate_accuracy(layer_actions_top1)
                acc_drop_top1 = acc_t_top1 - acc_s_top1
                print(f"  Top-1 (ep={top1_config['episode']}): Teacher={acc_t_top1:.2f}%, Student={acc_s_top1:.2f}%, Drop={acc_drop_top1:.2f}%")
            print()
    
    # Post-training evaluation: evaluate all top-30 configs
    print(f"\n[PPO] Training complete. Evaluating top {len(top_configs)} configurations on FULL test set...")
    
    evaluated_configs = []
    for i, cfg in enumerate(top_configs):
        print(f"  Evaluating config {i+1}/{len(top_configs)} (episode={cfg['episode']}, reward={cfg['reward']:.4f})...")
        layer_actions_eval = {lyr: torch.tensor(cfg['params'][lyr], dtype=torch.long) for lyr in CONV_LAYERS_USED}
        # Use full test set for final selection
        acc_t, acc_s = evaluate_accuracy(layer_actions_eval, max_batches=None)
        acc_drop = acc_t - acc_s
        
        evaluated_configs.append({
            'reward': cfg['reward'],
            'params': cfg['params'],
            'episode': cfg['episode'],
            'acc_teacher': acc_t,
            'acc_student': acc_s,
            'acc_drop': acc_drop
        })
        print(f"    → Teacher={acc_t:.2f}%, Student={acc_s:.2f}%, Drop={acc_drop:.2f}%")
    
    # Selection logic: prefer configs meeting constraint, otherwise best accuracy
    constrained_configs = [c for c in evaluated_configs if c['acc_drop'] <= max_acc_drop]
    
    if constrained_configs:
        # Choose the one with highest student accuracy among constrained
        chosen = max(constrained_configs, key=lambda x: x['acc_student'])
        selection_note = f"constrained-best (drop={chosen['acc_drop']:.2f}% <= {max_acc_drop}%)"
        print(f"\n[PPO] Selected: {selection_note}")
        print(f"      Episode={chosen['episode']}, Reward={chosen['reward']:.4f}")
        print(f"      Teacher={chosen['acc_teacher']:.2f}%, Student={chosen['acc_student']:.2f}%")
    else:
        # No config meets constraint, choose highest student accuracy
        chosen = max(evaluated_configs, key=lambda x: x['acc_student'])
        selection_note = f"best-accuracy (no config met {max_acc_drop}% constraint, min_drop={chosen['acc_drop']:.2f}%)"
        print(f"\n[PPO] Selected: {selection_note}")
        print(f"      Episode={chosen['episode']}, Reward={chosen['reward']:.4f}")
        print(f"      Teacher={chosen['acc_teacher']:.2f}%, Student={chosen['acc_student']:.2f}%")
    
    chosen_params = chosen['params']
    constrained_note = selection_note
    final_acc_t = chosen['acc_teacher']
    final_acc_s = chosen['acc_student']
    final_acc_drop = chosen['acc_drop']

    # Save best parameters
    result = {"layers": {}, "backend": args.backend, "model": model_type, "selection": constrained_note,
              "metrics": {"acc_teacher": final_acc_t, "acc_student": final_acc_s, "acc_drop": final_acc_drop}}
    for lyr in CONV_LAYERS_USED:
        codes = chosen_params[lyr]
        tmax_code = layer_tmax_codes[lyr]
        result["layers"][lyr] = {
            "t1_code": int(codes[0]),
            "v1_code": int(codes[1]),
            "t2_code": int(codes[2]),
            "v2_code": int(codes[3]),
            "tmax_code": int(tmax_code)
        }
    
    # 保存结果文件
    result_filename = getattr(args, "result_file", "result.json")
    save_dir = os.path.join(args.out, "tri_ppo_int_codes")
    ensure_dir(save_dir)
    result_path = os.path.join(save_dir, result_filename)
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[PPO] Saved {constrained_note} parameters to {result_path}")
    print(f"[PPO] Final metrics: Teacher={final_acc_t:.2f}%, Student={final_acc_s:.2f}%, Drop={final_acc_drop:.2f}%")

if __name__ == '__main__':
    # For standalone testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--out", type=str, default="./outputs")
    parser.add_argument("--backend", type=str, default="fbgemm")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--update-timestep", type=int, default=50)
    parser.add_argument("--kd-T", type=float, default=2.0)
    parser.add_argument("--max-acc-drop", type=float, default=2.0)
    parser.add_argument("--result-file", type=str, default="result.json", 
                        help="Output result filename (saved in outputs/tri_ppo_int_codes/)")
    args = parser.parse_args()
    
    train_ppo(args)
