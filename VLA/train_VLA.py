import argparse
import sys
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

from unittest.mock import MagicMock

# force run
sys.modules['lerobot.policies.groot'] = MagicMock()
sys.modules['lerobot.policies.groot.configuration_groot'] = MagicMock()
sys.modules['lerobot.policies.groot.modeling_groot'] = MagicMock()
sys.modules['lerobot.policies.groot.groot_n1'] = MagicMock()

from transformers import AutoProcessor
from peft import get_peft_model, LoraConfig

#from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from tqdm import tqdm
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from data_utils.dataloader_VLA import Comma_Continuous_VLA

'''
Normalizations hardcoded based on dataset statistics, run stats.py for reference
'''
ACTION_MEAN = torch.tensor([22.550936, -0.303842], dtype=torch.float32)
ACTION_STD  = torch.tensor([10.461417,  16.523193],  dtype=torch.float32)
ACTION_MIN  = torch.tensor([0.000000, -24.28], dtype=torch.float32)
ACTION_MAX  = torch.tensor([36.532150, 21.30],  dtype=torch.float32)

SMOLVLA_MODEL_ID = "lerobot/smolvla_base"   # Fixed: Underscore fixed
ACTION_DIM       = 2                          # [speed, steer]
LORA_RANK        = 64    #  
LORA_ALPHA       = 128   # 2× rank
LORA_DROPOUT     = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

'''
Attention Block:
q_proj (Query) — Fine-tuning it teaches the model what features to pay attention to in the driving context — speed signs, lane markings, lead vehicles.
v_proj (Value) — Once attention scores decide which tokens matter, v_proj determines what content gets pulled from them. Very impactful for behavioral learning.
k_proj (Key) — "What do I advertise about myself?" Each token uses this to say what information it holds. Pairs with q_proj. If you only have budget for 2 modules, the original LoRA paper showed q_proj + v_proj alone works well — keys matter less.

o_proj (Output) — After all attention heads run in parallel, o_proj merges them back into a single representation. It's the "reconciliation" layer. Important if you want the model to learn new ways of combining multi-head information.

Gated MLP:
up_proj — Expands the representation to a higher dimension (usually 4×). Think of it as "spreading out" the features into a richer space.
gate_proj — Runs in parallel with up_proj. Its output acts as a learned gate — it decides which of the expanded features are relevant and suppresses the rest via element-wise multiplication. This is where a lot of behavioral filtering happens.
down_proj — Projects back down to the original dimension. Compresses the gated features into the final output.
'''


# MODEL Changes


def freeze_vision_encoder(model: nn.Module) -> None:
    """ Freezes the SigLIP vision encoder inside SmolVLA. """
    if hasattr(model, "vlm") and hasattr(model.vlm, "vision_tower"):
        for param in model.vlm.vision_tower.parameters():
            param.requires_grad = False
        print("  [OK] Vision encoder (SigLIP) successfully frozen.")
    else:
        # Fallback candidate iteration loop
        candidates = ["vlm.vision_tower", "vision_tower", "vision_model"]
        for path in candidates:
            try:
                module = model
                for part in path.split("."):
                    module = getattr(module, part)
                for param in module.parameters():
                    param.requires_grad = False
                print(f"  [OK] Vision encoder frozen via fallback route: {path}")
                return
            except AttributeError:
                continue
        print("  [WARN] Could not freeze vision tower explicitly by name. Check naming conventions.")

    # we end up applying LORA to oru language bridge as well, if not, we can uncomment and utilize this
# def freeze_vision_language_bridge(model: nn.Module) -> None:
#     """ Freezes multimodal projector blocks connecting vision to text. """
#     if hasattr(model, "vlm") and hasattr(model.vlm, "multi_modal_projector"):
#         for param in model.vlm.multi_modal_projector.parameters():
#             param.requires_grad = False
#         print("  [OK] Vision-language bridge projector successfully frozen.")


def apply_lora(model: nn.Module) -> nn.Module:
    """ Injects LoRA adapters into specified blocks. """
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        modules_to_save=[],
    )
    if hasattr(model, "vlm"):
        model.vlm = get_peft_model(model.vlm, lora_config) 
        print("  [OK] LoRA injected successfully into the broader VLM wrapper.")
    else:
        model = get_peft_model(model, lora_config)

    return model


def replace_action_head(model: nn.Module, num_future_steps: int, hidden_dim: int) -> nn.Module:
    """
    Adapts the action prediction layer. Instead of predicting a single-step token,
    it tracks the trajectory planning size: (num_future_steps * ACTION_DIM).
    """
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    
    output_dim = num_future_steps * ACTION_DIM
    new_head = nn.Linear(hidden_dim, output_dim, bias=True)
    nn.init.xavier_uniform_(new_head.weight)
    nn.init.zeros_(new_head.bias)

    if hasattr(base_model, "action_expert") and hasattr(base_model.action_expert, "action_head"):
        old_head = base_model.action_expert.action_head
        print(f"  [OK] Replacing action head at: action_expert.action_head")
        print(f"        Old: {old_head}  →  New: {new_head}")
        base_model.action_expert.action_head = new_head
    elif hasattr(base_model, "action_head"):
        base_model.action_head = new_head
        print("  [OK] Swapped root level action_head module wrapper.")
    else:
        # 2. Force inject the layer directly into the unwrapped base model
        base_model.action_head = new_head
        print("  [OK] Injected missing action_head directly into the base model.")
        
    return model


# normalize actions based on dataset's mean and std precomputed using stats.py
# this data is currently hardcoded and would need to be updated when dataset changes

def normalize_actions(actions: torch.Tensor, device: torch.device) -> torch.Tensor:
    mean = ACTION_MEAN.to(device)  # (2,)
    std  = ACTION_STD.to(device)   # (2,)
    return (actions - mean) / (std + 1e-8)


def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])          # (B, N_frames, 3, H, W)
    state        = torch.stack([b["observation"]["state"] for b in batch]) # (B, N_hist, 2)
    actions      = torch.stack([b["actions"] for b in batch])              # (B, N_fut, 2)
    prompts      = [b["prompt"] for b in batch]

    return {
        "pixel_values": pixel_values,
        "observation":  {"state": state},
        "actions":      actions,
        "prompt":       prompts,
    }

# FORWARD PASS

def forward_pass(model, processor, batch: dict, device: torch.device, num_future_steps: int):
    pixel_values = batch["pixel_values"].to(device)    # (B, N_frames, 3, H, W)
    actions      = batch["actions"].to(device)         # (B, N_fut, 2)
    prompts      = batch["prompt"]

    text_inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128,
    ).to(device)

    base_policy = model.get_base_model() if hasattr(model, "get_base_model") else model

    from transformers import PreTrainedModel
    import inspect
    
    inner_vlm = None
    for name, module in base_policy.named_modules():
        if isinstance(module, PreTrainedModel):
            inner_vlm = module
            break
            
    if inner_vlm is None:
        for name, module in base_policy.named_modules():
            if hasattr(module, "forward"):
                try:
                    sig = inspect.signature(module.forward)
                    if "pixel_values" in sig.parameters and "input_ids" in sig.parameters:
                        inner_vlm = module
                        break
                except Exception:
                    pass
    
    outputs = inner_vlm(
        pixel_values=pixel_values,
        input_ids=text_inputs["input_ids"],
        attention_mask=text_inputs["attention_mask"],
        output_hidden_states=True,
        return_dict=True
    )

    last_hidden_state = outputs.hidden_states[-1]
    pooled_output = last_hidden_state[:, -1, :] 

    if hasattr(base_policy, "action_expert") and hasattr(base_policy.action_expert, "action_head"):
        pred_actions = base_policy.action_expert.action_head(pooled_output)
    elif hasattr(base_policy, "action_head"):
        pred_actions = base_policy.action_head(pooled_output)
    else:
        raise AttributeError("Could not locate the custom action head.")

    pred_actions = pred_actions.view(-1, num_future_steps, ACTION_DIM)

    norm_targets = normalize_actions(actions, device)
    norm_preds   = normalize_actions(pred_actions, device)

    mse_unreduced = F.mse_loss(norm_preds, norm_targets, reduction='none')
    mean_batch_mse = mse_unreduced.mean(dim=0)
    total_loss = mean_batch_mse.mean()
    speed_loss_intervals = mean_batch_mse[:, 0].detach().cpu().numpy() # [step1, step2, step3, step4, step5]
    steer_loss_intervals = mean_batch_mse[:, 1].detach().cpu().numpy()

    return total_loss, speed_loss_intervals.tolist(), steer_loss_intervals.tolist()


# TRAINING LOOP


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice Context initialized: {device}")
    print(f"\nLoading SmolVLA via LeRobot Ecosystem: {SMOLVLA_MODEL_ID}")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    model     = SmolVLAPolicy.from_pretrained(SMOLVLA_MODEL_ID)

    # we won't train the vision encoder, freeze it
    freeze_vision_encoder(model)
    model = apply_lora(model)
    
    # in increments we derive the speed and steer in increments of 0.2s, upto 1s into the future
    num_future_steps = 5
    replace_action_head(model, num_future_steps=num_future_steps, hidden_dim=args.hidden_dim)

    model = model.to(device)

    train_dataset = Comma_Continuous_VLA(
        chunk_path=Path(args.chunk_path),
        target_size=(384, 384),
        num_history_frames=4,
        frame_stride=5,
        future_window=1.0,
        num_future_steps=num_future_steps,
        split="train"
    )
    
    test_dataset = Comma_Continuous_VLA(
        chunk_path=Path(args.chunk_path),
        target_size=(384, 384),
        num_history_frames=4,
        frame_stride=5,
        future_window=1.0,
        num_future_steps=num_future_steps,
        split="test"
    )

    # 2. Create loaders directly
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda")
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"\nTrainable parameter weights isolated: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=2e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    training_history = {}
    best_val_loss = float('inf')

    print(f"\nTraining execution sequence running across {args.epochs} epochs...\n")

    for epoch in range(args.epochs):
        epoch_str = f"epoch_{epoch+1:03d}"

        training_history[epoch_str] = {
            "speed": {f"interval_{i+1}": [] for i in range(num_future_steps)},
            "steer": {f"interval_{i+1}": [] for i in range(num_future_steps)}
        }
        
        model.train()
        train_loss = 0.0
        train_speed_loss = 0.0
        train_steer_loss = 0.0
        batch_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")

        for batch in pbar:
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                loss, speed_loss, steer_loss = forward_pass(model, processor, batch, device, num_future_steps)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()

            for i in range(num_future_steps):
                training_history[epoch_str]["speed"][f"interval_{i+1}"].append(speed_loss[i])
                training_history[epoch_str]["steer"][f"interval_{i+1}"].append(steer_loss[i])

            batch_speed = sum(speed_loss) / len(speed_loss)
            batch_steer = sum(steer_loss) / len(steer_loss)

            train_speed_loss += batch_speed
            train_steer_loss += batch_steer
            train_loss += loss.item()
            batch_count += 1

            pbar.set_postfix({
                "spd": f"{batch_speed:.4f}", 
                "str": f"{batch_steer:.4f}"
            })

        avg_train_loss = train_loss / max(batch_count, 1)
        avg_train_speed = train_speed_loss / max(batch_count, 1)
        avg_train_steer = train_steer_loss / max(batch_count, 1)
        scheduler.step()

        model.eval()
        test_loss_total = 0.0
        test_speed_total = 0.0
        test_steer_total = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Test]"):
                with torch.amp.autocast('cuda', enabled=(device.type == "cuda"), dtype=torch.bfloat16):
                    v_loss, speed_loss, steer_loss = forward_pass(model, processor, batch, device, num_future_steps)
                
                test_loss_total += v_loss.item()
                test_speed_total += sum(speed_loss) / len(speed_loss)
                test_steer_total += sum(steer_loss) / len(steer_loss)
                test_batches += 1
                
        avg_test_loss = test_loss_total / max(test_batches, 1)
        avg_test_speed = test_speed_total / max(test_batches, 1)
        avg_test_steer = test_steer_total / max(test_batches, 1)

        print(f"\n--- Epoch {epoch+1:03d} Metric Summary ---")
        print(f"  Combined -> Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")
        print(f"  Velocity -> Train Loss: {avg_train_speed:.4f} | Test Loss: {avg_test_speed:.4f}")
        print(f"  Steering -> Train Loss: {avg_train_steer:.4f} | Test Loss: {avg_test_steer:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}\n")

        ckpt_path = output_dir / epoch_str
        ckpt_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path / "policy.pt")
        processor.save_pretrained(ckpt_path)
        
        if avg_test_loss < best_val_loss:
            best_val_loss = avg_test_loss
            best_path = output_dir / "best_model"
            best_path.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_path / "policy.pt")
            processor.save_pretrained(best_path)
            print(f"  [⭐] New Best Model Saved to: {best_path}")

        with open(output_dir / "loss_history.json", "w") as f:
            json.dump(training_history, f, indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune SmolVLA for highway driving")
    parser.add_argument("--chunk_path",  default= Path("comma2k19_data") / "extracted" / "Chunk_1")
    parser.add_argument("--output_dir",  default= Path("VLA") / "checkpoints")
    parser.add_argument("--epochs",      type=int,   default=1)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=2e-4)
    parser.add_argument("--num_workers", type=int,   default=2)
    parser.add_argument("--hidden_dim",  type=int,   default=960)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)