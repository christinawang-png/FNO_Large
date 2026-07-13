import torch
from pathlib import Path
from torchvision.utils import save_image
import random

# import model + dataset definitions
from train import PlaneDatasetParamsToImage, FNOPlusResNet  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir   = Path("./plane_dataset_2")
image_csv  = base_dir / "renders_Larger" / "metadata_images_None.csv"
volume_csv = base_dir / "metadata_volumes.csv"

full_dataset = PlaneDatasetParamsToImage(
    image_csv_path=str(image_csv),
    volume_csv_path=str(volume_csv),
    img_size=(64, 64),
    use_sh=True,
    normalize_params=True,
)
latent_dim = full_dataset.latent_dim

# fixed indices for comparison (choose once)
fixed_indices = [10, 123, 456, 789]  # or random.sample(range(len(full_dataset)), 4)

def load_model(ckpt_path):
    model = FNOPlusResNet(latent_dim=latent_dim, img_size=(64, 64)).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"]
    state.pop("_metadata", None)
    model.load_state_dict(state)
    model.eval()
    return model

def eval_model(ckpt_path, tag):
    model = load_model(ckpt_path)
    out_dir = Path(f"qualitative_fixed_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for idx in fixed_indices:
            param_vec, img_gt = full_dataset[idx]
            param_vec = param_vec.unsqueeze(0).to(device)
            img_gt    = img_gt.unsqueeze(0).to(device)
            img_pred  = model(param_vec).clamp(0, 1)

            sbs = torch.cat([img_gt, img_pred], dim=3)
            fname = out_dir / f"idx_{idx:06d}.png"
            save_image(sbs, fname)
            print(f"[{tag}] Saved:", fname)

# run for base (MSE) and finetuned models
eval_model("fno_params_to_image_large.pt", "mse")
eval_model("fno_params_to_image_finetuned_large.pt", "finetune")