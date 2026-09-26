
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import numpy as np
import copy
import time
import os
from torch.nn import Module
from models.binaryae import BinaryAutoEncoder, Generator
from models.transformer import TransformerBD
from binarylatent_bitdance import BinaryBitDanceDecouple
from hparams import get_sampler_hparams
from utils.data_utils import get_data_loaders
from utils.sampler_utils import retrieve_autoencoder_components_state_dicts,\
    get_online_samples, get_online_samples_guidance
from utils.train_utils import EMA, NativeScalerWithGradNormCount
from utils.log_utils import log, log_stats, config_log, start_training_log, \
    save_stats, load_stats, save_model, load_model, save_images, \
    MovingAverage
import misc
import torch.distributed as dist
from utils.lr_sched import adjust_lr, lr_scheduler
from PIL import Image
from einops import rearrange
import torchvision

from torch.utils.data import Dataset, DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.transforms import functional as F
from PIL import Image
import os
import time
import psutil
import gc
def get_bitdance_sampler(H):
    """Create the controlled-backbone BitDance-B adapter."""
    if H.sampler != 'bitdance_binary':
        raise ValueError(
            f"This training entry only supports sampler='bitdance_binary', got {H.sampler!r}."
        )
    denoise_fn = TransformerBD(H).cuda()
    sampler = BinaryBitDanceDecouple(H, denoise_fn, H.codebook_size)
    report = sampler.parameter_report()
    print(
        "BitDance controlled-backbone parameters: "
        f"global={report['global']/1e6:.2f}M, "
        f"inner={report['inner']/1e6:.2f}M, "
        f"special={report['special']/1e6:.3f}M, "
        f"trainable_total={report['trainable_total']/1e6:.2f}M",
        flush=True,
    )
    print(
        "BitDance schedule: "
        f"parallel_num={sampler.parallel_num}, outer_steps={sampler.outer_steps}, "
        f"inner_infer_steps={sampler.inner_infer_steps}, "
        f"diff_batch_mul={sampler.diff_batch_mul}, perturb_rate={sampler.perturb_rate}",
        flush=True,
    )
    return sampler


@torch.no_grad()
def save_packed_latent_preview(
        H, sampler, generator, embedding_weight, step, num_images=64):
    """Generate a small EMA preview without loading the BAE encoder."""
    was_training = sampler.training
    sampler.eval()
    try:
        preview_batch_size = int(os.environ.get('BITDANCE_PREVIEW_BATCH_SIZE', '2'))
        if preview_batch_size <= 0:
            raise ValueError("BITDANCE_PREVIEW_BATCH_SIZE must be positive")
        latent_chunks = []
        for offset in range(0, num_images, preview_batch_size):
            current_batch = min(preview_batch_size, num_images - offset)
            with torch.cuda.amp.autocast(enabled=H.amp):
                latent_chunks.append(sampler.sample(
                    inner_steps=H.bitdance_inner_infer_steps,
                    temp=1.0,
                    b=current_batch,
                    return_all=False,
                ).float())
        latents = torch.cat(latent_chunks, dim=0)
        if tuple(latents.shape) != (num_images, 256, 64):
            raise RuntimeError(f"Invalid BitDance preview latent shape: {tuple(latents.shape)}")
        if not torch.all((latents == 0) | (latents == 1)):
            raise RuntimeError("BitDance preview latents are not strictly binary")
        with torch.cuda.amp.autocast(enabled=H.amp):
            if H.use_tanh:
                latents = (latents - 0.5) * 2.0
            if not H.norm_first:
                latents = latents / float(H.codebook_size)
            quantized = latents @ embedding_weight
            quantized = quantized.permute(0, 2, 1).reshape(
                num_images,
                H.emb_dim,
                H.latent_shape[1],
                H.latent_shape[2],
            )
            images = generator(quantized).float().cpu().clamp(0, 1)
        preview_dir = Path(H.log_dir) / "previews"
        preview_dir.mkdir(parents=True, exist_ok=True)
        torchvision.utils.save_image(
            images,
            preview_dir / f"samples_step_{step:06d}.png",
            nrow=8,
            padding=0,
        )
        print(
            f"Saved BitDance EMA preview at step {step} "
            f"(outer={sampler.outer_steps}, inner={H.bitdance_inner_infer_steps})",
            flush=True,
        )
    finally:
        sampler.train(was_training)



def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"内存使用: {process.memory_info().rss / 1024 / 1024:.2f} MB")

class ImageDataset(Dataset):
    def __init__(self, folder_path, max_images=None):
        self.folder_path = folder_path
        
        # 获取所有有效的图像文件
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.image_files = []
        
        for f in os.listdir(folder_path):
            if f.lower().endswith(valid_extensions):
                full_path = os.path.join(folder_path, f)
                try:
                    # 验证图像是否可读
                    with Image.open(full_path) as img:
                        img.verify()
                    self.image_files.append(f)
                except Exception as e:
                    print(f"跳过损坏图像: {f}, 错误: {e}")
        
        if max_images and len(self.image_files) > max_images:
            self.image_files = self.image_files[:max_images]
            
        print(f"从 {folder_path} 找到 {len(self.image_files)} 张有效图像")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        try:
            image = Image.open(img_path).convert('RGB')
            # 直接转换为tensor，不调整大小
            image_tensor = F.to_tensor(image)  # 转换为tensor并归一化到[0,1]
            image_tensor = image_tensor.to(torch.uint8) 
            return image_tensor
        except Exception as e:
            print(f"处理图像 {img_path} 时出错: {e}")
            # 返回一个空白图像作为替代
            return torch.zeros(3, 256, 256)  # 假设所有图像都是256x256


class PackedBinaryLatentDataset(Dataset):
    """Memory-mapped [N,2048] cache of 16,384 little-endian BAE bits."""

    packed_width = 256 * 64 // 8

    def __init__(self, path):
        self.path = str(Path(path).expanduser().resolve())
        self.latents = np.load(self.path, mmap_mode="r")
        if self.latents.ndim != 2 or self.latents.shape[1] != self.packed_width:
            raise ValueError(
                f"Expected packed latent cache [N,{self.packed_width}], "
                f"got {self.latents.shape}"
            )
        if self.latents.dtype != np.uint8:
            raise ValueError(f"Expected uint8 latent cache, got {self.latents.dtype}")

    def __len__(self):
        return self.latents.shape[0]

    def __getitem__(self, index):
        packed = np.asarray(self.latents[index])
        bits = np.unpackbits(
            packed, count=256 * 64, bitorder="little"
        ).reshape(256, 64)
        return torch.from_numpy(bits)


class BinaryToDecimal(Module):
    """A Module to convert binaries encoded tensors to decimals.

    This is a utility class that allow to convert a binary encoding tensor (e.g. `1001`) to
    its decimal value (e.g. `9`)

    Args:
        num_bits (int): the number of bits to use for the bases table.
            The number of bits must be lower or equal to the input length and the input length
            must be divisible by ``num_bits``. If ``num_bits`` is lower than the number of
            bits in the input, the end result will be aggregated on the last dimension using
            :func:`~torch.sum`.
        device (torch.device): the device where inputs and outputs are to be expected.
        dtype (torch.dtype): the output dtype.
        convert_to_binary (bool, optional): if ``True``, the input to the ``forward``
            method will be cast to a binary input using :func:`~torch.heavyside`.
            Defaults to ``False``.

    Examples:
        >>> binary_to_decimal = BinaryToDecimal(
        ...    num_bits=4, device="cpu", dtype=torch.int32, convert_to_binary=True
        ... )
        >>> binary = torch.Tensor([[0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 10, 0]])
        >>> decimal = binary_to_decimal(binary)
        >>> assert decimal.shape == (2,)
        >>> assert (decimal == torch.Tensor([3, 2])).all()
    """

    def __init__(
        self,
        num_bits: int,
        device: torch.device,
        dtype: torch.dtype,
        convert_to_binary: bool = False,
    ):
        super().__init__()
        self.convert_to_binary = convert_to_binary
        self.bases = 2 ** torch.arange(num_bits - 1, -1, -1, device=device, dtype=dtype)
        self.num_bits = num_bits
        self.zero_tensor = torch.zeros((1,), device=device)
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        num_features = features.shape[-1]
        if self.num_bits > num_features:
            raise ValueError(f"{num_features=} is less than {self.num_bits=}")
        elif num_features % self.num_bits != 0:
            raise ValueError(f"{num_features=} is not divisible by {self.num_bits=}")

        binary_features = (
            torch.heaviside(features, self.zero_tensor)
            if self.convert_to_binary
            else features
        )
        feature_parts = binary_features.reshape(shape=(-1, self.num_bits))
        digits = torch.vmap(torch.dot, (None, 0))(
            self.bases, feature_parts.to(self.bases.dtype)
        )
        digits = digits.reshape(shape=(-1, features.shape[-1] // self.num_bits))
        aggregated_digits = torch.sum(digits, dim=-1)
        return aggregated_digits

def decode_from_binary_code(vq_gan_hack, binary_code, H):
    decimal_code = from_binary_tensor(binary_code) # [b 64*64]
    real_value_quant = vq_gan_hack.quantize.embedding(decimal_code).view(H.batch_size, H.latent_shape[1], H.latent_shape[2], H.latent_shape[0])
    real_value_quant = rearrange(real_value_quant, 'b h w c -> b c h w').contiguous()
    images = vq_gan_hack.decode(real_value_quant, force_not_quantize = True)
    images = torch.clamp(images, -1., 1.)
    images = (images + 1.) / 2.
    return images


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x

def preprocess_image(image_path):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = np.array(image).astype(np.uint8)
    image = (image/127.5 - 1.0).astype(np.float32)
    return image


# def load_model(config, ckpt, gpu, eval_mode):
#     if ckpt:
#         print(f"Loading model from {ckpt}")
#         pl_sd = torch.load(ckpt, map_location="cpu")
#         global_step = pl_sd["global_step"]
#     else:
#         pl_sd = {"state_dict": None}
#         global_step = None
#     model = load_model_from_config(config.model,
#                                    pl_sd["state_dict"])

#     return model, global_step


def main(H, vis):

    misc.init_distributed_mode(H)

    #ldm ae
    # ldm_ae_dir = "./ldm/models/ldm/ffhq256"
    # ckpt = "./ldm/models/ldm/ffhq256/model.ckpt"
    # base_configs = sorted(glob.glob(os.path.join(ldm_ae_dir, "config.yaml")))

    # configs = [OmegaConf.load(cfg) for cfg in base_configs]
    # config = configs[0]
 
    # model

    # gpu = True
    # eval_mode = True
    # ldm_model, _ = load_model(config, ckpt, gpu, eval_mode)
    # vq_gan_hack = ldm_model.first_stage_model

    bergan = None
    preview_generator = None
    preview_embedding_weight = None
    if getattr(H, "latent_cache", None):
        preview_state = retrieve_autoencoder_components_state_dicts(
            H,
            ['quantize', 'generator'],
            remove_component_from_key=True,
        )
        preview_embedding_weight = preview_state.pop('embed.weight').cuda()
        preview_generator = Generator(H)
        preview_generator.load_state_dict(preview_state, strict=False)
        preview_generator = preview_generator.cuda().eval()
        for parameter in preview_generator.parameters():
            parameter.requires_grad_(False)
        del preview_state
    else:
        ae_state_dict = retrieve_autoencoder_components_state_dicts(
            H,
            ['encoder', 'quantize', 'generator'],
            remove_component_from_key=False
        )

        bergan = BinaryAutoEncoder(H)
        bergan.load_state_dict(ae_state_dict, strict=True)
        bergan = bergan.cuda()
        del ae_state_dict


    sampler = get_bitdance_sampler(H).cuda()
    sampler_without_ddp = sampler

    if H.ema:
        ema = EMA(H.ema_beta)
        ema_sampler = copy.deepcopy(sampler)

    if H.distributed:
        find_unused = H.guidance
        sampler = torch.nn.parallel.DistributedDataParallel(sampler, device_ids=[H.gpu], find_unused_parameters=find_unused)
        sampler_without_ddp = sampler.module

    optim_eps = H.optim_eps
    optim = torch.optim.AdamW(sampler_without_ddp.parameters(), lr=H.lr, weight_decay=H.weight_decay, betas=(0.9, 0.95), eps=optim_eps)

    losses = np.array([])
    val_losses = np.array([])
    elbo = np.array([])
    val_elbos = np.array([])
    mean_losses = np.array([])
    start_step = 0
    log_start_step = 0

    loss_ma = MovingAverage(100)

    if H.load_model_step > 0:
        device = next(sampler.parameters()).device
        sampler = load_model(sampler, H.sampler, H.load_model_step, H.load_model_dir, device=device).cuda()

        
    scaler = NativeScalerWithGradNormCount(H.amp, H.init_scale)

    if H.load_step > 0:
        start_step = H.load_step + 1

        device = next(sampler.parameters()).device

        allow_mismatch = H.allow_mismatch
        sampler = load_model(sampler, H.sampler, H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch).cuda()
        if H.ema:
            # if EMA has not been generated previously, recopy newly loaded model
            try:
                ema_sampler = load_model(
                    ema_sampler, f'{H.sampler}_ema', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
            except Exception:
                ema_sampler = copy.deepcopy(sampler_without_ddp)
        
        if not allow_mismatch:
            if H.load_optim:
                optim = load_model(
                    optim, f'{H.sampler}_optim', H.load_step, H.load_dir, device=device, allow_mismatch=allow_mismatch)
                for param_group in optim.param_groups:
                    param_group['lr'] = H.lr
        try:
            train_stats = load_stats(H, H.load_step)
        except Exception:
            train_stats = None

        if not H.reset_step:
            if not H.reset_scaler:
                try:
                    scaler.load_state_dict(torch.load(os.path.join(
                        H.load_dir,
                        'saved_models',
                        f'{H.sampler}_scaler_{H.load_step}.th',
                    )))
                except Exception:
                    print('Failing to load scaler.')
        else:
            H.load_step = 0

        
        if train_stats is not None:
            losses, mean_losses, val_losses, elbo, H.steps_per_log

            losses = train_stats["losses"],
            mean_losses = train_stats["mean_losses"],
            val_losses = train_stats["val_losses"],
            val_elbos = train_stats["val_elbos"]
            log_start_step = 0

            losses = losses[0]
            mean_losses = mean_losses[0]
            val_losses = val_losses[0]
            val_elbos = torch.Tensor([0])

        else:
            log('No stats file found for loaded model, displaying stats from load step only.')
            log_start_step = start_step

        if H.reset_step:
            start_step = 0
    
  
    
    # train_dataset = instantiate_from_config(config.data.params.train)
    # val_dataset = instantiate_from_config(config.data.params.validation)

    # if H.distributed:
    #     num_tasks = misc.get_world_size()
    #     global_rank = misc.get_rank()
    #     sampler_train = torch.utils.data.DistributedSampler(
    #         train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
    #     )
    #     sampler_val = torch.utils.data.DistributedSampler(
    #         val_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=False
    #     )
    #     print("Sampler_train = %s" % str(sampler_train))
            
    # else:
    #     sampler_train = torch.utils.data.RandomSampler(train_dataset)
    #     sampler_val = torch.utils.data.SequentialSampler(val_dataset)

    # train_loader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         num_workers=4,
    #         sampler=sampler_train,
    #         batch_size=H.batch_size,
    #         pin_memory=True,
    #         drop_last=True
    #     )
    # val_loader = torch.utils.data.DataLoader(
    #         val_dataset,
    #         num_workers=4,
    #         sampler=sampler_val,
    #         batch_size=H.batch_size,
    #         pin_memory=True,
    #         drop_last=False  
    #     )

    if getattr(H, "latent_cache", None):
        train_dataset = PackedBinaryLatentDataset(H.latent_cache)
        train_sampler = (
            torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
            if H.distributed
            else None
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=H.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
        )
        val_loader = None
        print(
            f"Training directly from packed latent cache: "
            f"{train_dataset.path} ({len(train_dataset)} examples)",
            flush=True,
        )
    else:
        train_loader, val_loader = get_data_loaders(
            H.dataset,
            H.img_size,
            H.batch_size,
            get_val_dataloader=False,
            custom_dataset_path=H.path_to_data,
            num_workers=4,
            distributed=H.distributed,
            random=True,
            args=H,
        )

    # Public steps count optimizer updates, independent of gradient
    # accumulation. This keeps checkpoint 10000 semantically equal across
    # batch-size/update-frequency settings.
    lr_sched = lr_scheduler(base_value=H.lr, final_value=1e-6, iters=H.train_steps+1, warmup_steps=H.warmup_iters,
                     start_warmup_value=1e-6, lr_type='constant')
    print(lr_sched)
    completed_step = H.load_step
    micro_step = 0
    epoch = -1

    optim.zero_grad()
    label = None
    while True:
        epoch += 1
        if H.distributed:
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            micro_step += 1

            adjust_lr(optim, lr_sched, min(completed_step + 1, H.train_steps))
            step_start_time = time.time()
            #[b,3,256,256]
            if getattr(H, "latent_cache", None):
                x = batch.cuda(non_blocking=True).float()
            else:
                data, _ = batch
                img = data.cuda()
                with torch.no_grad():
                    code = bergan(img, code_only=True).detach()
                    b,c,h,w = code.shape
                    x = code.view(b,c,-1).permute(0,2,1).contiguous()

            with torch.cuda.amp.autocast(enabled=H.amp):
                if H.dataset.startswith('imagenet'):
                    stats = sampler(x, label)
                else:
                    stats = sampler(x)
                loss = stats['loss']
                loss = loss / H.update_freq

            if bergan is not None and micro_step == 1 and completed_step == 0 and (not H.distributed or dist.get_rank() == 0):
                images = get_online_samples(H, bergan, ema_sampler if H.ema else sampler, x=x)
                save_images(images, 'samples', 999999999, H.log_dir, H.save_individually)
                # save to test the reconstruction quality

            should_update = micro_step % H.update_freq == 0
            grad_norm = scaler(loss, optim, clip_grad=H.grad_norm,
                                parameters=sampler_without_ddp.parameters(), create_graph=False,
                                update_grad=should_update)

            if should_update:
                optim.zero_grad()
            losses = np.append(losses, loss.detach().float().item())
            loss_ma.update(loss.item())

            torch.cuda.synchronize()

            if not should_update:
                continue

            completed_step += 1
            if H.ema and completed_step % H.steps_per_update_ema == 0:
                ema.update_model_average(ema_sampler, sampler)

            if not H.distributed or dist.get_rank() == 0:
                if completed_step == 1 or completed_step % H.steps_per_log == 0:

                    stats['lr'] = optim.param_groups[0]['lr']
                    step_time_taken = time.time() - step_start_time
                    stats['step_time'] = step_time_taken
                    mean_loss = np.mean(losses)
                    stats['mean_loss'] = loss_ma.avg()

                    if "scale" in scaler.state_dict().keys():
                        stats['loss scale'] = scaler.state_dict()["scale"]
                    mean_losses = np.append(mean_losses, mean_loss)
                    losses = np.array([])

                    log_stats(completed_step, stats)

                if completed_step % H.steps_per_save_output == 0:
                    preview_sampler = ema_sampler if H.ema else sampler
                    if preview_generator is not None:
                        save_packed_latent_preview(
                            H,
                            preview_sampler,
                            preview_generator,
                            preview_embedding_weight,
                            completed_step,
                        )
                    elif bergan is not None:
                        if H.guidance:
                            images = get_online_samples_guidance(H, bergan, preview_sampler)
                        else:
                            images = get_online_samples(H, bergan, preview_sampler)
                        save_images(images, 'samples', completed_step, H.log_dir, H.save_individually)

                if completed_step % H.steps_per_checkpoint == 0 and completed_step > H.load_step:
                    save_model(sampler, H.sampler, completed_step, H.log_dir)
                    save_model(optim, f'{H.sampler}_optim', completed_step, H.log_dir)
                    save_model(scaler, f'{H.sampler}_scaler', completed_step, H.log_dir)

                    if H.ema:
                        save_model(ema_sampler, f'{H.sampler}_ema', completed_step, H.log_dir)

                    train_stats = {
                        'losses': losses,
                        'mean_losses': mean_losses,
                        'val_losses': val_losses,
                        'elbo': elbo,
                        'val_elbos': val_elbos,
                        'steps_per_log': H.steps_per_log,
                        'steps_per_eval': H.steps_per_eval,
                    }
                    save_stats(H, train_stats, completed_step)
            
            if completed_step >= H.train_steps:
                if not H.distributed or dist.get_rank() == 0:
                    print(f"BitDance training completed at step {completed_step}.")
                return


def get_bitdance_hparams_compat():
    """Parse the existing sampler CLI without requiring global hparams edits.

    The local project already accepts ``dfm_binary`` for the DFM baseline.
    We temporarily route ``bitdance_binary`` through that parser, then restore
    the method name and attach BitDance-specific settings from environment
    variables so this experiment stays self-contained.
    """
    original_argv = list(sys.argv)
    rewritten = list(sys.argv)
    for i, arg in enumerate(rewritten[:-1]):
        if arg == '--sampler' and rewritten[i + 1] == 'bitdance_binary':
            rewritten[i + 1] = 'dfm_binary'
    sys.argv = rewritten
    try:
        H = get_sampler_hparams()
    finally:
        sys.argv = original_argv

    H.sampler = 'bitdance_binary'
    H.bitdance_parallel_num = int(os.environ.get('BITDANCE_PARALLEL_NUM', '4'))
    H.bitdance_inner_infer_steps = int(os.environ.get('BITDANCE_INNER_INFER_STEPS', '100'))
    H.bitdance_diff_batch_mul = int(os.environ.get('BITDANCE_DIFF_BATCH_MUL', '1'))
    H.bitdance_diff_dim = int(os.environ.get('BITDANCE_DIFF_DIM', str(H.bert_n_emb)))
    H.bitdance_diff_layers = int(os.environ.get('BITDANCE_DIFF_LAYERS', '6'))
    H.bitdance_diff_adaln_layers = int(os.environ.get('BITDANCE_DIFF_ADALN_LAYERS', '2'))
    H.bitdance_perturb_rate = float(os.environ.get('BITDANCE_PERTURB_RATE', '0.1'))
    H.bitdance_time_schedule = os.environ.get('BITDANCE_TIME_SCHEDULE', 'logit_normal')
    H.bitdance_time_shift = float(os.environ.get('BITDANCE_TIME_SHIFT', '1.0'))
    H.bitdance_p_std = float(os.environ.get('BITDANCE_P_STD', '0.8'))
    H.bitdance_p_mean = float(os.environ.get('BITDANCE_P_MEAN', '-0.8'))
    return H


if __name__ == '__main__':
    H = get_bitdance_hparams_compat()
    config_log(H.log_dir)
    log('---------------------------------')
    log(f'Setting up training for {H.sampler}')
    start_training_log(H)
    main(H, None)
