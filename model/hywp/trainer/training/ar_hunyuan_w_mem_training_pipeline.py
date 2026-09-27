# SPDX-License-Identifier: Apache-2.0
from copy import deepcopy
import os
import sys
sys.path.append(os.path.abspath('.'))

import torch
import torch.distributed as dist

from trainer.trainer_args import TrainerArgs, TrainingArgs
from trainer.logger import init_logger
from trainer.training.ar_hunyuan_mem_training_pipeline import TrainingPipeline
from trainer.utils import is_vsa_available

vsa_available = is_vsa_available()

logger = init_logger(__name__)


class HunyuanTrainingPipeline(TrainingPipeline):
    """
    A training pipeline for Hunyuan.
    """
    _required_config_modules = ["transformer"]

    def initialize_pipeline(self, trainer_args: TrainerArgs):
        pass

    def create_training_stages(self, training_args: TrainingArgs):
        pass

    def initialize_validation_pipeline(self, training_args: TrainingArgs):
        pass

    def _maybe_eval(self, step: int) -> None:
        """Every validation_steps, render the seen/unseen eval trajectories from the
        checkpoint just saved, in a background subprocess on --eval_gpus."""
        if not self.training_args.temporal_crossattn_per_block_training:
            return
        eval_trajectories = []
        if self.training_args.eval_pose_json and self.training_args.eval_image_path:
            eval_trajectories.append(("seen", self.training_args.eval_pose_json, self.training_args.eval_image_path))
        unseen_pose = getattr(self.training_args, 'eval_pose_json_unseen', '')
        unseen_img  = getattr(self.training_args, 'eval_image_path_unseen', '')
        if unseen_pose and unseen_img:
            eval_trajectories.append(("unseen", unseen_pose, unseen_img))
        if not eval_trajectories:
            return

        import datetime

        # Extend NCCL timeout so ranks 1-3 can wait for eval subprocess(es)
        _eval_timeout = datetime.timedelta(hours=4)
        torch.distributed.distributed_c10d._set_pg_timeout(_eval_timeout)

        # Use the checkpoint saved at this step (already contains all trainable weights)
        tmp_ckpt = os.path.join(
            self.training_args.output_dir,
            f"checkpoint-{step}", "transformer", "diffusion_pytorch_model.safetensors"
        )
        dist.barrier()  # ensure checkpoint is fully written before subprocess reads

        # Non-blocking eval: rank 0 starts a background thread that runs the eval
        # subprocesses (queued behind any previous eval). Training resumes immediately
        # on all ranks; the evals run on --eval_gpus.
        if self.global_rank == 0:
            import threading

            action_ckpt = self.training_args.eval_action_ckpt or ""
            num_eval_gpus = len(self.training_args.eval_gpus.split(","))
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = self.training_args.eval_gpus
            hywp_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            env["PYTHONPATH"] = hywp_root + os.pathsep + env.get("PYTHONPATH", "")

            prev = getattr(self, '_eval_thread', None)

            def _run_evals():
                import shutil
                import subprocess as _sp
                if prev is not None:
                    prev.join()  # evals run one after another, never concurrently
                if not os.path.isfile(tmp_ckpt):
                    logger.warning("step %d  checkpoint %s no longer exists (checkpoints_total_limit) "
                                   "— skipping eval", step, tmp_ckpt)
                    return
                port = int(os.environ.get("EVAL_START_PORT", "29810"))
                for tag, pose_json, image_path in eval_trajectories:
                    out_dir = os.path.join(self.training_args.output_dir, f"eval_step{step}_{tag}")
                    os.makedirs(out_dir, exist_ok=True)
                    pose_string = os.path.join(os.path.dirname(pose_json), "pose_string.txt")
                    if os.path.isfile(pose_string):
                        shutil.copy(pose_string, out_dir)
                    log_path = os.path.join(out_dir, "eval.log")
                    cmd = [
                        sys.executable, "-m", "torch.distributed.run",
                        f"--nproc_per_node={num_eval_gpus}",
                        f"--master_port={port}",
                        os.path.join(hywp_root, "scripts", "inference.py"),
                        "--model_path", self.training_args.pretrained_model_name_or_path,
                        "--action_ckpt", action_ckpt,
                        "--temporal_embed_ckpt", tmp_ckpt,
                        "--pose_json", pose_json,
                        "--image_path", image_path,
                        "--output_dir", out_dir,
                        "--num_inference_steps", "30",
                        "--seed", "42",
                    ]
                    eval_prompt = getattr(self.training_args, "eval_prompt", "")
                    if eval_prompt:
                        cmd += ["--prompt", eval_prompt]
                    logger.info("step %d  launching %s eval on GPUs %s (non-blocking, log: %s)",
                                step, tag, self.training_args.eval_gpus, log_path)
                    with open(log_path, "w") as log_f:
                        result = _sp.run(cmd, env=env, stdout=log_f, stderr=_sp.STDOUT)
                    logger.info("step %d  %s eval done (exit=%d)", step, tag, result.returncode)
                    port += 1

            t = threading.Thread(target=_run_evals, daemon=True)
            t.start()
            self._eval_thread = t

    def train(self) -> None:
        super().train()
        t = getattr(self, '_eval_thread', None)
        if t is not None and t.is_alive():
            logger.info("Training done; waiting for the remaining background evals ...")
            t.join()


def main(args) -> None:
    logger.info("Starting training pipeline...")

    pipeline = HunyuanTrainingPipeline.from_pretrained(
        args.pretrained_model_name_or_path, args=args)
    args = pipeline.training_args
    pipeline.train()
    logger.info("Training pipeline done")


if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    argv = sys.argv
    from trainer.trainer_args import TrainingArgs
    from trainer.utils import FlexibleArgumentParser
    parser = FlexibleArgumentParser()
    parser = TrainingArgs.add_cli_args(parser)
    parser = TrainerArgs.add_cli_args(parser)
    args = parser.parse_args()
    args.dit_cpu_offload = False
    main(args)
