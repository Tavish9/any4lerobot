import argparse
import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tensorflow_datasets.core.file_adapters import FileFormat
from tensorflow_datasets.core.utils.lazy_imports_utils import apache_beam as beam
from tensorflow_datasets.rlds import rlds_base

os.environ["NO_GCE_CHECK"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tfds.core.utils.gcs_utils._is_gcs_disabled = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_config_from_features(features, encoding_format, **kwargs):
    action_info = {
        **{
            "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
                shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
            )
            for k, v in features.items()
            if "action" in k  # for compatibility with actions.action_key and action
        },
    }
    action_info = action_info if len(action_info) > 1 else action_info.popitem()[1]
    return dict(
        observation_info={
            **{
                k.split(".")[-1]: tfds.features.Image(
                    shape=v["shape"], dtype=np.uint8, encoding_format=encoding_format, doc=v["names"]
                )
                for k, v in features.items()
                if "observation.image" in k and "depth" not in k
            },
            **{
                k.split(".")[-1]: tfds.features.Tensor(shape=v["shape"][:-1], dtype=np.float32, doc=v["names"])
                for k, v in features.items()
                if "observation.image" in k and "depth" in k
            },
            **{
                "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                )
                for k, v in features.items()
                if "observation.state" in k  # for compatibility with observation.states.state_key and observation.state
            },
        },
        action_info=action_info,
        step_metadata_info={
            "language_instruction": tfds.features.Text(),
        },
        citation=kwargs.get("citation", ""),
        homepage=kwargs.get("homepage", ""),
        overall_description=kwargs.get("overall_description", ""),
        description=kwargs.get("description", ""),
    )


def parse_step(data_item):
    observation_info = {
        **{
            # lerobot image is (C, H, W) and in range [0, 1]
            k.split(".")[-1]: np.array(v * 255, dtype=np.uint8).transpose(1, 2, 0)
            for k, v in data_item.items()
            if "observation.image" in k and "depth" not in k
        },
        **{
            # lerobot depth is (1, H, W) and in range [0, 1]
            k.split(".")[-1]: v.float().squeeze()
            for k, v in data_item.items()
            if "observation.image" in k and "depth" in k
        },
        **{"_".join(k.split(".")[2:]) or k.split(".")[-1]: v for k, v in data_item.items() if "observation.state" in k},
    }
    action_info = {
        **{"_".join(k.split(".")[2:]) or k.split(".")[-1]: v for k, v in data_item.items() if "action" in k},
    }
    action_info = action_info if len(action_info) > 1 else action_info.popitem()[1]

    return observation_info, action_info, data_item["task"]


class DatasetBuilder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    def __init__(self, raw_dir, name, dataset_config, enable_beam, *, file_format=None, **kwargs):
        self.name = name
        self.VERSION = kwargs["version"]
        self.raw_dir = raw_dir
        self.dataset_config = dataset_config
        self.enable_beam = enable_beam
        self.__module__ = "lerobot2rlds"
        super().__init__(file_format=file_format, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return rlds_base.build_info(
            rlds_base.DatasetConfig(
                name=self.name,
                **self.dataset_config,
            ),
            self,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dl_manager._download_dir.rmtree(missing_ok=True)
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self):
        """Yields examples."""

        def _generate_examples_beam(episode_index, raw_dir):
            episode = []
            dataset = LeRobotDataset("", raw_dir, episodes=[episode_index])
            logging.info(f"processing episode {episode_index}")
            for data_item in dataset:
                observation_info, action_info, language_instruction = parse_step(data_item)
                episode.append(
                    {
                        "observation": observation_info,
                        "action": action_info,
                        "language_instruction": language_instruction,
                        "is_first": data_item["frame_index"].item() == 0,
                        "is_last": data_item["frame_index"].item()
                        == dataset.meta.episodes[episode_index]["length"] - 1,
                        "is_terminal": data_item["frame_index"].item()
                        == dataset.meta.episodes[episode_index]["length"] - 1,
                    }
                )
            return episode_index, {"steps": episode}

        def _generate_examples_regular():
            dataset = LeRobotDataset("", self.raw_dir)
            episode = []
            current_episode_index = 0
            for data_item in dataset:
                if data_item["episode_index"] != current_episode_index:
                    episode[-1]["is_last"] = True
                    episode[-1]["is_terminal"] = True
                    yield f"{current_episode_index}", {"steps": episode}
                    current_episode_index = data_item["episode_index"]
                    episode.clear()

                observation_info, action_info, language_instruction = parse_step(data_item)
                episode.append(
                    {
                        "observation": observation_info,
                        "action": action_info,
                        "language_instruction": language_instruction,
                        "is_first": data_item["frame_index"].item() == 0,
                        "is_last": False,
                        "is_terminal": False,
                    }
                )
            episode[-1]["is_last"] = True
            episode[-1]["is_terminal"] = True
            yield f"{current_episode_index}", {"steps": episode}

        if self.enable_beam:
            metadata = LeRobotDatasetMetadata("", self.raw_dir)
            return beam.Create(list(metadata.episodes.keys())) | beam.Map(
                partial(_generate_examples_beam, raw_dir=self.raw_dir)
            )
        else:
            # NOTE: we should return a generator, not yield
            return _generate_examples_regular()


def main(src_dir, output_dir, task_name, version, encoding_format, enable_beam, **kwargs):
    raw_dataset_meta = LeRobotDatasetMetadata("", root=src_dir)

    dataset_config = generate_config_from_features(raw_dataset_meta.features, encoding_format, **kwargs)

    dataset_builder = DatasetBuilder(
        raw_dir=src_dir,
        name=task_name,
        data_dir=output_dir,
        version=version,
        dataset_config=dataset_config,
        enable_beam=enable_beam,
        file_format=FileFormat.TFRECORD,
    )

    if enable_beam:
        logging.warning("beam processing is enabled. Some episodes might be lost, a bug with apache beam.")
        logging.warning("disable beam processing if your dataset is small or you want to save all episodes.")
        from apache_beam.options.pipeline_options import PipelineOptions
        from apache_beam.runners import create_runner

        if "threading" in kwargs["beam_run_mode"]:
            logging.warning("multi_threading might have issues when sharding and saving.")
            logging.warning("recommend using multi_processing instead.")

        beam_options = PipelineOptions(
            direct_running_mode=kwargs["beam_run_mode"],
            direct_num_workers=kwargs["beam_num_workers"],
        )
        beam_runner = create_runner("DirectRunner")
    else:
        beam_options = None
        beam_runner = None

    dataset_builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            try_download_gcs=False,
            verify_ssl=False,
            beam_options=beam_options,
            beam_runner=beam_runner,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, help="Path to the local lerobot dataset.")
    parser.add_argument("--output-dir", type=Path, help="Path to the output directory.")
    parser.add_argument("--task-name", type=str, help="Task name.")
    parser.add_argument("--enable-beam", action="store_true", help="Enable beam processing.")
    parser.add_argument("--beam-run-mode", choices=["multi_threading", "multi_processing"], default="multi_processing")
    parser.add_argument("--beam-num-workers", type=int, default=5)
    parser.add_argument("--encoding-format", type=str, choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--version", type=str, help="x.y.z", default="0.1.0")
    parser.add_argument("--citation", type=str, help="Citation.", default="")
    parser.add_argument("--homepage", type=str, help="Homepage.", default="")
    parser.add_argument("--overall-description", type=str, help="Overall description.", default="")
    parser.add_argument("--description", type=str, help="Description.", default="")
    args = parser.parse_args()

    main(**vars(args))
