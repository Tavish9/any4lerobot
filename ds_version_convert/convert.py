from pathlib import Path

from .lerobot_dataset_version_converter import (
    DatasetContext,
    build_default_converter,
)

def convert_local_dataset(repo_name: str, source: Path, workspace: Path, target_version: str):
    converter = build_default_converter()

    context = DatasetContext(
        repo_id=repo_name,
        source_root=source,
        output_root=workspace / "converted_outputs",
    )

    plan = converter.convert(context=context, target_version=target_version)

    print(f"Executed {len(plan)} step(s):")
    for step in plan:
        print(f"  {step.description} (supports_inplace={step.supports_inplace})")

    print("Final dataset at:", context.current_root)

if __name__ == "__main__":
    convert_local_dataset(
        repo_name="yihao-brain-bot/test_data_converter",             
        source=Path("/home/yihao/.cache/huggingface/lerobot/yihao-brain-bot/test_data_converter"),        
        workspace=Path("./.lerobot_converts"), 
        target_version="v2.0",
    )
