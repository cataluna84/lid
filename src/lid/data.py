import pandas as pd
from datasets import load_dataset

from lid.constants import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_FILE,
    LANG_CODES_BLOCK,
    LANG_TO_ISO,
    TASK_DESCRIPTION,
    VALID_OPTIONS,
)


def build_instruct_prompt(text: str) -> str:
    return (
        f"TASK : \n{TASK_DESCRIPTION}\n\n"
        f"LANG CODES : \n{LANG_CODES_BLOCK}\n\n"
        f"INPUT TEXT : \n{text}\n\n"
        f"OUTPUT : \n"
    )


def build_training_sample(text: str, iso_code: str) -> str:
    return build_instruct_prompt(text) + iso_code


def load_lid_dataset(
    dataset_name: str = DEFAULT_DATASET,
    file_path: str | None = DEFAULT_DATASET_FILE,
    token: str | None = None,
    sample_frac: float = 1.0,
    random_state: int = 1024,
) -> pd.DataFrame:
    if file_path:
        dataset = load_dataset(
            "parquet",
            data_files={"train": f"hf://datasets/{dataset_name}/{file_path}"},
            token=token,
        )["train"]
    else:
        dataset = load_dataset(dataset_name, token=token)["train"]

    df = dataset.to_pandas()

    if "ISO-693-3" not in df.columns:
        df["ISO-693-3"] = df["lang"].map(LANG_TO_ISO)

    if sample_frac < 1.0:
        df = (
            df.groupby("ISO-693-3", group_keys=False)
            .sample(frac=sample_frac, random_state=random_state)
            .reset_index(drop=True)
        )

    df["INSTRUCT"] = df["text"].apply(build_instruct_prompt)

    return df


def load_commonlid_dataset(
    token: str | None = None,
    valid_isos: list[str] | None = None,
) -> pd.DataFrame:
    """Load CommonLID test set, filtered to our 67 ISO classes."""
    ds = load_dataset("commoncrawl/CommonLID", split="test", token=token)
    df = ds.to_pandas()

    if valid_isos is None:
        valid_isos = VALID_OPTIONS

    df = df[df["tag"].isin(valid_isos)].reset_index(drop=True)
    df = df.rename(columns={"tag": "ISO-693-3"})
    df["INSTRUCT"] = df["text"].apply(build_instruct_prompt)

    return df
