from pathlib import Path

import pandas as pd


def get_bboxes(df: pd.DataFrame, df_path: Path) -> list:
    bboxes = []
    df_dir = df_path.parent
    for filepath, x_A, y_A, x_B, y_B, x_C, y_C, x_D, y_D in df[
        ["image", "x_A", "y_A", "x_B", "y_B", "x_C", "y_C", "x_D", "y_D"]
    ].values:
        filepath = df_dir / filepath
        bbox = [
            min(x_A, x_B, x_C, x_D),
            min(y_A, y_B, y_C, y_D),
            max(x_A, x_B, x_C, x_D),
            max(y_A, y_B, y_C, y_D),
        ]

        bboxes.append((filepath, bbox))

    return bboxes


def remove_nonexistent(bboxes: list) -> list:
    existent = []
    for filepath, bbox in bboxes:
        if filepath.exists():
            existent.append((filepath, bbox))

    if len(existent) != len(bboxes):
        n_removed = len(bboxes) - len(existent)
        print(
            f"Removed {n_removed} ({n_removed/len(bboxes)*100:.2f}%) non-existent images."
        )

    return existent


def remove_big(bboxes: list, max_size: int) -> list:
    small = []
    for filepath, bbox in bboxes:
        if bbox[2] - bbox[0] < max_size and bbox[3] - bbox[1] < max_size:
            small.append((filepath, bbox))

    if len(small) != len(bboxes):
        n_removed = len(bboxes) - len(small)
        print(
            f"Removed {n_removed} ({n_removed/len(bboxes)*100:.2f}%) big bounding boxes."
        )

    return small


def prepare(bboxes: list, link_file: Path, bboxes_dir: Path) -> None:
    bboxes_dir.mkdir(parents=True, exist_ok=True)

    links = []
    for filepath, bbox in bboxes:
        bbox_path = bboxes_dir / filepath.name
        bbox_path = bbox_path.with_suffix(".txt")
        links.append((filepath.absolute(), bbox_path.absolute()))

        # Add the fictitious class 0
        bbox = ["0"] + [str(x) for x in bbox]
        with open(bbox_path, "w") as f:
            f.write(" ".join(bbox))

    links = [f"{filepath} {bbox_path}" for filepath, bbox_path in links]
    with open(link_file, "w") as f:
        f.write("\n".join(links))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Path to the LARD dataset directory",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Directory {dataset_path} does not exist.")

    TRAIN_PATH = dataset_path / "LARD_train.csv"
    TEST_SYNTH = dataset_path / "LARD_test_synth/LARD_test_synth.csv"
    TEST_REAL_NOMINAL = (
        dataset_path
        / "LARD_test_real/LARD_test_real_nominal_cases/LARD_test_real_nominal_cases.csv"
    )
    TEST_REAL_ADAPTATION = (
        dataset_path
        / "LARD_test_real/LARD_test_real_domain_adaptation/LARD_test_real_domain_adaptation.csv"
    )
    TEST_PATHS = [
        TEST_SYNTH,
        TEST_REAL_NOMINAL,
        TEST_REAL_ADAPTATION,
    ]
    df = pd.read_csv(TRAIN_PATH, sep=";")
    train = get_bboxes(df, TRAIN_PATH)

    test = []
    for test_path in TEST_PATHS:
        df = pd.read_csv(test_path, sep=";")
        test.extend(get_bboxes(df, test_path))

    train = remove_nonexistent(train)
    train = remove_big(train, max_size=448)
    print(f"train: {len(train)}")

    test = remove_nonexistent(test)
    test = remove_big(test, max_size=448)
    print(f"test: {len(test)}")

    prepare(train, dataset_path / "train.txt", dataset_path / "train_bboxes")
    prepare(test, dataset_path / "test.txt", dataset_path / "test_bboxes")
    print("OK")
