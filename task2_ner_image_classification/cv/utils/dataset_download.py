import zipfile
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path="task2_ner_image_classification/.env")

from kaggle.api.kaggle_api_extended import KaggleApi


def download_kaggle_dataset(dataset_slug: str = "alessiocorrado99/animals10", download_dir: str = "task2_ner_image_classification/data"):
    """
    Download and extract a Kaggle dataset using the Kaggle API (no CLI).
    
    :param dataset_slug: Dataset identifier from Kaggle, e.g. 'alessiocorrado99/animals10'
    :param download_dir: Folder where dataset will be stored
    :return: Path to extracted dataset
    """
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"📥 Downloading dataset: {dataset_slug}")
    api.dataset_download_files(dataset_slug, path=str(download_dir), unzip=False)

    zip_files = list(download_dir.glob("*.zip"))
    if not zip_files:
        raise FileNotFoundError("❌ No zip file downloaded. Check dataset slug or Kaggle API key.")

    zip_path = zip_files[0]
    extract_dir = download_dir / dataset_slug.split("/")[-1]

    print(f"📂 Extracting {zip_path} to {extract_dir}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

    print("✅ Download complete.")
    return extract_dir

download_kaggle_dataset()
