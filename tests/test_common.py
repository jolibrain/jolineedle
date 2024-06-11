import shutil
import os
from urllib.request import urlretrieve

TOY_LARD = {
    "url": "https://www.deepdetect.com/dd/datasets/toylard.tar.gz",
    "subdir": "toylard",
}


def download_dataset(work_dir, dset):
    dset_url = dset["url"]
    dest_filename = dset_url.split("/")[-1]
    tar_path = os.path.join(work_dir, dest_filename)

    if not os.path.exists(tar_path):
        print("Downloading test dataset: %s" % dset_url)
        urlretrieve(dset_url, tar_path)

    # unzip
    dest_dir = os.path.join(work_dir, dset["subdir"])
    if not os.path.exists(dest_dir):
        print("Unpacking test dataset: %s" % tar_path)
        shutil.unpack_archive(tar_path, work_dir)

    assert os.path.exists(dest_dir), "dir %s does not exist after unpacking" % dest_dir
    print("Dataset %s is ready" % dest_dir)
