from fastai import vision
from fastai.vision.data import ImageImageList, ImageDataBunch, imagenet_stats


def get_colorize_data(
        sz: int,
        bs: int,
        crappy_path: vision.Path,
        good_path: vision.Path,
        random_seed: int = None,
        keep_pct: float = 1.0,
        num_workers: int = 8,
        stats: tuple = imagenet_stats,
        xtra_tfms=None,
) -> ImageDataBunch:
    if xtra_tfms is None:
        xtra_tfms = []
    data = ImageImageList.from_folder(
        crappy_path, ignore_empty=True
    ).split_none(
    ).label_from_func(
        lambda x: good_path / x.relative_to(crappy_path)
    ).transform(
        vision.transform.get_transforms(
            max_zoom=1.2, max_lighting=0.5, max_warp=0.25, xtra_tfms=xtra_tfms
        ),
        size=sz,
        tfm_y=True,
    ).databunch(
        bs=bs, num_workers=num_workers, no_check=True
    ).normalize(
        stats, do_y=True
    )
    data.c = 3
    return data


def get_dummy_databunch(temp_dir) -> ImageDataBunch:
    path = vision.Path(temp_dir)
    return get_colorize_data(
        sz=1, bs=1, crappy_path=path, good_path=path, keep_pct=0.001
    )
