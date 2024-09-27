#  HIVE, creates 3D mesh videos.
#  Copyright (C) 2024 Anthony Dickson anthony.dickson9656@gmail.com
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
from argparse import ArgumentParser, Namespace

import cv2
import numpy as np
from compare_image_pair import compare_images
from lpips import LPIPS
from pytorch_msssim import MS_SSIM


def get_arguments() -> Namespace:
    parser = ArgumentParser(
        description="Run image similarity metrics on the 4D-GS outputs on the HyperNeRF dataset."
    )
    parser.add_argument(
        "outputs_path",
        type=str,
        help="The path to the HyperNeRF outputs.",
    )
    parser.add_argument(
        "--save_latex",
        action="store_true",
        help="Whether to save the results in a latex file. "
        "This option assumes image pair folder names are in the format `<dataset>_<config>`.",
    )

    args = parser.parse_args()

    return args


def main(outputs_path: str, save_latex: bool):
    lpips_fn = LPIPS(net="alex")
    ssim_fn = MS_SSIM(data_range=255, size_average=True, channel=1)

    items = os.listdir(os.path.join(outputs_path))
    scenes = list(
        filter(lambda item: os.path.isdir(os.path.join(outputs_path, item)), items)
    )

    scores = {scene: {"ssim": [], "psnr": [], "lpips": []} for scene in scenes}

    for scene in scenes:
        ref_path = os.path.join(outputs_path, scene, "test", "ours_14000", "gt")
        est_path = os.path.join(outputs_path, scene, "test", "ours_14000", "renders")

        if not os.path.isdir(ref_path):
            print(f"Error: Could not find the folder {ref_path}", file=sys.stderr)

        if not os.path.isdir(est_path):
            print(f"Error: Could not find the folder {est_path}", file=sys.stderr)

        ref_filenames = sorted(os.listdir(ref_path))
        est_filenames = sorted(os.listdir(est_path))

        if len(ref_filenames) != len(est_filenames):
            print(
                f"Error: The folders {ref_path} and {est_path} do not have the same number of images.",
                file=sys.stderr,
            )
            return

        for i, (ref_filename, est_filename) in enumerate(
            zip(os.listdir(ref_path), os.listdir(est_path))
        ):
            print(f"\r{i + 1:3d}/{len(ref_filenames):3d}", end="")
            ref_image_path = os.path.join(ref_path, ref_filename)
            est_image_path = os.path.join(est_path, est_filename)
            ref_image = cv2.imread(ref_image_path)
            est_image = cv2.imread(est_image_path)
            ssim_score, psnr_score, lpips_score = compare_images(
                ref_image, est_image, lpips_fn=lpips_fn, ssim_fn=ssim_fn
            )

            scores[scene]["ssim"].append(ssim_score)
            scores[scene]["psnr"].append(psnr_score)
            scores[scene]["lpips"].append(lpips_score)

        print(
            f"\n{scene} - MS-SSIM: {ssim_score:.3f} - PSNR: {psnr_score:.1f} - LPIPS: {lpips_score:.3f}"
        )

    mean_all = {"ssim": [], "psnr": [], "lpips": []}
    mean_by_scene = dict()

    for scene, result in scores.items():
        mean_all["ssim"] += result["ssim"]
        mean_all["psnr"] += result["psnr"]
        mean_all["lpips"] += result["lpips"]

        mean_by_scene[scene] = {
            metric: np.mean(values) for metric, values in result.items()
        }

    mean_all = {metric: np.mean(values) for metric, values in mean_all.items()}

    latex_lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Scene & MS-SSIM & PSNR & LPIPS \\",
        r"\midrule",
    ]

    for scene, metrics in mean_by_scene.items():
        latex_lines.append(
            rf"{scene} & {metrics['ssim']:.3f} & {metrics['psnr']:.1f} & {metrics['lpips']:.3f} \\"
        )

    latex_lines.append(r"\midrule")
    latex_lines.append(
        rf"mean & {mean_all['ssim'] :.3f} & {mean_all['psnr'] :.1f} & {mean_all['lpips'] :.3f} \\"
    )

    latex_lines.append(r"\bottomrule")
    latex_lines.append(r"\end{tabular}")

    if save_latex:
        latex_path = os.path.join(outputs_path, "table.tex")

        with open(latex_path, "w") as f:
            f.write("\n".join(latex_lines))

        print(f"Saved latex table to {latex_path}.")
    else:
        print("\n".join(latex_lines))


if __name__ == "__main__":
    args = get_arguments()
    main(outputs_path=args.outputs_path, save_latex=args.save_latex)
