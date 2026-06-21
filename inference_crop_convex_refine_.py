import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Amodal3R'))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['SPCONV_ALGO'] = 'native'

import argparse

import numpy as np
from amodal3r.pipelines import Amodal3RImageTo3DPipeline

from keyframe.decorators import load_mask, load_image
from keyframe.preprocessing import preprocess_frame
from keyframe.pipeline_io import filter_candidates, extract_features, save_outputs
from keyframe.selection import remove_outlier_views, select_frames_fps, select_frames_rotation

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _run_batch(args, pipeline):
    ALL_OCCLUDE_METHODS = ["convex"]
    ALL_BLACK_BG        = [False, True]
    ALL_OUTLIER_SIGMAS  = [None, 1.5]
    n_views_list = args.n_views if args.n_views else [4, 6, 8]

    seq_dirs = sorted(
        os.path.join(args.train_dir, d)
        for d in os.listdir(args.train_dir)
        if os.path.isdir(os.path.join(args.train_dir, d))
    )

    fps_per_nv   = len(ALL_OUTLIER_SIGMAS) * len(ALL_OCCLUDE_METHODS) * len(ALL_BLACK_BG)
    runs_per_seq = len(n_views_list) * fps_per_nv
    total        = len(seq_dirs) * runs_per_seq
    print(f"Sequences : {len(seq_dirs)}")
    print(f"n_views   : {n_views_list}")
    print(f"Total runs: {total}")
    print("=" * 60)

    run = skipped = failed = 0

    for seq_dir in seq_dirs:
        seq_name  = os.path.basename(seq_dir)
        image_dir = os.path.join(seq_dir, "rgb")
        mask_dir  = os.path.join(seq_dir, "masks")
        print(f"\n{'='*60}\nSequence: {seq_name}")

        # 시퀀스 간 lru_cache 초기화 (메모리 누수 방지)
        load_mask.cache_clear()
        load_image.cache_clear()

        all_frames = sorted(os.listdir(image_dir))
        candidates = filter_candidates(
            all_frames, image_dir, mask_dir,
            target_val=50, occlude_val=150,
            blur_threshold=args.blur_threshold,
            visible_threshold=args.visible_threshold,
        )
        candidate_names = [c[0] for c in candidates]
        visible_ratios  = np.array([c[1] for c in candidates])
        print(f"  Candidates: {len(candidates)} / {len(all_frames)}")

        if not candidate_names:
            print("  [WARN] 후보 없음, 시퀀스 건너뜀")
            run += runs_per_seq
            continue

        fps_features_base = extract_features(pipeline, candidate_names, image_dir)

        for n_views in n_views_list:
            if len(candidates) < n_views:
                print(f"  [WARN] n_views={n_views}: 후보 부족, 건너뜀")
                run += fps_per_nv
                continue

            fps_cache: dict = {}
            for sigma in ALL_OUTLIER_SIGMAS:
                feats  = fps_features_base.copy()
                cnames = list(candidate_names)
                vrats  = visible_ratios.copy()
                if sigma is not None:
                    feats, cnames, vrats = remove_outlier_views(feats, cnames, vrats, n_views, sigma=sigma)
                if len(cnames) < n_views:
                    fps_cache[sigma] = []
                    continue
                idx = select_frames_fps(feats, vrats, cnames, n_views, args.visible_threshold)
                fps_cache[sigma] = [cnames[i] for i in idx]

            for sigma in ALL_OUTLIER_SIGMAS:
                selected_fnames = fps_cache.get(sigma, [])
                for occlude_method in ALL_OCCLUDE_METHODS:
                    for black_bg in ALL_BLACK_BG:
                        run += 1

                        out_name = "ch_fps"
                        if occlude_method == "dilate":
                            out_name += f"_dilate{args.dilate_px}"
                        elif occlude_method == "flood":
                            out_name += "_flood"
                        if black_bg:
                            out_name += "_blackbg"
                        if sigma is not None:
                            out_name += f"_os{sigma}"
                        out_dir = os.path.join("./output", seq_name, f"n{n_views}", out_name)

                        if os.path.exists(os.path.join(out_dir, "mesh.ply")):
                            skipped += 1
                            print(f"  [{run}/{total}] SKIP  {seq_name}/n{n_views}/{out_name}")
                            continue

                        if not selected_fnames:
                            failed += 1
                            continue

                        print(f"  [{run}/{total}] {seq_name}/n{n_views}/{out_name}")

                        images, masks, used_ids = [], [], []
                        for fname in selected_fnames:
                            fid = os.path.splitext(fname)[0]
                            img, msk = preprocess_frame(
                                os.path.join(image_dir, fname),
                                os.path.join(mask_dir, fid + ".png"),
                                margin=args.margin, output_size=args.output_size,
                                black_bg=black_bg, occlude_method=occlude_method,
                                dilate_px=args.dilate_px,
                            )
                            if img is None:
                                continue
                            images.append(img)
                            masks.append(msk)
                            used_ids.append(fid)

                        if not images:
                            print("    -> 유효 프레임 없음, 건너뜀")
                            failed += 1
                            continue

                        try:
                            save_outputs(pipeline, images, masks, out_dir,
                                         seq_dir, used_ids, "fps", n_views,
                                         occlude_method, black_bg, sigma, args)
                            print("    -> OK")
                        except Exception as e:
                            print(f"    -> FAILED: {e}")
                            failed += 1

    print("=" * 60)
    print(f"Done.  Total={total}  Skipped={skipped}  Failed={failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    seq_group = parser.add_mutually_exclusive_group(required=True)
    seq_group.add_argument("--seq_dir",   type=str)
    seq_group.add_argument("--train_dir", type=str)

    parser.add_argument("--margin",      type=int,   default=40)
    parser.add_argument("--output_size", type=int,   default=512)

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument("--frames",      type=int, nargs="+")
    mode.add_argument("--auto_select", action="store_true")

    parser.add_argument("--select_method",    type=str,   choices=["fps", "rotation"], default="fps")
    parser.add_argument("--n_views",          type=int,   nargs="+", default=None)
    parser.add_argument("--blur_threshold",   type=float, default=100.0)
    parser.add_argument("--visible_threshold",type=float, default=0.4)
    parser.add_argument("--black_bg",         action="store_true")
    parser.add_argument("--no_crop",          action="store_true")
    parser.add_argument("--occlude_method",   type=str,   choices=["convex","dilate","flood"], default="convex")
    parser.add_argument("--dilate_px",        type=int,   default=10)
    parser.add_argument("--outlier_sigma",    type=float, nargs="?", const=1.5, default=None)
    args = parser.parse_args()

    print("Loading pipeline...")
    pipeline = Amodal3RImageTo3DPipeline.from_pretrained("Sm0kyWu/Amodal3R")
    pipeline.cuda()

    if args.train_dir:
        _run_batch(args, pipeline)

    else:
        if not args.auto_select and not args.frames:
            parser.error("single mode에서는 --frames 또는 --auto_select 가 필요합니다.")

        n_views   = args.n_views[0] if args.n_views else 4
        seq_name  = os.path.basename(os.path.normpath(args.seq_dir))
        image_dir = os.path.join(args.seq_dir, "rgb")
        mask_dir  = os.path.join(args.seq_dir, "masks")

        if args.auto_select:
            output_name = f"ch_{args.select_method}"
        else:
            output_name = f"ch_manual_{len(args.frames)}f"
        if args.occlude_method == "dilate":
            output_name += f"_dilate{args.dilate_px}"
        elif args.occlude_method == "flood":
            output_name += "_flood"
        if args.no_crop:
            output_name += "_nocrop"
        if args.black_bg:
            output_name += "_blackbg"
        if args.auto_select and args.select_method == "fps" and args.outlier_sigma is not None:
            output_name += f"_os{args.outlier_sigma}"

        output_dir = (os.path.join("./output", seq_name, f"n{n_views}", output_name)
                      if args.auto_select
                      else os.path.join("./output", seq_name, output_name))
        os.makedirs(output_dir, exist_ok=True)

        if args.auto_select:
            all_frames = sorted(os.listdir(image_dir))
            candidates = filter_candidates(
                all_frames, image_dir, mask_dir,
                target_val=50, occlude_val=150,
                blur_threshold=args.blur_threshold,
                visible_threshold=args.visible_threshold,
            )
            candidate_names = [c[0] for c in candidates]
            visible_ratios  = np.array([c[1] for c in candidates])
            print(f"  {len(candidates)} / {len(all_frames)} frames passed filtering")
            assert len(candidates) >= n_views, "후보 프레임이 너무 적습니다. 필터 조건을 낮추세요."

            if args.select_method == "fps":
                features = extract_features(pipeline, candidate_names, image_dir)
                if args.outlier_sigma is not None:
                    features, candidate_names, visible_ratios = remove_outlier_views(
                        features, candidate_names, visible_ratios, n_views, sigma=args.outlier_sigma)
                selected_idx    = select_frames_fps(features, visible_ratios, candidate_names,
                                                    n_views, args.visible_threshold)
                selected_fnames = [candidate_names[i] for i in selected_idx]
            else:
                selected_fnames = select_frames_rotation(candidate_names, args.seq_dir, n_views)

            print(f"Selected frames: {selected_fnames}")
            frame_ids  = [os.path.splitext(f)[0] for f in selected_fnames]
            rgb_paths  = [os.path.join(image_dir, f) for f in selected_fnames]
            mask_paths = [os.path.join(mask_dir, os.path.splitext(f)[0] + ".png")
                          for f in selected_fnames]
        else:
            frame_ids  = [f"{f:04d}" for f in args.frames]
            rgb_paths  = [os.path.join(image_dir, f"{f:04d}.jpg") for f in args.frames]
            mask_paths = [os.path.join(mask_dir,  f"{f:04d}.png") for f in args.frames]

        images, masks, used_ids = [], [], []
        for fid, rp, mp in zip(frame_ids, rgb_paths, mask_paths):
            img, msk = preprocess_frame(rp, mp, margin=args.margin, output_size=args.output_size,
                                        black_bg=args.black_bg, occlude_method=args.occlude_method,
                                        dilate_px=args.dilate_px, no_crop=args.no_crop)
            if img is None:
                print(f"[WARN] frame {fid}: no object pixels, skipping")
                continue
            images.append(img)
            masks.append(msk)
            used_ids.append(fid)

        save_outputs(pipeline, images, masks, output_dir,
                     args.seq_dir, used_ids, args.select_method, n_views,
                     args.occlude_method, args.black_bg, args.outlier_sigma, args)
