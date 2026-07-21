#!/usr/bin/env python3
import argparse
import copy
import csv
import glob
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def natural_key(s: str):
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", s)]


def load_yaml(path: Path) -> dict:
    with path.open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {path}")
    return data


def dump_yaml(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            width=120,
        )


def list_meshes(mesh_dir: Path, pattern: str) -> List[Path]:
    if not mesh_dir.exists() or not mesh_dir.is_dir():
        raise FileNotFoundError(f"Mesh directory not found or not a directory: {mesh_dir}")
    paths = [Path(p).resolve() for p in glob.glob(str(mesh_dir / pattern))]
    paths = [p for p in paths if p.is_file()]
    paths.sort(key=lambda p: natural_key(p.name))
    return paths


def list_experiments(exp_dir: Path, pattern: str) -> List[Path]:
    if not exp_dir.exists() or not exp_dir.is_dir():
        raise FileNotFoundError(f"Experiments directory not found or not a directory: {exp_dir}")
    paths = [Path(p).resolve() for p in glob.glob(str(exp_dir / pattern))]
    paths = [p for p in paths if p.is_file()]
    paths.sort(key=lambda p: natural_key(p.name))
    return paths


def deep_merge(base: Dict[str, Any], patch: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursive merge:
      - dict + dict => recurse
      - otherwise patch overwrites base
    """
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_mods(path: Path) -> List[dict]:
    data = load_yaml(path)
    mods = data.get("mods")
    if not isinstance(mods, list) or len(mods) == 0:
        raise ValueError("mods YAML must contain non-empty list key: 'mods'")

    normalized = []
    for i, item in enumerate(mods):
        if not isinstance(item, dict):
            raise ValueError(f"mods[{i}] must be a dict")
        name = str(item.get("name", f"mod_{i}"))
        patch = item.get("patch", {})
        if not isinstance(patch, dict):
            raise ValueError(f"mods[{i}].patch must be a dict")
        normalized.append({"name": name, "patch": patch})
    return normalized


def build_triples(
    mods: List[dict],
    meshes: List[Path],
    experiments: List[str],
    order: str,
):
    """
    Returns list of tuples:
      (mod_idx, mesh_idx, exp_idx, mod, mesh_path, experiment_ref)
    """
    triples = []

    if order == "mod_experiment_mesh":
        for mod_idx, mod in enumerate(mods):
            for exp_idx, exp in enumerate(experiments):
                for mesh_idx, mesh in enumerate(meshes):
                    triples.append((mod_idx, mesh_idx, exp_idx, mod, mesh, exp))

    elif order == "experiment_mod_mesh":
        for exp_idx, exp in enumerate(experiments):
            for mod_idx, mod in enumerate(mods):
                for mesh_idx, mesh in enumerate(meshes):
                    triples.append((mod_idx, mesh_idx, exp_idx, mod, mesh, exp))

    elif order == "mesh_mod_experiment":
        for mesh_idx, mesh in enumerate(meshes):
            for mod_idx, mod in enumerate(mods):
                for exp_idx, exp in enumerate(experiments):
                    triples.append((mod_idx, mesh_idx, exp_idx, mod, mesh, exp))
    else:
        raise ValueError(f"Unknown order: {order}")

    return triples


def main():
    ap = argparse.ArgumentParser(
        description="Generate exp_XXXX.yaml from baseline + per-mod patches over meshes and experiments."
    )
    ap.add_argument("--baseline", required=True, help="Path to baseline YAML.")
    ap.add_argument("--mods_yaml", required=True, help="Path to mods YAML with list key 'mods'.")
    ap.add_argument("--mesh_dir", required=True, help="Directory containing mesh files.")
    ap.add_argument("--mesh_pattern", default="*.pth", help='Glob pattern for meshes (default: "*.pth").')
    ap.add_argument("--out_dir", required=True, help="Output directory for exp configs.")

    # Experiment selection: either single fixed experiment or folder of experiments
    ap.add_argument("--experiment", default=None,
                    help="Single experiment value to set in every generated config.")
    ap.add_argument("--experiments_dir", default=None,
                    help="Directory containing multiple experiment YAMLs to assign across generated configs.")
    ap.add_argument("--experiments_pattern", default="*.yaml",
                    help='Glob pattern in experiments_dir (default: "*.yaml").')
    ap.add_argument("--use_absolute_experiment_paths", action="store_true",
                    help="When using --experiments_dir, store absolute paths in cfg['experiment'].")

    ap.add_argument("--prefix", default="exp_", help='Filename prefix (default: "exp_").')
    ap.add_argument("--pad", type=int, default=4, help="Zero-pad width for index (default: 4).")
    ap.add_argument("--start_index", type=int, default=0, help="Starting experiment index (default: 0).")
    ap.add_argument(
        "--order",
        choices=[
            "mod_mesh", "mesh_mod",                    # single-experiment mode
            "mod_experiment_mesh",                     # multi-experiment mode
            "experiment_mod_mesh",
            "mesh_mod_experiment",
        ],
        default="mod_mesh",
        help="Generation order. Use *_experiment_* variants when --experiments_dir is set."
    )
    ap.add_argument("--dry_run", action="store_true", help="Print plan only; do not write files.")
    args = ap.parse_args()

    # Validate mutually exclusive experiment inputs
    if args.experiment is not None and args.experiments_dir is not None:
        print("ERROR: pass only one of --experiment or --experiments_dir", file=sys.stderr)
        sys.exit(2)

    if args.experiment is None and args.experiments_dir is None:
        print("ERROR: you must pass one of --experiment or --experiments_dir", file=sys.stderr)
        sys.exit(2)

    baseline_path = Path(args.baseline).resolve()
    mods_path = Path(args.mods_yaml).resolve()
    mesh_dir = Path(args.mesh_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    baseline = load_yaml(baseline_path)
    mods = load_mods(mods_path)
    meshes = list_meshes(mesh_dir, args.mesh_pattern)

    if not meshes:
        print(f"No mesh files matched {args.mesh_pattern} in {mesh_dir}", file=sys.stderr)
        sys.exit(2)

    # Build experiment refs
    experiment_refs: List[str] = []
    if args.experiment is not None:
        experiment_refs = [args.experiment]
    else:
        exp_dir = Path(args.experiments_dir).resolve()
        exp_files = list_experiments(exp_dir, args.experiments_pattern)
        if not exp_files:
            print(f"No experiment files matched {args.experiments_pattern} in {exp_dir}", file=sys.stderr)
            sys.exit(2)

        if args.use_absolute_experiment_paths:
            experiment_refs = [str(p.resolve()) for p in exp_files]
        else:
            experiment_refs = [str(p) for p in exp_files]

    manifest = []

    if len(experiment_refs) == 1:
        # single experiment mode: keep old pair logic
        pairs = []
        if args.order == "mod_mesh":
            for mod_idx, mod in enumerate(mods):
                for mesh_idx, mesh in enumerate(meshes):
                    pairs.append((mod_idx, mesh_idx, mod, mesh, experiment_refs[0]))
        elif args.order == "mesh_mod":
            for mesh_idx, mesh in enumerate(meshes):
                for mod_idx, mod in enumerate(mods):
                    pairs.append((mod_idx, mesh_idx, mod, mesh, experiment_refs[0]))
        elif args.order == "mod_experiment_mesh":
            for mod_idx, mod in enumerate(mods):
                for mesh_idx, mesh in enumerate(meshes):
                    pairs.append((mod_idx, mesh_idx, mod, mesh, experiment_refs[0]))
        elif args.order == "experiment_mod_mesh":
            for mod_idx, mod in enumerate(mods):
                for mesh_idx, mesh in enumerate(meshes):
                    pairs.append((mod_idx, mesh_idx, mod, mesh, experiment_refs[0]))
        elif args.order == "mesh_mod_experiment":
            for mesh_idx, mesh in enumerate(meshes):
                for mod_idx, mod in enumerate(mods):
                    pairs.append((mod_idx, mesh_idx, mod, mesh, experiment_refs[0]))
        else:
            raise ValueError(f"Unsupported order in single experiment mode: {args.order}")

        for offset, (mod_idx, mesh_idx, mod, mesh_path, exp_ref) in enumerate(pairs):
            exp_idx = args.start_index + offset
            idx = str(exp_idx).zfill(args.pad) if args.pad > 0 else str(exp_idx)
            out_path = out_dir / f"{args.prefix}{idx}_{mod['name']}.yaml"

            cfg = deep_merge(baseline, mod["patch"])
            cfg["target_mesh"] = str(mesh_path)
            cfg["experiment"] = exp_ref

            manifest.append({
                "exp_index": exp_idx,
                "file": str(out_path),
                "mod_index": mod_idx,
                "mod_name": mod["name"],
                "experiment_index": 0,
                "experiment": exp_ref,
                "mesh_index": mesh_idx,
                "mesh_file": str(mesh_path),
            })

            if args.dry_run:
                exp_label = Path(exp_ref).name if "/" in exp_ref else exp_ref
                print(
                    f"[DRY RUN] exp={exp_idx} file={out_path.name} "
                    f"mod={mod['name']} mesh={mesh_path.name} experiment={exp_label}"
                )
            else:
                dump_yaml(cfg, out_path)

    else:
        # multi experiment mode: map compact orders for backward compatibility
        order = args.order
        if order == "mod_mesh":
            order = "mod_experiment_mesh"
        elif order == "mesh_mod":
            order = "mesh_mod_experiment"

        triples = build_triples(mods, meshes, experiment_refs, order)

        for offset, (mod_idx, mesh_idx, exp_i, mod, mesh_path, exp_ref) in enumerate(triples):
            exp_idx = args.start_index + offset
            idx = str(exp_idx).zfill(args.pad) if args.pad > 0 else str(exp_idx)
            out_path = out_dir / f"{args.prefix}{idx}_{mod['name']}.yaml"

            cfg = deep_merge(baseline, mod["patch"])
            cfg["target_mesh"] = str(mesh_path)
            cfg["experiment"] = exp_ref

            manifest.append({
                "exp_index": exp_idx,
                "file": str(out_path),
                "mod_index": mod_idx,
                "mod_name": mod["name"],
                "experiment_index": exp_i,
                "experiment": exp_ref,
                "mesh_index": mesh_idx,
                "mesh_file": str(mesh_path),
            })

            if args.dry_run:
                print(
                    f"[DRY RUN] exp={exp_idx} file={out_path.name} "
                    f"mod={mod['name']} exp={Path(exp_ref).name} mesh={mesh_path.name}"
                )
            else:
                dump_yaml(cfg, out_path)

    total = len(manifest)
    print(f"Mods: {len(mods)}")
    print(f"Experiments: {len(experiment_refs)}")
    print(f"Meshes: {len(meshes)}")
    print(f"Total configs: {total} (mods × experiments × meshes)")

    if not args.dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = out_dir / "manifest.csv"
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "exp_index", "file",
                    "mod_index", "mod_name",
                    "experiment_index", "experiment",
                    "mesh_index", "mesh_file",
                ],
            )
            w.writeheader()
            w.writerows(manifest)
        print(f"Wrote configs to: {out_dir}")
        print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()