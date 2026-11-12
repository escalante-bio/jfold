"""Generate `docs/models/README.md` from the released models' `model.json` files.

    uv run python tools/model_table.py
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download

from jfold import pretrained

OUT = Path(__file__).resolve().parents[1] / "docs" / "models" / "README.md"
MODELS = {
    "Boltz-2": "boltz2",
    "ProMERA": "promera",
    "Protenix": "protenix_base",
    "Mol-1": "mol1_base",
    "OpenDDE": "opendde_v1",
    "OpenFold3": "openfold3",
    "ESMFold2": "esmfold2",
    "BioEmu2": "bioemu2",
    "AtlasFold-M": "atlasfold_m",
    "AtlasFold": "atlasfold",
}
def is_module(node) -> bool:
    return isinstance(node, dict) and "module" in node


def name(node) -> str:
    return node["module"].rsplit(":", 1)[1]


def first_array(node):
    if isinstance(node, dict):
        if "array" in node:
            return node
        for value in node.get("fields", node).values():
            if (found := first_array(value)) is not None:
                return found
    return None


def depth(stack) -> int:
    """Return the number of layers in a stack, counting its optional `last` layer."""
    fields = stack["fields"]
    return first_array(fields["layers"])["shape"][0] + is_module(fields.get("last"))


def layers(stack) -> str:
    """Describe a pair stack's depth and layer type, such as "48 Pairformer"."""
    if name(stack) == "Mol1Pairformer":
        return " + ".join(layers(stack["fields"][k]) for k in ("pairmixer", "pairformer"))
    if name(stack) == "AtlasFoldPairformer":
        f = stack["fields"]
        return f"{depth(f['pair_only'])} pair-only + {layers(f['full'])}"
    kind = name(stack["fields"]["layers"]).removesuffix("Layer")
    return f"{depth(stack)} {kind}"


def msa(node) -> tuple[str, str]:
    """Describe the MSA module's layers and how many MSA rows each pass uses."""
    if not is_module(node):
        return "—", "—"
    wrapped = name(node) in ("ResidualMSA", "TrackedMSA")
    module = node["fields"]["module"] if wrapped else node
    f = module["fields"]
    if f["max_rows"] is None:
        rows = "all"
    else:
        rows = str(f["max_rows"]) + ", query kept" * f["keep_query"]
        rows += ", random depth" * (f["min_rows"] is not None)
    residual = ", residual" * (name(node) == "ResidualMSA")
    return f"{depth(module)} layers{residual}", rows


def trunk(node) -> tuple[dict, list[str]]:
    """Return the trunk's template, MSA and pair stack modules, and its other parts."""
    f = node["fields"]
    match name(node):
        case "Trunk":
            return {"templates": f["templates"], "msa": f["msa"], "pairformer": f["pairformer"]}, []
        case "ConditionedTrunk":
            parts, _ = trunk(f["trunk"])
            return parts, ["unconditional branch"] * is_module(f["unconditional"])
        case "ParcaeTrunk":
            other = ["Parcae recurrence", f"coda ({layers(f['coda'])})"]
            if is_module(f["lm_encoder"]):
                other.insert(0, f"LM encoder ({layers(f['lm_encoder'])})")
            return {"templates": None, "msa": f["msa"], "pairformer": f["trunk"]}, other
        case "RecyclingPairTrunk":
            return {"templates": None, "msa": f["msa"], "pairformer": f["trunk"]}, []
        case "AtlasFoldTrunk":
            lm = f["language_model"]["fields"]["stack"]
            other = [
                f"AtlasLM every pass ({f['mlm_rate']:.0%} masked)",
                f"LM pair stack ({first_array(lm)['shape'][0]} layers)",
            ]
            return {"templates": f["templates"], "msa": None, "pairformer": f["pairformer"]}, other
    raise ValueError(f"unknown trunk {name(node)}")


def diffusion(node) -> tuple[str, list[str]]:
    f = node["fields"]
    match name(node):
        case "StructuralDiffusion":
            inner, _ = diffusion(f["diffusion"])
            refiner = f" + {layers(f['refiner'])} refiner" * is_module(f["refiner"])
            return f"{inner}, on structural tokens{refiner}", []
        case "DualStreamModule":
            t = f["transformer"]["fields"]
            return f"dual-stream: {depth(t['tokens'])} token + {depth(t['atoms'])} atom layers", []
    if node["module"].startswith("jfold.models.atlasfold:"):
        return f"{depth(f['transformer'])} layers, on atom14 slots", []
    return f"{depth(f['transformer'])} layers", []


def confidence(node) -> tuple[str, list[str]]:
    if not is_module(node):
        return "—", []
    if name(node) == "MultimerConfidence":
        return confidence(node["fields"]["module"])
    f = node["fields"]
    if name(node) == "MonomerConfidence":
        stacks = f"{layers(f['single_stack'])} + {layers(f['pair_stack'])} (pair only)"
        return f"{stacks}, per-token pLDDT", []
    per = "atom" if name(f["plddt"]) == "AtomHead" else "token"
    other = ["confidence pair inputs"] * is_module(f.get("pair_inputs"))
    return f"{layers(f['stack'])}, per-{per} pLDDT", other


def row(model: dict) -> dict[str, str]:
    f = model["fields"]
    parts, other = trunk(f["trunk"])
    msa_layers, msa_rows = msa(parts["msa"])
    diffusion_cell, more = diffusion(f["diffusion"])
    confidence_cell, extra = confidence(f.get("confidence"))
    trunk_init = name(f["trunk_init"])
    if is_module(f["trunk_init"]["fields"].get("language_model")):
        trunk_init += " (+ language model)"
    templates = parts["templates"]
    return {
        "input embedder": name(f["input_embedder"]),
        "trunk init": trunk_init,
        "trunk": name(f["trunk"]),
        "templates": f"{depth(templates['fields']['stack'])} layers" if is_module(templates) else "—",
        "MSA": msa_layers,
        "MSA rows / pass": msa_rows,
        "pair stack": layers(parts["pairformer"]),
        "diffusion": diffusion_cell,
        "confidence": confidence_cell,
        "other": ", ".join(other + more + extra) or "—",
    }


def main():
    with TemporaryDirectory() as tmp:
        patterns = [f"{n}/model.json" for n in MODELS.values()]
        root = Path(snapshot_download(pretrained.HUB_REPO, allow_patterns=patterns, local_dir=tmp))
        rows = {
            label: row(json.loads((root / n / "model.json").read_text())["model"])
            for label, n in MODELS.items()
        }
    parts = list(next(iter(rows.values())))
    lines = [
        "# Models",
        "",
        "![StructureModel](structure_model.svg)",
        "",
        "| | " + " | ".join(rows) + " |",
        "|---|" + "---|" * len(rows),
        *(f"| {part} | " + " | ".join(r[part] for r in rows.values()) + " |" for part in parts),
        "",
        f"<sub>Generated by `tools/model_table.py` from `{pretrained.HUB_REPO}`.</sub>",
        "",
    ]
    OUT.write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
