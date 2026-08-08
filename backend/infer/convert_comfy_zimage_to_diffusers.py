"""Convertit un checkpoint Z-Image au format ComfyUI (.safetensors single-file)
vers le format diffusers (dossier transformer/) attendu par ai-toolkit, en
utilisant la table de mapping OFFICIELLE de ComfyUI.

Le mapping `z_image_to_diffusers` est copié verbatim depuis ComfyUI
(comfy/utils.py) — c'est la table autoritative que ComfyUI emploie pour charger
les modèles diffusers Z-Image. Aucune devinette : renommage des couches groupées
(all_x_embedder.2-1, all_final_layer.2-1) + split de l'attention QKV fusionnée
en to_q/to_k/to_v.

Usage (lancer avec le python d'ai-toolkit pour la validation diffusers) :
  python convert_comfy_zimage_to_diffusers.py <input.safetensors> <official_config.json> [--save <out_dir>]

Sans --save : mode GATE seul (compare les clés/shapes à ZImageTransformer2DModel
sur device 'meta', sans rien écrire).

MÉMOIRE — CE QUE CE FICHIER NE FAIT PLUS
----------------------------------------
Il appelait `safetensors.torch.load_file(comfy_path)`, qui memory-mappe le
conteneur ENTIER. Windows facture ce mapping au commit système à sa CRÉATION,
avant qu'un seul nombre soit lu. Mesuré ici sur le vrai
`z_image_turbo_bf16.safetensors` (11,46 Go) : `load_file` seul fait passer le
commit privé du process de 0,18 Go à 11,67 Go en 0,1 s, résident inchangé —
signature d'une réservation, pas d'une lecture. La construction du state-dict
diffusers montait ensuite à 14,87 Go de commit (les clones q/k/v), et l'écriture
touchait les 11,46 Go restants.

Ce sont des tailles ordinaires ici : les Z-Image sur ce disque vont de 5,7 Go à
16,9 Go, et le gros demande donc ~22 Go de commit pour être CONVERTI. Sur une
machine à 16 Go avec le fichier d'échange par défaut, c'est `OSError 1455` ("le
fichier de pagination est insuffisant") ou une mort par OOM — sur la machine où
le code s'écrit, qui a 106 Go de pagefile, c'est invisible. Même classe de
défaut que celui corrigé dans `fp8_export` le même jour, sur une autre voie.

Donc : plus aucun mapping. `_Reader` cherche l'offset que l'en-tête donne déjà et
lit UN tenseur à la fois ; le plan de sortie (clés, dtypes, shapes, offsets) est
une fonction pure de l'en-tête SOURCE, donc l'en-tête de sortie est écrit avant
la première lecture et chaque tenseur est lu, découpé, écrit, puis relâché. Le
pic tient un tenseur, pas un checkpoint. Le GATE, lui, ne lit plus AUCUN octet de
poids : il ne compare que des shapes, et les shapes sont dans l'en-tête.
"""
import sys
import os
import json
import shutil
import re
import struct

import torch


# ---- ComfyUI comfy/utils.py: z_image_to_diffusers (verbatim, attribution) -----
def z_image_to_diffusers(mmdit_config, output_prefix=""):
    n_layers = mmdit_config.get("n_layers", 0)
    hidden_size = mmdit_config.get("dim", 0)
    n_context_refiner = mmdit_config.get("n_refiner_layers", 2)
    n_noise_refiner = mmdit_config.get("n_refiner_layers", 2)
    key_map = {}

    def add_block_keys(prefix_from, prefix_to, has_adaln=True):
        for end in ("weight", "bias"):
            k = "{}.attention.".format(prefix_from)
            qkv = "{}.attention.qkv.{}".format(prefix_to, end)
            key_map["{}to_q.{}".format(k, end)] = (qkv, (0, 0, hidden_size))
            key_map["{}to_k.{}".format(k, end)] = (qkv, (0, hidden_size, hidden_size))
            key_map["{}to_v.{}".format(k, end)] = (qkv, (0, hidden_size * 2, hidden_size))
        block_map = {
            "attention.norm_q.weight": "attention.q_norm.weight",
            "attention.norm_k.weight": "attention.k_norm.weight",
            "attention.to_out.0.weight": "attention.out.weight",
            "attention.to_out.0.bias": "attention.out.bias",
            "attention_norm1.weight": "attention_norm1.weight",
            "attention_norm2.weight": "attention_norm2.weight",
            "feed_forward.w1.weight": "feed_forward.w1.weight",
            "feed_forward.w2.weight": "feed_forward.w2.weight",
            "feed_forward.w3.weight": "feed_forward.w3.weight",
            "ffn_norm1.weight": "ffn_norm1.weight",
            "ffn_norm2.weight": "ffn_norm2.weight",
        }
        if has_adaln:
            block_map["adaLN_modulation.0.weight"] = "adaLN_modulation.0.weight"
            block_map["adaLN_modulation.0.bias"] = "adaLN_modulation.0.bias"
        for k, v in block_map.items():
            key_map["{}.{}".format(prefix_from, k)] = "{}.{}".format(prefix_to, v)

    for i in range(n_layers):
        add_block_keys("layers.{}".format(i), "{}layers.{}".format(output_prefix, i))
    for i in range(n_context_refiner):
        add_block_keys("context_refiner.{}".format(i), "{}context_refiner.{}".format(output_prefix, i))
    for i in range(n_noise_refiner):
        add_block_keys("noise_refiner.{}".format(i), "{}noise_refiner.{}".format(output_prefix, i))

    MAP_BASIC = [
        ("final_layer.linear.weight", "all_final_layer.2-1.linear.weight"),
        ("final_layer.linear.bias", "all_final_layer.2-1.linear.bias"),
        ("final_layer.adaLN_modulation.1.weight", "all_final_layer.2-1.adaLN_modulation.1.weight"),
        ("final_layer.adaLN_modulation.1.bias", "all_final_layer.2-1.adaLN_modulation.1.bias"),
        ("x_embedder.weight", "all_x_embedder.2-1.weight"),
        ("x_embedder.bias", "all_x_embedder.2-1.bias"),
        ("x_pad_token", "x_pad_token"),
        ("cap_embedder.0.weight", "cap_embedder.0.weight"),
        ("cap_embedder.1.weight", "cap_embedder.1.weight"),
        ("cap_embedder.1.bias", "cap_embedder.1.bias"),
        ("cap_pad_token", "cap_pad_token"),
        ("t_embedder.mlp.0.weight", "t_embedder.mlp.0.weight"),
        ("t_embedder.mlp.0.bias", "t_embedder.mlp.0.bias"),
        ("t_embedder.mlp.2.weight", "t_embedder.mlp.2.weight"),
        ("t_embedder.mlp.2.bias", "t_embedder.mlp.2.bias"),
    ]
    for c, diffusers in MAP_BASIC:
        key_map[diffusers] = "{}{}".format(output_prefix, c)
    return key_map


PREFIX = "model.diffusion_model."

# Un vrai en-tête safetensors est très en dessous ; au-delà, les 8 premiers
# octets ne sont pas une longueur d'en-tête.
_HEADER_LEN_MAX = 512 * 1024 * 1024

_DTYPE_BYTES = {
    'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
    'I16': 2, 'U16': 2, 'F16': 2, 'BF16': 2,
    'I32': 4, 'U32': 4, 'F32': 4,
    'I64': 8, 'U64': 8, 'F64': 8,
}


def _torch_dtype_for(name):
    return {
        'BF16': torch.bfloat16, 'F16': torch.float16, 'F32': torch.float32,
        'F64': torch.float64, 'F8_E4M3': torch.float8_e4m3fn,
        'F8_E5M2': torch.float8_e5m2, 'I64': torch.int64, 'I32': torch.int32,
        'I16': torch.int16, 'I8': torch.int8, 'U8': torch.uint8,
        'BOOL': torch.bool,
    }.get(str(name).upper())


def read_header(path):
    """L'en-tête safetensors (index des tenseurs + `__metadata__`). En-tête SEUL :
    le corps multi-Go n'est jamais touché."""
    try:
        with open(path, 'rb') as fh:
            raw = fh.read(8)
            if len(raw) != 8:
                raise ValueError('fichier trop court pour un conteneur safetensors')
            n = struct.unpack('<Q', raw)[0]
            if n <= 0 or n > _HEADER_LEN_MAX:
                raise ValueError("longueur d'en-tête safetensors implausible")
            blob = fh.read(n)
            if len(blob) != n:
                raise ValueError("en-tête safetensors tronqué")
            obj = json.loads(blob.decode('utf-8'))
    except (OSError, ValueError, UnicodeDecodeError) as e:
        raise RuntimeError(f'fichier .safetensors illisible ({e})') from e
    if not isinstance(obj, dict):
        raise RuntimeError("l'en-tête safetensors n'est pas un objet")
    return obj


def _entries(header):
    return {k: v for k, v in header.items()
            if k != '__metadata__' and isinstance(v, dict)}


class _Reader:
    """Un tenseur à la fois hors d'un .safetensors, par I/O fichier ordinaire.

    DÉLIBÉRÉMENT PAS `safetensors.safe_open` / `load_file`. Ceux-là mappent le
    conteneur entier, et Windows facture un mapping multi-Go au commit dès sa
    création — mesures dans le docstring du module : 11,67 Go de commit pour
    OUVRIR un fichier de 11,46 Go, avant toute lecture.

    ET CETTE DUPLICATION DE `fp8_export._Reader` EST VOULUE — NE PAS LA FACTORISER.
    Ce fichier n'est pas importé : il est lancé comme CLI par le python d'ai-toolkit
    (un autre interpréteur, un autre venv), et il peut être expédié tel quel en
    TEXTE SOURCE à un pod loué, sans le paquet LDS autour. Un `from ..services
    import fp8_export` marcherait ici et échouerait là-bas — le pire endroit pour
    l'apprendre. Même raison, même formulation que celle écrite dans fp8_export
    le jour où le mapping y a été retiré.
    """

    def __init__(self, path):
        self.path = str(path)
        self._fh = open(self.path, 'rb')
        try:
            raw = self._fh.read(8)
            if len(raw) != 8:
                raise ValueError('fichier trop court pour un conteneur safetensors')
            n = struct.unpack('<Q', raw)[0]
            if n <= 0 or n > _HEADER_LEN_MAX:
                raise ValueError("longueur d'en-tête safetensors implausible")
            blob = self._fh.read(n)
            if len(blob) != n:
                raise ValueError("en-tête safetensors tronqué")
            self.header = json.loads(blob.decode('utf-8'))
            if not isinstance(self.header, dict):
                raise ValueError("l'en-tête safetensors n'est pas un objet")
        except Exception as e:
            self._fh.close()
            raise RuntimeError(f'fichier .safetensors illisible ({e})') from e
        self._start = 8 + n
        self.entries = _entries(self.header)

    def get_tensor(self, name):
        spec = self.entries.get(name)
        if spec is None:
            raise RuntimeError(f'{name} absent de {os.path.basename(self.path)}')
        dtype = _torch_dtype_for(spec.get('dtype'))
        if dtype is None:
            raise RuntimeError(f'dtype non supporté {spec.get("dtype")!r} sur {name}')
        begin, end = int(spec['data_offsets'][0]), int(spec['data_offsets'][1])
        nbytes = end - begin
        self._fh.seek(self._start + begin)
        # readinto sur un bytearray : torch.frombuffer avertit sur un buffer en
        # lecture seule, et recopier ensuite tiendrait deux exemplaires du tenseur.
        raw = bytearray(nbytes)
        got = self._fh.readinto(raw)
        if got != nbytes:
            raise RuntimeError(
                f'{os.path.basename(self.path)} est tronqué : {name} annonce '
                f'{nbytes} octets, {got} présents.')
        shape = [int(d) for d in (spec.get('shape') or [])]
        return torch.frombuffer(raw, dtype=dtype).reshape(shape)

    def close(self):
        try:
            self._fh.close()
        except OSError:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
        return False


def _raw_bytes(tensor):
    """Payload contigu little-endian de n'importe quel tenseur, bf16 et fp8 inclus.
    `numpy()` n'a ni float8 ni bfloat16 : on réinterprète par une vue entière de
    MÊME itemsize — octets exacts, aucune conversion de valeurs."""
    flat = tensor.detach().to('cpu').contiguous().reshape(-1)
    view = {torch.float8_e4m3fn: torch.uint8, torch.float8_e5m2: torch.uint8,
            torch.bfloat16: torch.int16}.get(flat.dtype)
    if view is not None:
        flat = flat.view(view)
    return flat.numpy().tobytes()


def _pack_header(index, metadata=None):
    obj = dict(index)
    if metadata:
        obj['__metadata__'] = {str(k): str(v) for k, v in metadata.items()}
    blob = json.dumps(obj, separators=(',', ':')).encode('utf-8')
    blob += b' ' * ((-len(blob)) % 8)        # section data alignée sur 8 octets
    return struct.pack('<Q', len(blob)) + blob


def plan_diffusers_tensors(header):
    """`(plan, unmapped, extra)` — fonction PURE de l'en-tête source.

    `plan` = liste ordonnée de `(diffusers_key, source_key, slice_or_None, dtype,
    shape)`, où `slice` vaut `(dim, start, length)` pour les trois tranches d'un
    QKV fusionné. Ordonnée PAR SOURCE : chaque tenseur source est donc lu une
    seule fois même quand il alimente trois clés diffusers.

    Tout ce dont le GATE a besoin — les clés et les shapes — sort d'ici, donc le
    GATE ne lit aucun octet de poids. C'est aussi ce qui permet d'écrire l'en-tête
    de sortie avant la première lecture, et donc de streamer.
    """
    entries = _entries(header)
    src = {}
    for k, spec in entries.items():
        stripped = k[len(PREFIX):] if k.startswith(PREFIX) else k
        src[stripped] = (k, spec)

    n_layers = max([int(m.group(1)) for k in src if (m := re.match(r"layers\.(\d+)\.", k))], default=-1) + 1
    n_ref = max([int(m.group(1)) for k in src if (m := re.match(r"context_refiner\.(\d+)\.", k))], default=-1) + 1
    n_noise = max([int(m.group(1)) for k in src if (m := re.match(r"noise_refiner\.(\d+)\.", k))], default=-1) + 1
    if 'layers.0.attention.qkv.weight' not in src:
        raise RuntimeError(
            "layers.0.attention.qkv.weight absent : ce n'est pas un checkpoint "
            "Z-Image au format ComfyUI.")
    qkv_rows = int(src['layers.0.attention.qkv.weight'][1]['shape'][0])
    # // 3 suppose MHA (qkv fusionne = 3*dim, n_kv_heads == n_heads). Vrai pour
    # Z-Image (30 heads, pas de GQA). On asserte la divisibilite : un format GQA
    # (qkv = dim + 2*kv_dim) donnerait une dim fausse + des slices q/k/v decalees.
    if qkv_rows % 3 != 0:
        raise RuntimeError(
            f"qkv.weight rows={qkv_rows} non divisible par 3 : checkpoint non-MHA "
            f"(GQA ?) non supporte par ce convertisseur Z-Image.")
    dim = qkv_rows // 3
    print(f"  derived config: n_layers={n_layers} n_context_refiner={n_ref} n_noise_refiner={n_noise} dim={dim}")
    if n_noise != n_ref:
        # z_image_to_diffusers utilise UN seul n_refiner_layers pour les DEUX refiners.
        # Si les profondeurs different, la table n'emet que min(n_noise, n_ref) couches :
        # les couches reelles en trop (ex. noise_refiner.n_ref..n_noise-1) seraient
        # SILENCIEUSEMENT abandonnees, et le GATE (qui compare a un modele cfg.json a
        # profondeur unique) afficherait quand meme PASSED. Le modele diffusers cible
        # ne peut de toute facon pas representer des profondeurs asymetriques -> on
        # refuse fort plutot que de produire un transformer incomplet en vert.
        raise RuntimeError(
            f"noise_refiner depth ({n_noise}) != context_refiner ({n_ref}) : profondeurs "
            f"asymetriques non representables en diffusers ZImageTransformer2DModel "
            f"(n_refiner_layers unique). Conversion refusee (poids seraient perdus).")
    key_map = z_image_to_diffusers({"n_layers": n_layers, "dim": dim, "n_refiner_layers": n_ref})

    plan, unmapped = [], []
    for diff_key, ref in key_map.items():
        cut = None
        ck = ref
        if isinstance(ref, tuple):
            ck, cut = ref[0], ref[1]
        if ck not in src:
            unmapped.append((diff_key, ck))
            continue
        real_key, spec = src[ck]
        shape = [int(d) for d in (spec.get('shape') or [])]
        if cut is not None:
            d, _start, length = cut
            if d >= len(shape):
                unmapped.append((diff_key, ck))
                continue
            shape = list(shape)
            shape[d] = int(length)
        plan.append((diff_key, real_key, cut, str(spec.get('dtype')), shape))
    used = {(s[0] if isinstance(s, tuple) else s) for s in key_map.values()}
    extra = [k for k in src if k not in used]
    # Groupé par source : un qkv fusionné alimente to_q/to_k/to_v et n'est donc
    # relu ni deux ni trois fois pendant l'écriture.
    plan.sort(key=lambda e: (e[1], e[0]))
    print(f"  mapped {len(plan)} diffusers keys | {len(unmapped)} src-absent | {len(extra)} comfy keys unused")
    for mk in unmapped[:8]:
        print("     SRC-ABSENT:", mk)
    for e in extra[:8]:
        print("     UNUSED-COMFY:", e)
    return plan, unmapped, extra


def gate(plan, cfg_path):
    """Compare les clés/shapes du PLAN au ZImageTransformer2DModel de `cfg_path`,
    construit sur device 'meta'. Aucun octet de poids n'est lu : les shapes
    viennent de l'en-tête source."""
    with open(cfg_path) as f:
        cfg = json.load(f)
    try:
        from diffusers import ZImageTransformer2DModel
    except ImportError:
        from diffusers.models.transformers.transformer_z_image import ZImageTransformer2DModel
    with torch.device("meta"):
        model = ZImageTransformer2DModel.from_config(cfg)
    exp = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    got = {e[0]: tuple(e[4]) for e in plan}
    missing = [k for k in exp if k not in got]
    unexpected = [k for k in got if k not in exp]
    mism = [(k, exp[k], got[k]) for k in exp if k in got and exp[k] != got[k]]
    print(f"\n=== GATE === model keys={len(exp)} | converted={len(got)}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}  shape_mismatch={len(mism)}")
    for m in missing[:15]:
        print("     missing:", m)
    for u in unexpected[:10]:
        print("     unexpected:", u)
    for k, e, g in mism[:10]:
        print(f"     shape: {k} expected {e} got {g}")
    # `unexpected` DOIT compter dans le verdict : diffusers/ai-toolkit chargent en
    # strict=True et levent RuntimeError sur la moindre cle surnumeraire. Un GATE qui
    # ignore `unexpected` peut afficher PASSED puis sauver un .safetensors inchargeable
    # (ex. biais d'attention / adaLN context_refiner emis par la table mais absents du
    # modele). On echoue donc aussi des qu'il y a des cles inattendues.
    ok = (len(missing) == 0 and len(mism) == 0 and len(unexpected) == 0)
    print("\n[GATE PASSED] toutes les cles diffusers remplies, shapes OK, aucune cle en trop" if ok
          else "\n[GATE FAILED] remap incomplet (manquantes / shapes / cles inattendues)")
    return ok


def write_transformer(comfy_path, plan, out_path):
    """Écrit le .safetensors diffusers en STREAMING : l'en-tête de sortie est
    calculé depuis le plan (donc depuis l'en-tête source), puis chaque tenseur
    source est lu une fois, découpé si besoin, écrit, relâché.

    Le pic mémoire est UN tenseur, pas un checkpoint — et rien n'est mappé, donc
    la taille du fichier n'a plus de rapport avec le commit disponible. Écriture
    dans un `.part` renommé à la fin : une conversion interrompue ne laisse pas
    un transformer à moitié écrit que le cache prendrait pour valide."""
    index, offset = {}, 0
    for diff_key, _src_key, _cut, dtype, shape in plan:
        width = _DTYPE_BYTES.get(str(dtype).upper())
        if not width:
            raise RuntimeError(f'dtype non supporté {dtype!r} sur {diff_key}')
        numel = 1
        for d in shape:
            numel *= int(d)
        nbytes = numel * width
        index[diff_key] = {'dtype': dtype, 'shape': list(shape),
                           'data_offsets': [offset, offset + nbytes]}
        offset += nbytes

    tmp = str(out_path) + '.part'
    written = 0
    try:
        with _Reader(comfy_path) as reader, open(tmp, 'wb') as out:
            out.write(_pack_header(index))
            cached_key, cached = None, None
            for diff_key, src_key, cut, _dtype, _shape in plan:
                if src_key != cached_key:
                    cached, cached_key = reader.get_tensor(src_key), src_key
                tensor = cached
                if cut is not None:
                    d, start, length = cut
                    tensor = cached.narrow(d, int(start), int(length))
                out.write(_raw_bytes(tensor))
                written += 1
            del cached
        os.replace(tmp, out_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return written


def main():
    inp, cfg_path = sys.argv[1], sys.argv[2]
    save_dir = None
    if "--save" in sys.argv:
        save_dir = sys.argv[sys.argv.index("--save") + 1]
    print(f"Loading {inp} ...")
    plan, _unmapped, _extra = plan_diffusers_tensors(read_header(inp))
    ok = gate(plan, cfg_path)
    if ok and save_dir:
        tdir = os.path.join(save_dir, "transformer")
        os.makedirs(tdir, exist_ok=True)
        n = write_transformer(inp, plan,
                              os.path.join(tdir, "diffusion_pytorch_model.safetensors"))
        shutil.copy2(cfg_path, os.path.join(tdir, "config.json"))
        print(f"\nsaved diffusers transformer ({n} tensors) -> {tdir}")


if __name__ == "__main__":
    main()
