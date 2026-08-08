"""🧬 PROVENANCE DE GÉNÉRATION d'une pile — de quelles pastilles du board sort
cette image.

Trois notions de descendance cohabitent sur le ◉ LoRA Canvas et il ne faut pas
les confondre :

  · la lignée d'ENTRAÎNEMENT (`tree.edges`) : quel run a continué quel run ;
  · le lien image→pastille : quelle sauvegarde a produit cette image ;
  · la PROVENANCE DE GÉNÉRATION, celle-ci : un blend charge N LoRA, donc son
    image descend de N pastilles à la fois. C'est la seule qui soit
    multi-parents, et c'est pourquoi elle a besoin de sa propre donnée.

Une cellule savait déjà dire d'où vient son LoRA de TÊTE (colonnes `record_id` /
`step`). Les membres empilés, eux, n'étaient identifiés que par leur nom de
fichier. Ce module épingle le fait qu'ils portent désormais leur origine, et —
tout aussi important — qu'une pile lancée AVANT ce changement rende `None` au
lieu d'une origine inventée.
"""
from tests.test_blend_sweep import _two_lora_stack


def test_each_stacked_member_records_the_pill_it_came_from(app, monkeypatch, tmp_path):
    """Le membre d'une pile porte (record_id, step) — l'identité de la pastille
    cliquée sur le board, pas seulement un nom de fichier."""
    import json
    from app.models import LoraTestImage
    with app.app_context():
        launch, cp_a, cp_b, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weight': 0.9, 'record_id': 11, 'step': 2000},
                     {'weight': 0.55, 'record_id': 22, 'step': 1000})
        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        # La tête garde son chemin d'origine : des colonnes, comme avant.
        assert (row.record_id, row.step) == (11, 2000)
        member = [m for m in json.loads(row.extra_loras) if m.get('combined')][0]
        assert member['filename'] == cp_b
        assert (member['record_id'], member['step']) == (22, 1000)


def test_the_stack_reads_back_as_one_parent_per_member(app, monkeypatch, tmp_path):
    """`stack_of_row` rend la pile ENTIÈRE avec une origine par membre, tête
    comprise : c'est la liste dont les arêtes multi-parents seront tirées."""
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        launch, cp_a, cp_b, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weight': 0.9, 'record_id': 11, 'step': 2000},
                     {'weight': 0.55, 'record_id': 22, 'step': 1000})
        row = LoraTestImage.query.filter_by(run_id=out['run_id']).one()
        members = lts.stack_of_row(row)
        assert [(m['filename'], m['record_id'], m['step'], m['head']) for m in members] == [
            (cp_a, 11, 2000, True),
            (cp_b, 22, 1000, False),
        ]


def test_a_sweep_stamps_the_same_origins_on_every_combination(app, monkeypatch, tmp_path):
    """Le balayage change les POIDS, jamais la provenance : les quatre images
    descendent des deux mêmes pastilles."""
    from app.services import lora_test_studio as lts
    from app.models import LoraTestImage
    with app.app_context():
        launch, cp_a, cp_b, _, _, _ = _two_lora_stack(tmp_path, monkeypatch)
        out = launch({'weights': [0.6, 0.8], 'record_id': 11, 'step': 2000},
                     {'weights': [0.4, 1.0], 'record_id': 22, 'step': 1000})
        assert out['created'] == 4
        rows = LoraTestImage.query.filter_by(run_id=out['run_id']).all()
        origins = {tuple((m['record_id'], m['step']) for m in lts.stack_of_row(r))
                   for r in rows}
        assert origins == {((11, 2000), (22, 1000))}


def test_a_stack_launched_before_this_says_UNKNOWN_rather_than_inventing_one(app):
    """Le contrat qui compte pour la vue : le JSON d'une cellule est figé à sa
    création, donc une pile plus ancienne n'a pas d'origine pour ses membres. Elle
    doit rendre None — « je ne sais pas » — et surtout pas retomber sur une
    pastille voisine. Pas de parent connu = pas d'arête, et la vue le dira."""
    import json
    from types import SimpleNamespace
    from app.services import lora_test_studio as lts
    with app.app_context():
        legacy = SimpleNamespace(
            dataset_id=None, checkpoint='z image' + chr(92) + 'lora_aaa.safetensors',
            strength=0.9, record_id=None, step=None,
            # exactement ce qu'écrivaient les runs d'avant : ni record_id ni step
            extra_loras=json.dumps([{'filename': 'z image' + chr(92) + 'lora_bbb.safetensors',
                                     'strength': 0.55, 'combined': True,
                                     'dataset_id': 7, 'trigger': 'bbb'}]))
        members = lts.stack_of_row(legacy)
        assert len(members) == 2
        assert members[1]['record_id'] is None and members[1]['step'] is None
        # …et le reste de la composition reste lisible : on perd l'arête, pas la pile.
        assert members[1]['trigger'] == 'bbb'
        assert members[1]['weight'] == 0.55
