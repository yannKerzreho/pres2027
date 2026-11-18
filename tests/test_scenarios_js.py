"""Calcul de l'onglet Scénarios (`docs/scenarios.html`), exécuté sous node.

Régression du 2026-11-19 : le bruit de projection au scrutin était tiré par RANG
dans la liste cochée. Retirer Arthaud décalait le bruit de tous les candidats
suivants : l'écart avant/après était un nouveau tirage de Monte-Carlo, et
P(Mélenchon au 2nd tour) pouvait baisser quand on retirait un candidat de gauche.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

PAGE = Path(__file__).resolve().parent.parent / "docs" / "scenarios.html"
NODE = shutil.which("node")


def _fonction(src: str, nom: str) -> str:
    """Source d'une `function nom(...) {...}` de premier niveau (accolades appariées)."""
    i = src.index(f"function {nom}(")
    j = src.index("{", i)
    prof = 0
    for k in range(j, len(src)):
        prof += {"{": 1, "}": -1}.get(src[k], 0)
        if prof == 0:
            return src[i:k + 1]
    raise ValueError(nom)


def _js(code: str):
    src = PAGE.read_text()
    reps = re.search(r"const REPLIQUES = \d+;", src).group(0)
    prog = "\n".join([reps] + [_fonction(src, n) for n in ("rng", "normales", "projeteAuScrutin")]
                     + [code])
    out = subprocess.run([NODE, "-e", prog], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.skipif(NODE is None, reason="node absent")
def test_retirer_un_candidat_ne_rebat_pas_le_bruit_des_autres():
    # 4 candidats, parts fixes. Sans le candidat 0, on renormalise : le RAPPORT
    # entre 2 et 3 est inchangé avant projection, il doit l'être après, tirage
    # par tirage (le bruit en log-ratio s'ajoute, la normalisation s'annule).
    r = _js("""
      var D = {candidats: ["A","B","C","E"], projection: {sigma2: 0.1, tau: 260,
               horizon_jours: 150, delta_scale: 0.4, delta_skew: 0.03, delta_tail: 0.96}};
      var tous = [], sans0 = [];
      for (var s = 0; s < 50; s++) { tous.push([0.1, 0.3, 0.35, 0.25]); sans0.push([0.3/0.9, 0.35/0.9, 0.25/0.9]); }
      var a = projeteAuScrutin(tous, [0,1,2,3]), b = projeteAuScrutin(sans0, [1,2,3]);
      var ecart = 0;
      for (var s = 0; s < a.length; s++)
        ecart = Math.max(ecart, Math.abs(Math.log(a[s][2]/a[s][3]) - Math.log(b[s][1]/b[s][2])));
      console.log(JSON.stringify({ecart: ecart, n: a.length}));
    """)
    assert r["n"] == 300
    assert r["ecart"] < 1e-9
