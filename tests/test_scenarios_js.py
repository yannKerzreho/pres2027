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


@pytest.mark.skipif(NODE is None, reason="node absent")
def test_texte_des_boutons_coches_lisible_dans_les_deux_themes():
    """Un bouton coché (aplat de la couleur du candidat) garde au moins 3:1 de
    contraste avec son texte, en clair comme en sombre — Riner sombre (#c6d478)
    tombait à 1,6:1 en blanc."""
    palette = PAGE.parent / "assets" / "palette.js"
    prog = f"""
      global.window = global; var theme = false;
      global.matchMedia = () => ({{matches: theme}});
      require({json.dumps(str(palette))});
      var C = window.Couleurs, pire = 99, qui = "";
      var lum = h => {{ var v = h.slice(1).match(/../g).map(x => {{ var c = parseInt(x,16)/255;
        return c <= 0.04045 ? c/12.92 : Math.pow((c+0.055)/1.055, 2.4); }});
        return 0.2126*v[0] + 0.7152*v[1] + 0.0722*v[2]; }};
      for (theme of [false, true]) for (var n of Object.keys(C.CANDIDATS)) {{
        var f = C.couleur(n), t = C.texteSur(f), a = lum(f), b = lum(t === "#fff" ? "#ffffff" : t);
        var r = (Math.max(a,b) + 0.05) / (Math.min(a,b) + 0.05);
        if (r < pire) {{ pire = r; qui = n + (theme ? " (sombre)" : ""); }}
      }}
      console.log(JSON.stringify({{pire: pire, qui: qui}}));
    """
    out = subprocess.run([NODE, "-e", prog], capture_output=True, text=True, check=True)
    r = json.loads(out.stdout)
    assert r["pire"] >= 3, r
