"""Complète les dates MANQUANTES d'un CSV historique figé à partir d'une
réextraction Wikipedia, sans toucher au reste du fichier.

Les CSV `intentions_2017_wiki.csv`/`_2022` alimentent la calibration : les
régénérer entièrement change des libellés de notice (désambiguïsation
« n°<id> »), en ajoute ou en retire. Ici, seules les cellules `date_debut`,
`date_fin`, `date_notice` VIDES sont remplies, et seulement quand la
réextraction donne une date UNIQUE pour la notice (même URL, même libellé ou
libellé suffixé « n°… » ; à défaut, même URL désignant une seule notice de
part et d'autre). Une notice ambiguë reste sans date.

    python -m sondages.complete_dates --page "Liste de sondages sur l'élection présidentielle française de 2017" \\
        --csv data/parsed/intentions_2017_wiki.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from sondages import wiki

DATES = ["date_debut", "date_fin", "date_notice"]


def complete_dates(ancien: pd.DataFrame, nouveau: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = ancien.copy()
    notes = []
    manquantes = out.loc[out["date_fin"].isna(), ["notice", "notice_url"]].drop_duplicates()
    for notice, url in manquantes.itertuples(index=False):
        base = notice[:-len(" (WP)")] if notice.endswith(" (WP)") else notice
        cand = nouveau[(nouveau["notice_url"] == url)
                       & ((nouveau["notice"] == notice)
                          | nouveau["notice"].str.startswith(notice + " n°"))
                       & nouveau["date_fin"].notna()]
        if cand.empty:
            # libellé corrigé depuis sur Wikipedia (« au 6 avril » -> « 1 au 6
            # avril ») : repli sur l'URL seule, si elle ne désigne qu'UNE notice
            # de chaque côté.
            meme_url = nouveau[(nouveau["notice_url"] == url) & nouveau["date_fin"].notna()]
            if (meme_url["notice"].nunique() == 1
                    and ancien.loc[ancien["notice_url"] == url, "notice"].nunique() == 1):
                cand = meme_url
        dates = cand[DATES[:2]].drop_duplicates()
        if len(dates) != 1:
            if len(dates) > 1:
                notes.append(f"ambigu, laissé sans date : {notice!r} ({len(dates)} dates)")
            continue
        d1, d2 = dates.iloc[0]
        mask = (out["notice"] == notice) & (out["notice_url"] == url) & out["date_fin"].isna()
        out.loc[mask, "date_debut"] = d1
        out.loc[mask, "date_fin"] = d2
        out.loc[mask & out["date_notice"].isna(), "date_notice"] = d2
        notes.append(f"{base} -> {d1} / {d2}")
    return out, notes


def corrige_dates_apres_scrutin(ancien: pd.DataFrame, nouveau: pd.DataFrame, annee: str
                                ) -> tuple[pd.DataFrame, list[str]]:
    """Remplace les dates POSTÉRIEURES au scrutin (après fin mai de l'année de
    l'élection, impossibles) par celles de la réextraction, si la notice (même
    libellé, même URL) y a une date unique et plausible. Cas réel : 2022,
    « Harris Interactive 28-31 décembre » et « Cluster17 27-31 décembre »
    datés 2022-12-31 au lieu de 2021-12-31 (horizon négatif en calibration)."""
    out = ancien.copy()
    notes = []
    borne = f"{annee}-05-31"
    fausses = out.loc[out["date_fin"].notna() & (out["date_fin"] > borne),
                      ["notice", "notice_url"]].drop_duplicates()
    for notice, url in fausses.itertuples(index=False):
        cand = nouveau[(nouveau["notice"] == notice) & (nouveau["notice_url"] == url)
                       & nouveau["date_fin"].notna() & (nouveau["date_fin"] <= borne)]
        dates = cand[DATES[:2]].drop_duplicates()
        if len(dates) != 1:
            notes.append(f"date impossible non corrigée : {notice!r}")
            continue
        d1, d2 = dates.iloc[0]
        mask = (out["notice"] == notice) & (out["notice_url"] == url)
        avant = out.loc[mask, "date_fin"].iloc[0]
        out.loc[mask, "date_debut"] = d1
        out.loc[mask, "date_fin"] = d2
        out.loc[mask & (out["date_notice"] == avant), "date_notice"] = d2
        notes.append(f"corrigé : {notice} {avant} -> {d2}")
    return out, notes


def ajoute_notices_absentes(ancien: pd.DataFrame, nouveau: pd.DataFrame
                            ) -> tuple[pd.DataFrame, list[str]]:
    """Ajoute (en fin de fichier, sans toucher aux lignes existantes) les
    sondages de la réextraction ABSENTS de l'ancien CSV : ni leur URL de notice
    (pour ce tour), ni le couple (institut, date_fin) n'y figurent — un même
    sondage relu sous un autre libellé n'est donc jamais ajouté en double."""
    urls = set(zip(ancien["notice_url"], ancien["tour"]))
    dates = set(zip(ancien["institut"], ancien["date_fin"], ancien["tour"]))
    neuf = nouveau[[(u, t) not in urls and (i, d, t) not in dates
                    for u, t, i, d in zip(nouveau["notice_url"], nouveau["tour"],
                                          nouveau["institut"], nouveau["date_fin"])]]
    neuf = neuf[~neuf["notice"].isin(set(ancien["notice"]))]
    notes = [f"ajouté : {n} ({t})" for n, t in neuf[["notice", "tour"]].drop_duplicates().itertuples(index=False)]
    return pd.concat([ancien, neuf[list(ancien.columns)]], ignore_index=True), notes


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--page", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--ajoute-absentes", action="store_true",
                    help="ajoute aussi les sondages absents de l'ancien CSV (cf. ajoute_notices_absentes)")
    ap.add_argument("--corrige-apres-scrutin", action="store_true",
                    help="corrige aussi les dates postérieures au scrutin (cf. corrige_dates_apres_scrutin)")
    a = ap.parse_args()
    ancien = pd.read_csv(a.csv, dtype=str, keep_default_na=False).replace("", None)
    nouveau, _ = wiki.build_dataset(a.page, pdf_date_fallback=False)
    nouveau = nouveau.astype(object).where(nouveau.notna(), None)
    out, notes = complete_dates(ancien, nouveau)
    if a.corrige_apres_scrutin:
        annee = wiki._year_from_text(a.page)
        out, corr = corrige_dates_apres_scrutin(out, nouveau, annee)
        notes += corr
    if a.ajoute_absentes:
        out, ajouts = ajoute_notices_absentes(out, nouveau)
        notes += ajouts
    print("\n".join(notes))
    print(f"lignes sans date_fin : {ancien['date_fin'].isna().sum()} -> {out['date_fin'].isna().sum()}")
    out.to_csv(a.csv, index=False)


if __name__ == "__main__":
    main()
