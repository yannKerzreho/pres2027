"""Résultat générique d'une calibration bayésienne.

`Bank` ne connaît rien du domaine (pas de "house effects", pas de "biais par
bloc") : c'est un sac de paramètres nommés, résumés en moyenne/écart-type
postérieurs (xarray), accessibles par attribut. Les noms, les dimensions et
les coordonnées sont entièrement décidés par le modèle qui construit la
Bank — via les noms qu'il donne à ses sites NumPyro (`numpyro.deterministic`)
et les `dims`/`coords` qu'il passe à la construction.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr


def _samples_to_dataset(samples: dict, dims: dict[str, list[str]],
                        coords: dict[str, list]) -> xr.Dataset:
    data_vars = {}
    for name, arr in samples.items():
        arr = np.asarray(arr)
        extra = dims.get(name, [f"{name}_dim{i}" for i in range(arr.ndim - 2)])
        data_vars[name] = (["chain", "draw", *extra], arr)
    return xr.Dataset(data_vars, coords=coords)


def _dataset_to_jsonable(ds: xr.Dataset) -> dict:
    def var(v):
        return {"dims": list(v.dims), "data": np.asarray(v.values).tolist()}
    return {"coords": {k: var(v) for k, v in ds.coords.items()},
            "data_vars": {k: var(v) for k, v in ds.data_vars.items()}}


def _dataset_from_jsonable(d: dict) -> xr.Dataset:
    coords = {k: (v["dims"], np.array(v["data"])) for k, v in d["coords"].items()}
    data_vars = {k: (v["dims"], np.array(v["data"])) for k, v in d["data_vars"].items()}
    return xr.Dataset(data_vars, coords=coords)


class Param:
    """Un paramètre calibré : moyenne/écart-type du postérieur, indexable par
    coordonnées nommées si le site en a (ex. `institut`, `bloc`)."""

    def __init__(self, mean: xr.DataArray, sd: xr.DataArray):
        self.mean, self.sd = mean, sd

    def at(self, **coords) -> tuple[float, float]:
        """Lookup par nom : `bank.institut_bias.at(institut="Ifop", bloc="droite")`."""
        return float(self.mean.sel(**coords)), float(self.sd.sel(**coords))

    def item(self) -> tuple[float, float]:
        """Pour un paramètre scalaire (pas de dimension indexée)."""
        return float(self.mean), float(self.sd)


class Bank:
    """Résultat d'une calibration : `bank.<nom_du_site>` -> `Param`. Aucun
    schéma fixe imposé — construite avec les `samples` bruts du postérieur
    (dict `{site: array (chain, draw, *event_shape)}`, tel que renvoyé par
    `run_numpyro_mcmc`) et, pour les sites indexés, leurs `dims`/`coords`."""

    def __init__(self, samples: dict, dims: dict[str, list[str]] | None = None,
                 coords: dict[str, list] | None = None):
        ds = _samples_to_dataset(samples, dims or {}, coords or {})
        self._mean = ds.mean(("chain", "draw"))
        self._sd = ds.std(("chain", "draw"))

    def __getattr__(self, name: str) -> Param:
        try:
            return Param(self._mean[name], self._sd[name])
        except KeyError:
            raise AttributeError(name)

    def save(self, path: Path) -> None:
        payload = {"mean": _dataset_to_jsonable(self._mean),
                   "sd": _dataset_to_jsonable(self._sd)}
        Path(path).write_text(json.dumps(payload))

    @classmethod
    def load(cls, path: Path) -> "Bank | None":
        p = Path(path)
        if not p.exists():
            return None
        payload = json.loads(p.read_text())
        bank = cls.__new__(cls)
        bank._mean = _dataset_from_jsonable(payload["mean"])
        bank._sd = _dataset_from_jsonable(payload["sd"])
        return bank
