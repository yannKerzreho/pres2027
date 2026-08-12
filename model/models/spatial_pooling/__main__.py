"""`.venv/bin/python -m model.models.spatial_pooling` : calibre la variance
d'excès (house effects) propre à ce modèle (`calibration.py`). Le saut
terminal (nowcast -> scrutin) n'a plus de calibration propre à ce modèle --
il vient de la banque commune (`model/core/opinion.py`), calibrée par
`.venv/bin/python -m model.core.opinion`, pas d'ici."""

from .calibration import main

if __name__ == "__main__":
    main()
