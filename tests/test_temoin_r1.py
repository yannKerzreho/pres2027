"""§8.77 (c) : témoin R1 reproduit sur 2022 à T1−30 (MAE 2,85)."""
from model.backtest.temoin_r1 import evaluer


def test_r1_2022():
    r = evaluer(2022)
    assert len(r["paires"]) == 4
    assert abs(r["mae"] - 2.85) < 0.05
    ec = {p["paire"][1]: p["ecart"] for p in r["paires"]}
    assert abs(ec["Jean-Luc Mélenchon"] - (-1.9)) < 0.1 and abs(ec["Valérie Pécresse"] - 4.8) < 0.1
