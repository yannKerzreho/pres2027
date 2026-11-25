"""§8.92 : le retrait V2 ôte duels ET reports de la paire ; V1 n'ôte que les reports."""
import model.backtest.essai_2027 as X


def test_retraits():
    raw, rep, champ = X.donnees()
    assert len(champ) >= 5 and "Le Pen" in champ
    d = X.duels(raw)
    r2, p2 = X.retirer(raw, rep, "v2:Le Pen|Philippe")
    e = X.duels(r2)
    assert e[(e.a == "Le Pen") & (e.b == "Philippe")].empty
    assert len(d) - len(e) == len(d[(d.a == "Le Pen") & (d.b == "Philippe")]) > 0
    assert not p2.hypothese.eq("Philippe – Le Pen").any() and p2.hypothese.eq("Attal – Le Pen").any()
    r1, p1 = X.retirer(raw, rep, "v1:Attal – Le Pen")
    assert len(r1) == len(raw) and not p1.hypothese.eq("Attal – Le Pen").any()
