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


def test_verdict_regle_figee():
    from model.backtest.lire_essai_2027 import verdict

    def S(lfi_s, lfi_c, v2_s, v2_c, v1_s, v1_c, suf=""):
        return {f"LFI_RN_segment{suf}": lfi_s, f"LFI_RN_cercle{suf}": lfi_c, f"V2_segment{suf}": [v2_s, 8],
                f"V2_cercle{suf}": [v2_c, 8], f"V1_segment{suf}": [v1_s, 9], f"V1_cercle{suf}": [v1_c, 9]}
    assert verdict(S(9.41, 10.53, 2.18, 1.95, 7.54, 8.88), "") == "non séparés"     # §8.97
    assert verdict(S(10, 8, 2, 2.5, 7, 7.5), "") == "cercle"                         # bords inclus
    assert verdict(S(10, 8, 2, 2.6, 7, 7), "") == "non séparés"
    assert verdict(S(6, 9, 2.4, 2, 7.4, 7, "_date"), "_date") == "segment"
