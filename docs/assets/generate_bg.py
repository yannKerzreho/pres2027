"""Génère les bandeaux graphiques (masthead / footer) du site à partir d'un
champ de bruit de Fourier pointilliste, dans l'esprit medialab Sciences Po.
Usage : python3 generate_bg.py  (regénère les PNG dans docs/assets/)

Le bandeau du haut porte le NOM DU SITE, non pas imprimé par-dessus mais
**creusé dans le champ**. L'intensité d'un carré est une valeur signée de +4
(rouge plein) à -4 (bleu plein), 0 valant le fond nu ; là où passe le mot, on la
rapproche de zéro de deux crans — `signe(n) * max(0, |n| - 2)`, cf. `attenue`.
Aucune teinte nouvelle n'entre donc dans l'image : le mot est fait des deux mêmes
couleurs, en moins intense, et du fond là où l'intensité s'annule.

Il apparaît par conséquent là où le champ était dense et s'efface là où il était
déjà clair. C'est volontairement imparfait : un masque net aurait donné un logo
posé sur une texture, pas un titre qui appartient à l'image.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def fourier_noise(x, y, seed=0.0, scale=1.0, jitter=0.0, rng=None):
    # scale : augmente la fréquence spatiale -> taches plus petites.
    # jitter : bruit aléatoire ajouté au warp + à la valeur -> mélange plus
    # fin des deux couleurs, moins de grandes plages lisses monochromes.
    fx, fy = x * scale, y * scale
    warp_x = np.sin(0.08 * fy + seed) * 6.0 + np.cos(0.15 * fx) * 3.0
    warp_y = np.cos(0.08 * fx + seed) * 6.0 + np.sin(0.15 * fy) * 3.0
    if rng is not None and jitter:
        warp_x += rng.uniform(-jitter, jitter)
        warp_y += rng.uniform(-jitter, jitter)
    nx = fx + warp_x
    ny = fy + warp_y
    n = 0
    n += 1.00 * np.sin(0.18 * nx + 0.22 * ny)
    n += 0.80 * np.cos(0.25 * nx - 0.15 * ny + 2.0)
    n += 0.70 * np.sin(0.50 * nx + 0.40 * ny + np.sin(0.4 * nx) * 3.0)
    n += 0.45 * np.sin(1.8 * fx) * np.cos(2.2 * fy)
    # composante haute fréquence : casse les gros blocs uniformes en taches
    # plus petites et davantage imbriquées entre les deux couleurs.
    n += 0.55 * np.sin(3.4 * fx + 1.3) * np.cos(3.0 * fy - 0.7)
    if rng is not None and jitter:
        n += rng.normal(0, jitter * 0.16)
    return n


# Police du titre. DejaVuSans-Bold est livrée avec matplotlib, déjà dans
# `requirements.txt` : le bandeau se régénère donc à l'identique sur n'importe
# quelle machine du projet, y compris la CI Linux, sans dépendre des polices
# système (Arial n'existe pas sur un runner Ubuntu, et un repli silencieux sur la
# bitmap par défaut de PIL produirait un titre minuscule et illisible).
def _police(taille_px):
    chemins = []
    try:
        import matplotlib
        chemins.append(os.path.join(os.path.dirname(matplotlib.__file__),
                                    "mpl-data", "fonts", "ttf", "DejaVuSans-Bold.ttf"))
    except ImportError:
        pass
    chemins += ["/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    for c in chemins:
        if os.path.exists(c):
            return ImageFont.truetype(c, taille_px)
    sys.exit("aucune police grasse trouvée — installer matplotlib (requirements.txt)")


def masque_titre(cols, rows, block_size, texte, hauteur_rel=0.46, couverture=0.5):
    """Masque booléen (rows, cols) : True sur les carrés que le mot traverse.

    Le texte est rasterisé à pleine résolution puis moyenné par carré, et un
    carré compte comme « dans le mot » s'il est couvert à plus de `couverture`.
    Passer directement par la grille de carrés donnerait des lettres crénelées
    au hasard de l'arrondi ; ici le seuil est explicite et les bords restent
    réguliers à l'échelle qui compte, celle du carré.
    """
    W, H = cols * block_size, rows * block_size
    # dichotomie sur la taille de police : `getbbox` est la seule mesure fiable
    # (les métriques nominales incluent des jambages absents de « Pres2027 »).
    cible = hauteur_rel * H
    taille, police = 10, _police(10)
    for _ in range(40):
        essai = _police(taille)
        b = essai.getbbox(texte)
        if b[3] - b[1] >= cible:
            break
        police, taille = essai, int(taille * 1.15) + 1
    else:
        police = essai

    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = d.textbbox((0, 0), texte, font=police)
    d.text(((W - (x1 - x0)) / 2 - x0, (H - (y1 - y0)) / 2 - y0), texte, font=police, fill=255)
    a = np.asarray(img, dtype=float).reshape(rows, block_size, cols, block_size)
    return a.mean(axis=(1, 3)) / 255.0 > couverture


def niveau_signe(val, seuils=(2.2, 1.3, 0.6, 0.15)):
    """Champ continu -> intensité SIGNÉE, de +4 (rouge plein) à -4 (bleu plein),
    0 = rien du tout (le fond nu). C'est la grandeur sur laquelle le masque du
    titre travaille : une seule échelle traversant le blanc, plutôt qu'un couple
    (couleur, niveau positif) où « atténuer » n'aurait pas de sens univoque."""
    s4, s3, s2, s1 = seuils
    a = abs(val)
    if a <= s1:
        return 0
    n = 4 if a > s4 else 3 if a > s3 else 2 if a > s2 else 1
    return n if val > 0 else -n


def attenue(n, crans=2):
    """`signe(n) * max(0, |n| - crans)` — le titre RAPPROCHE DE ZÉRO.

    Zéro, c'est le blanc du bandeau : pas un blanc mélangé à la couleur, le fond
    lui-même. Aucune teinte nouvelle n'apparaît donc dans les lettres, seulement
    du rouge et du bleu moins intenses, et du vide là où l'intensité tombe à 0.

    Version précédente écartée : mélanger la COULEUR vers le blanc en gardant
    l'intensité. Elle produisait des roses et des bleus pâles absents du reste de
    l'image, ce qui donnait un mot peint par-dessus au lieu d'un mot creusé
    dedans.
    """
    return (1 if n > 0 else -1) * max(0, abs(n) - crans) if n else 0


class Bandeau:
    def __init__(self, width, height, block_size, bg, color_a, color_b, y_offset=0, seed=0.0,
                 scale=1.7, jitter=2.4, rng_seed=0, titre=None,
                 titre_hauteur=0.62, titre_crans=2):
        self.width, self.height, self.block_size = width, height, block_size
        self.cols = width // block_size
        self.rows = height // block_size
        self.bg, self.color_a, self.color_b = bg, color_a, color_b
        self.y_offset = y_offset
        self.seed = seed
        self.scale = scale
        self.jitter = jitter
        self.rng = np.random.default_rng(rng_seed)
        self.titre = titre
        self.titre_hauteur = titre_hauteur
        self.titre_crans = titre_crans

    def draw_block(self, draw, x0, y0, level, color):
        bs = self.block_size
        if level == 0:
            return
        if level == 4:
            draw.rectangle([x0, y0, x0 + bs - 1, y0 + bs - 1], fill=color)
        elif level == 3:
            draw.rectangle([x0, y0, x0 + bs - 1, y0 + bs - 1], fill=color)
            dot, spacing = 2, 5
            for dy in range(1, bs, spacing):
                for dx in range(1, bs, spacing):
                    if dx + dot <= bs and dy + dot <= bs:
                        draw.rectangle([x0 + dx, y0 + dy, x0 + dx + dot - 1, y0 + dy + dot - 1], fill=self.bg)
        else:
            dot = 2
            spacing = 6 if level == 1 else 4
            for dy in range(0, bs, spacing):
                for dx in range(0, bs, spacing):
                    if dx + dot <= bs and dy + dot <= bs:
                        draw.rectangle([x0 + dx, y0 + dy, x0 + dx + dot - 1, y0 + dy + dot - 1], fill=color)

    def render(self):
        img = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        creux = (masque_titre(self.cols, self.rows, self.block_size, self.titre,
                              self.titre_hauteur) if self.titre else None)
        for y in range(self.rows):
            for x in range(self.cols):
                val = fourier_noise(x, self.y_offset + y, self.seed, self.scale, self.jitter, self.rng)
                n = niveau_signe(val)
                # Le titre RAPPROCHE DE ZÉRO les carrés qu'il traverse : mêmes
                # deux couleurs, deux crans d'intensité en moins, et le blanc du
                # fond dès que l'intensité s'annule (cf. `attenue`).
                if creux is not None and creux[y, x]:
                    n = attenue(n, self.titre_crans)
                color = self.color_a if n > 0 else self.color_b
                self.draw_block(draw, x * self.block_size, y * self.block_size, abs(n), color)
        return img


if __name__ == "__main__":
    # Bandeaux très allongés (8:1 en haut) : ils s'affichent en pleine largeur
    # sans recadrage, donc leur hauteur à l'écran est celle de l'image divisée
    # par son rapport. Un format plus carré donnerait un titre haut de 400 px sur
    # un grand écran.
    W, BS = 2400, 9  # blocs plus petits -> taches plus fines
    # Identité "Marianne" (charte de l'État, 2020) : bleu marine assez sombre
    # adopté sous Macron pour le drapeau et le logo de l'État, plutôt que le
    # bleu clair historique.
    RED = (225, 0, 15)    # Rouge Marianne #E1000F
    BLUE = (0, 0, 145)    # Bleu Marianne #000091
    RED_DARK = (255, 90, 100)   # variante éclaircie pour fond sombre
    BLUE_DARK = (110, 110, 255)  # variante éclaircie pour fond sombre

    TITRE = "Pres2027"
    # Bandeau masthead : c'est LE titre du site, donc plus haut qu'un simple
    # ornement (300 px pour 1800 de large, soit 6:1). Le front l'affiche en
    # `<img>` pleine largeur plutôt qu'en `background-size:cover` : le mot y
    # reste entier à toutes les tailles d'écran, là où un fond recadré l'aurait
    # amputé sur mobile.
    Bandeau(W, 300, BS, (255, 255, 255, 255), RED, BLUE, y_offset=0, rng_seed=1,
            titre=TITRE).render().save("bg-top.png")
    Bandeau(W, 300, BS, (18, 22, 30, 255), RED_DARK, BLUE_DARK, y_offset=0, rng_seed=1,
            titre=TITRE).render().save("bg-top-dark.png")
    # bandeau pied de page (clair) — continuité du même champ de bruit, plus bas
    Bandeau(W, 160, BS, (255, 255, 255, 255), RED, BLUE, y_offset=60, rng_seed=2).render().save("bg-bottom.png")
    # bandeau pied de page (sombre)
    Bandeau(W, 160, BS, (18, 22, 30, 255), RED_DARK, BLUE_DARK, y_offset=60, rng_seed=2).render().save("bg-bottom-dark.png")
    print("bandeaux générés.")
