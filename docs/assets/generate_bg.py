"""Génère les bandeaux graphiques (masthead / footer) du site à partir d'un
champ de bruit de Fourier pointilliste, dans l'esprit medialab Sciences Po.
Usage : python3 generate_bg.py  (regénère les PNG dans site/assets/)
"""
import numpy as np
from PIL import Image, ImageDraw

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


class Bandeau:
    def __init__(self, width, height, block_size, bg, color_a, color_b, y_offset=0, seed=0.0,
                 scale=1.7, jitter=2.4, rng_seed=0):
        self.width, self.height, self.block_size = width, height, block_size
        self.cols = width // block_size
        self.rows = height // block_size
        self.bg, self.color_a, self.color_b = bg, color_a, color_b
        self.y_offset = y_offset
        self.seed = seed
        self.scale = scale
        self.jitter = jitter
        self.rng = np.random.default_rng(rng_seed)

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
        for y in range(self.rows):
            for x in range(self.cols):
                val = fourier_noise(x, self.y_offset + y, self.seed, self.scale, self.jitter, self.rng)
                color, level = None, 0
                if val > 0.15:
                    color = self.color_a
                    level = 4 if val > 2.2 else 3 if val > 1.3 else 2 if val > 0.6 else 1
                elif val < -0.15:
                    color = self.color_b
                    level = 4 if val < -2.2 else 3 if val < -1.3 else 2 if val < -0.6 else 1
                self.draw_block(draw, x * self.block_size, y * self.block_size, level, color)
        return img


if __name__ == "__main__":
    W, BS = 1800, 9  # blocs plus petits -> taches plus fines
    # Identité "Marianne" (charte de l'État, 2020) : bleu marine assez sombre
    # adopté sous Macron pour le drapeau et le logo de l'État, plutôt que le
    # bleu clair historique.
    RED = (225, 0, 15)    # Rouge Marianne #E1000F
    BLUE = (0, 0, 145)    # Bleu Marianne #000091
    RED_DARK = (255, 90, 100)   # variante éclaircie pour fond sombre
    BLUE_DARK = (110, 110, 255)  # variante éclaircie pour fond sombre

    # bandeau masthead (clair) — sert de fond derrière le header
    Bandeau(W, 220, BS, (255, 255, 255, 255), RED, BLUE, y_offset=0, rng_seed=1).render().save("bg-top.png")
    # bandeau masthead (sombre)
    Bandeau(W, 220, BS, (18, 22, 30, 255), RED_DARK, BLUE_DARK, y_offset=0, rng_seed=1).render().save("bg-top-dark.png")
    # bandeau pied de page (clair) — continuité du même champ de bruit, plus bas
    Bandeau(W, 160, BS, (255, 255, 255, 255), RED, BLUE, y_offset=60, rng_seed=2).render().save("bg-bottom.png")
    # bandeau pied de page (sombre)
    Bandeau(W, 160, BS, (18, 22, 30, 255), RED_DARK, BLUE_DARK, y_offset=60, rng_seed=2).render().save("bg-bottom-dark.png")
    print("bandeaux générés.")
