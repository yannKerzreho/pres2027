/* Briques communes aux pages du site : formatage, chargement, barres de
 * candidatures et fiche de distribution au survol.
 *
 * Les deux onglets qui affichent des barres ne partagent PAS leurs données —
 * l'un lit un snapshot calculé côté serveur, l'autre recalcule tout dans le
 * navigateur à chaque clic — mais ils doivent afficher la même chose de la même
 * façon. D'où un composant qui prend un résumé de distribution, d'où qu'il
 * vienne, et ne sait rien du modèle qui l'a produit.
 */
(function (global) {
  "use strict";

  const NS = "http://www.w3.org/2000/svg";

  // --- formatage (français : virgule décimale, espace insécable avant %) -------
  const nb = (x, d = 1) => x.toFixed(d).replace(".", ",");
  const pct = (x) => nb(100 * x) + " %";
  const prob = (x) => Math.round(100 * x) + " %";
  const el = (t, a = {}) => {
    const e = document.createElementNS(NS, t);
    for (const k in a) e.setAttribute(k, a[k]);
    return e;
  };
  const jours = (s) => Date.parse(s) / 8.64e7;
  const echappe = (s) => String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  async function getJSON(u) {
    const r = await fetch(u);
    if (!r.ok) throw new Error(u);
    return r.json();
  }

  // --- résumé de distribution à partir de tirages ------------------------------
  // MÊME forme que ce qu'exporte `model/core/simulate.py` (quantiles + densité en
  // 24 classes, hauteurs relatives au pic) : l'onglet Scénarios calcule dans le
  // navigateur ce que l'onglet Suivi reçoit tout fait, et la fiche de survol ne
  // fait pas la différence.
  const N_CLASSES = 24;

  function quantile(tri, q) {
    const p = (tri.length - 1) * q, lo = Math.floor(p), hi = Math.ceil(p);
    return tri[lo] + (tri[hi] - tri[lo]) * (p - lo);
  }

  function resumeDepuisTirages(col) {
    const tri = col.slice().sort((a, b) => a - b);
    const moyenne = col.reduce((a, b) => a + b, 0) / col.length;
    const q = [0.05, 0.25, 0.5, 0.75, 0.95].map((p) => quantile(tri, p));
    const lo = quantile(tri, 0.005), hi = quantile(tri, 0.995);
    const dx = (hi - lo) / N_CLASSES;
    const y = new Array(N_CLASSES).fill(0);
    if (dx > 0) {
      for (const v of col) {
        const j = Math.floor((v - lo) / dx);
        if (j >= 0 && j < N_CLASSES) y[j]++;
      }
    }
    const pic = Math.max(1, ...y);
    return { moyenne, ic90: [q[0], q[4]], quantiles: q,
             densite: { x0: lo, dx, y: y.map((v) => Math.round(100 * v / pic)) } };
  }

  // --- fiche flottante ---------------------------------------------------------
  let _fiche = null;
  function fiche() {
    if (!_fiche) {
      _fiche = document.createElement("div");
      _fiche.className = "fiche";
      _fiche.setAttribute("role", "tooltip");
      document.body.appendChild(_fiche);
    }
    return _fiche;
  }
  function montrerFiche(html, ev) {
    const f = fiche();
    f.innerHTML = html;
    f.dataset.visible = "1";
    placerFiche(ev);
  }
  function placerFiche(ev) {
    const f = fiche(), r = f.getBoundingClientRect(), marge = 14;
    let x = ev.clientX + marge, y = ev.clientY + marge;
    if (x + r.width > innerWidth - 8) x = ev.clientX - marge - r.width;
    if (y + r.height > innerHeight - 8) y = Math.max(8, ev.clientY - marge - r.height);
    f.style.left = Math.max(8, x) + "px";
    f.style.top = y + "px";
  }
  function cacherFiche() {
    if (_fiche) _fiche.dataset.visible = "0";
  }

  /* Croquis de densité + boîte à moustaches, dans la fiche.
   * Un IC 90 % dit où la masse se trouve, jamais comment elle s'y répartit : ces
   * distributions sont franchement dissymétriques (la moyenne d'un candidat en
   * hausse tombe au-dessus de sa médiane), et c'est précisément l'information
   * qu'on perd en n'affichant qu'un intervalle. */
  function croquis(d, q, moyenne, couleur, L = 274, H = 62) {
    if (!d || !d.y || !d.y.length || !d.dx) return "";
    const n = d.y.length, x1 = d.x0 + n * d.dx;
    const bas = H - 16;                          // sous la courbe : la boîte
    const X = (v) => Math.max(0, Math.min(L, L * (v - d.x0) / (x1 - d.x0)));
    const pts = d.y.map((v, j) => `${X(d.x0 + (j + 0.5) * d.dx)},${bas - (bas - 4) * v / 100}`);
    const aire = `${X(d.x0)},${bas} ${pts.join(" ")} ${X(x1)},${bas}`;
    const yb = H - 8;
    return `<svg viewBox="0 0 ${L} ${H}" width="${L}" height="${H}" aria-hidden="true">
      <polygon points="${aire}" fill="${couleur}" opacity="0.22"/>
      <polyline points="${pts.join(" ")}" fill="none" stroke="${couleur}" stroke-width="1.6"/>
      <line x1="0" y1="${bas}" x2="${L}" y2="${bas}" stroke="currentColor" opacity=".18"/>
      <line x1="${X(q[0])}" y1="${yb}" x2="${X(q[4])}" y2="${yb}" stroke="${couleur}"
            stroke-width="1.4" opacity=".75"/>
      <rect x="${X(q[1])}" y="${yb - 3.5}" width="${Math.max(1, X(q[3]) - X(q[1]))}" height="7"
            rx="1.5" fill="${couleur}" opacity=".55"/>
      <line x1="${X(q[2])}" y1="${yb - 4.5}" x2="${X(q[2])}" y2="${yb + 4.5}"
            stroke="currentColor" stroke-width="1.6" opacity=".8"/>
      <line x1="${X(moyenne)}" y1="4" x2="${X(moyenne)}" y2="${bas}" stroke="currentColor"
            stroke-width="1" stroke-dasharray="2 2" opacity=".55"/>
    </svg>`;
  }

  /* Contenu de la fiche pour une candidature. Tolérant : un modèle qui n'exporte
   * ni quantiles ni densité (contrat minimal, cf. README « Ajouter un modèle »)
   * garde une fiche lisible, seulement moins détaillée. */
  function ficheCandidature(it) {
    const c = it.couleur, q = it.quantiles;
    const parti = global.Couleurs ? global.Couleurs.parti(it.nom) : null;
    let corps = "";
    if (q) {
      corps += croquis(it.densite, q, it.moyenne, c);
      corps += `<dl>
        <dt>moyenne</dt><dd>${pct(it.moyenne)}</dd>
        <dt>médiane</dt><dd>${pct(q[2])}</dd>
        <dt>moitié centrale</dt><dd>${pct(q[1])} – ${pct(q[3])}</dd>
        <dt>IC 90 %</dt><dd>${pct(q[0])} – ${pct(q[4])}</dd>`;
    } else {
      corps += `<dl><dt>moyenne</dt><dd>${pct(it.moyenne)}</dd>`;
      if (it.ic90) corps += `<dt>IC 90 %</dt><dd>${pct(it.ic90[0])} – ${pct(it.ic90[1])}</dd>`;
    }
    if (it.pTop2 != null) {
      corps += `<div class="sep"></div>
        <dt>qualifié·e au 2<sup>nd</sup> tour</dt><dd>${prob(it.pTop2)}</dd>
        <dt>en tête au 1<sup>er</sup> tour</dt><dd>${prob(it.pPremier)}</dd>`;
    }
    corps += "</dl>";
    const asym = q && Math.abs(it.moyenne - q[2]) > 0.004
      ? `<p class="note">Distribution dissymétrique : la moyenne est
         ${it.moyenne > q[2] ? "au-dessus" : "en dessous"} de la médiane.</p>` : "";
    return `<h4><i style="background:${c}"></i>${echappe(it.nom)}</h4>` +
           (parti ? `<div class="parti">${echappe(parti)}</div>` : "") + corps + asym;
  }

  /* Barres de candidatures — le composant partagé par les deux onglets.
   *
   * `items` : [{nom, moyenne, ic90, pTop2, pPremier, quantiles?, densite?}],
   * dans l'ordre d'affichage souhaité. La couleur n'est PAS un paramètre : elle
   * vient de la banque (assets/palette.js), sinon deux pages donneraient deux
   * couleurs au même candidat.
   */
  function barres(host, items, opts) {
    opts = opts || {};
    const col = (nom) => (global.Couleurs ? global.Couleurs.couleur(nom) : "var(--fg)");
    // Échelle commune : bornée par la moustache la plus longue, pour que les
    // barres restent comparables d'une candidature à l'autre.
    const hautes = items.map((it) => (it.ic90 ? it.ic90[1] : it.moyenne));
    const echelle = (opts.echelle || Math.max(...hautes)) * 1.02;
    host.innerHTML = "";
    host.className = "barres";

    const tete = document.createElement("div");
    tete.className = "entetes";
    tete.innerHTML = `<div>Candidature</div><div>Part au scrutin</div>
      <div class="p"><span>2<sup>nd</sup> t.</span><span>1<sup>er</sup></span></div>`;
    host.appendChild(tete);

    items.forEach((it) => {
      const c = (it.couleur = col(it.nom));
      const r = document.createElement("div");
      r.className = "rangee";
      const mous = it.ic90
        ? `<div class="moustache" style="left:${100 * it.ic90[0] / echelle}%;
             width:${100 * (it.ic90[1] - it.ic90[0]) / echelle}%"></div>` : "";
      const probs = it.pTop2 != null
        ? `<span class="p2">${prob(it.pTop2)}</span><span class="p1">${prob(it.pPremier)}</span>`
        : "";
      r.innerHTML = `<div class="nom">${echappe(it.nom)}</div>
        <div class="track">
          <div class="bar" style="width:${100 * it.moyenne / echelle}%;background:${c}"></div>
          ${mous}
          <div class="val" style="left:${100 * it.moyenne / echelle}%">${pct(it.moyenne)}</div>
        </div>
        <div class="probs">${probs}</div>`;
      // La fiche suit le pointeur sur toute la rangée, pas seulement sur la
      // barre : viser un rectangle de 6 px de haut à la souris est pénible, et
      // au clavier il n'y aurait aucune cible du tout.
      r.addEventListener("mouseenter", (ev) => montrerFiche(ficheCandidature(it), ev));
      r.addEventListener("mousemove", placerFiche);
      r.addEventListener("mouseleave", cacherFiche);
      host.appendChild(r);
    });
  }

  global.Site = {
    NS, el, nb, pct, prob, jours, echappe, getJSON,
    barres, resumeDepuisTirages, croquis,
    montrerFiche, placerFiche, cacherFiche,
  };
})(window);
