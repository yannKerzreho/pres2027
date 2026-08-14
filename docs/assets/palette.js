/* Banque de couleurs du site — une seule source pour toutes les pages.
 *
 * POURQUOI une banque, et pas une palette par graphique : la même candidature
 * apparaît dans le suivi des intentions, dans les scénarios, dans les légendes
 * et dans les barres. Une palette indexée sur le RANG (ce que faisait le site)
 * donnait au RN le rouge un jour et l'orange le lendemain, selon qu'il passait
 * devant ou derrière — la couleur ne voulait alors plus rien dire.
 *
 * LES TEINTES sont celles que la Wikipédia francophone associe aux partis. Les
 * LUMINOSITÉS, elles, sont ajustées : les conventions donnent trois bleus
 * marine presque identiques (LR, RN, Reconquête) et trois rouges presque
 * identiques (LO, LFI, PCF), indistinguables sur une courbe. Chaque parti garde
 * donc sa teinte mais reçoit sa propre place sur l'échelle clair-sombre, et deux
 * personnalités d'un même parti (Le Pen / Bardella, Attal / Lecornu) sont
 * séparées de la même façon — l'onglet Scénarios peut les afficher côte à côte.
 *
 * AJOUTER UN CANDIDAT : une ligne dans CANDIDATS. S'il n'y est pas, il reçoit
 * une couleur neutre stable (jamais une couleur de parti qui ne serait pas la
 * sienne) — mieux vaut un gris identifiable qu'un faux signal partisan.
 */
(function (global) {
  "use strict";

  const PARTIS = {
    LO:   "Lutte ouvrière",
    NPA:  "Nouveau Parti anticapitaliste",
    LFI:  "La France insoumise",
    PCF:  "Parti communiste français",
    PS:   "Parti socialiste",
    PP:   "Place publique",
    EELV: "Les Écologistes",
    RE:   "Renaissance",
    HOR:  "Horizons",
    LR:   "Les Républicains",
    DLF:  "Debout la France",
    RN:   "Rassemblement national",
    REC:  "Reconquête",
    DVD:  "Divers droite",
  };

  // Ordre volontairement gauche → droite : lu de haut en bas, le tableau donne
  // aussi le dégradé de l'axe politique, ce que l'onglet Scénarios dessine.
  const CANDIDATS = {
    "Arthaud":        { parti: "LO",   couleur: "#8E1B2B" },
    "Poutou":         { parti: "NPA",  couleur: "#A8323C" },
    "Mélenchon":      { parti: "LFI",  couleur: "#CC2443" },  // teinte LFI de référence
    "Ruffin":         { parti: "LFI",  couleur: "#E9636F" },
    "Roussel":        { parti: "PCF",  couleur: "#E4502F" },
    "Glucksmann":     { parti: "PP",   couleur: "#E4548C" },
    "Hollande":       { parti: "PS",   couleur: "#B2426F" },
    "Faure":          { parti: "PS",   couleur: "#F07FA6" },  // rose PS de référence
    "Tondelier":      { parti: "EELV", couleur: "#3B9E56" },
    "Villepin":       { parti: "DVD",  couleur: "#96805C" },
    "Attal":          { parti: "RE",   couleur: "#DFA300" },  // jaune Renaissance, assombri
    "Lecornu":        { parti: "RE",   couleur: "#B0821A" },
    "Philippe":       { parti: "HOR",  couleur: "#0E97A8" },
    "Retailleau":     { parti: "LR",   couleur: "#2C61C6" },  // bleu LR de référence
    "Wauquiez":       { parti: "LR",   couleur: "#5E8AD9" },
    "Lisnard":        { parti: "LR",   couleur: "#7FA3E0" },
    "Darmanin":       { parti: "LR",   couleur: "#4374C9" },
    "Dupont-Aignan":  { parti: "DLF",  couleur: "#8089C2" },
    // Bleu acier DÉSATURÉ, pas un bleu RN plus clair : Bardella et Retailleau
    // peuvent figurer ensemble dans un scénario, et deux bleus saturés voisins
    // y devenaient interchangeables. La teinte reste celle du RN, c'est le
    // chroma qui les sépare.
    "Bardella":       { parti: "RN",   couleur: "#54789F" },
    "Le Pen":         { parti: "RN",   couleur: "#123C74" },  // marine RN de référence
    "Zemmour":        { parti: "REC",  couleur: "#4B3A82" },
    "Knafo":          { parti: "REC",  couleur: "#7A66B0" },
  };

  // Un slot du modèle peut être nommé par le PARTI et non par la personne
  // (« RN » dans SLOTS, model/core/live_dataset.py). On le rattache ici à la
  // candidature qu'il représente, plutôt que de dupliquer une couleur.
  const SLOTS_PARTI = { "RN": "Le Pen" };

  const NEUTRES = ["#6B7280", "#8B7355", "#5F7A6B", "#7A6A80", "#4E6A78"];

  const sansAccents = (s) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();
  const INDEX = {};
  for (const nom in CANDIDATS) INDEX[sansAccents(nom)] = nom;
  for (const slot in SLOTS_PARTI) INDEX[sansAccents(slot)] = SLOTS_PARTI[slot];

  /* Libellé → nom de candidat connu.
   * Les libellés du site portent souvent le parti entre parenthèses
   * (« Mélenchon (LFI) ») alors que les scénarios n'utilisent que le patronyme
   * (« Mélenchon ») : on retire donc la parenthèse avant de chercher. */
  function resoudre(libelle) {
    if (!libelle) return null;
    const nu = sansAccents(String(libelle).replace(/\s*\(.*\)\s*$/, "").trim());
    return INDEX[nu] || null;
  }

  // --- conversions couleur (pour la variante sombre) ---------------------------
  function hexVersHsl(hex) {
    const n = parseInt(hex.slice(1), 16);
    const r = ((n >> 16) & 255) / 255, v = ((n >> 8) & 255) / 255, b = (n & 255) / 255;
    const mx = Math.max(r, v, b), mn = Math.min(r, v, b), d = mx - mn;
    let h = 0;
    if (d) {
      if (mx === r) h = ((v - b) / d) % 6;
      else if (mx === v) h = (b - r) / d + 2;
      else h = (r - v) / d + 4;
    }
    const l = (mx + mn) / 2;
    return [((h * 60) + 360) % 360, d ? d / (1 - Math.abs(2 * l - 1)) : 0, l];
  }
  function hslVersHex(h, s, l) {
    const c = (1 - Math.abs(2 * l - 1)) * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = l - c / 2;
    const t = h < 60 ? [c, x, 0] : h < 120 ? [x, c, 0] : h < 180 ? [0, c, x]
            : h < 240 ? [0, x, c] : h < 300 ? [x, 0, c] : [c, 0, x];
    return "#" + t.map((v) => Math.round(255 * (v + m)).toString(16).padStart(2, "0")).join("");
  }

  /* Variante pour fond sombre : même teinte, luminosité relevée.
   * Le marine du RN (#123C74) sur un fond #0d1117 est un aplat quasi invisible ;
   * l'éclaircir préserve l'identité (la teinte ne bouge pas) là où choisir une
   * autre couleur en mode sombre casserait la reconnaissance d'un thème à l'autre.
   * On plafonne aussi la saturation : à pleine saturation sur fond noir, les
   * rouges vibrent.
   *
   * La pente compte autant que le décalage. Une première version comprimait tout
   * dans [0,60 ; 0,78] et rendait le marine du RN et le bleu LR quasi identiques
   * en mode sombre — or c'est précisément l'écart de luminosité qui les sépare en
   * mode clair. On garde donc une pente qui préserve l'ORDRE des luminosités du
   * catalogue, sur une plage assez large pour que l'écart survive. */
  function versSombre(hex) {
    const [h, s, l] = hexVersHsl(hex);
    return hslVersHex(h, Math.min(s, 0.72), Math.min(0.86, Math.max(0.46, 0.30 + 0.75 * l)));
  }

  const cacheSombre = {};
  const sombre = () => global.matchMedia && global.matchMedia("(prefers-color-scheme: dark)").matches;

  function hachage(s) {
    let h = 0;
    for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
    return h;
  }

  /* Couleur d'une candidature, adaptée au thème courant. */
  function couleur(libelle) {
    const nom = resoudre(libelle);
    const base = nom ? CANDIDATS[nom].couleur
                     : NEUTRES[hachage(String(libelle || "")) % NEUTRES.length];
    if (!sombre()) return base;
    return (cacheSombre[base] = cacheSombre[base] || versSombre(base));
  }

  /* Nom développé du parti, pour les infobulles — « Rassemblement national »
   * en dit plus au lecteur que « RN », et le site n'a nulle part ailleurs où
   * l'apprendre. Renvoie null si la candidature n'est pas au catalogue. */
  function parti(libelle) {
    const nom = resoudre(libelle);
    return nom ? PARTIS[CANDIDATS[nom].parti] || null : null;
  }

  /* Rejoue un rendu quand l'utilisateur bascule son thème système : les couleurs
   * sont calculées en JS (pas en CSS), donc rien ne se remettrait à jour seul. */
  function surChangementDeTheme(rappel) {
    if (!global.matchMedia) return;
    const mq = global.matchMedia("(prefers-color-scheme: dark)");
    (mq.addEventListener ? mq.addEventListener.bind(mq, "change") : mq.addListener.bind(mq))(rappel);
  }

  global.Couleurs = { couleur, parti, resoudre, sombre, surChangementDeTheme, CANDIDATS, PARTIS };
})(window);
