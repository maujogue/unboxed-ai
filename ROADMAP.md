# Roadmap

## Facile — infra déjà en place

### Lier les lésions aux images avec SEG superposé
`generate_nodule_images.py` fait déjà l'overlay CT+SEG.
Il faut juste le brancher sur `findings_validation.json` pour pointer vers les bons findings.

### Rapports sur imageries sans rapport
`generate_final_report()` existe déjà dans `report_generation.py` avec routing LUNGRAD/RECIST.
C'est du prompt engineering + filtrage des cas sans rapport.

### Patient à la fois
Refactoring pur — extraire la boucle `for` en fonction `process_accession(acc_num)`.

---

## Moyen — nouveau logic mais bien délimité

### Valider les lésions SEG sans rapport
Identifier les findings SEG qui n'ont pas de correspondant dans le rapport, puis interface de validation (boutons Oui/Non dans Gradio ou HTML interactif).

### Évolution des lésions
Matcher les F1/F2/... d'un patient entre ses visites (données déjà dans `nodules_export.json` avec taille et position), calculer Δ%, classer stable/augmentation/diminution.
La difficulté principale : le matching quand les labels changent entre visites.

### Relier au front
`frontend/front.py` est vide. Gradio est déjà disponible.
La complexité dépend de l'ambition : Gradio simple → facile ; vraie app web → difficile.

---

## Difficile — concept à définir + ML

### Clustering des pathologies par liaisons observées
"Liaisons observées" n'est pas encore défini : co-occurrence de lésions ? proximité spatiale dans le même CT ? corrélation temporelle ?
Il faut d'abord clarifier la sémantique avec un radiologue, puis construire le graphe/matrice de features, puis choisir l'algo de clustering.
C'est du ML de zéro sur un domaine médical.

---

## Ordre suggéré

1. **Patient à la fois** — débloque tout le reste (chaque tâche peut tourner en isolation)
2. **Lier lésions aux images** — rapide, impact visuel immédiat
3. **Évolution des lésions** — utile pour le front
4. **Valider SEG sans rapport** — nécessite le front
5. **Rapports sans imagerie** — peut tourner indépendamment
6. **Front** — agrège tout
7. **Clustering** — R&D, à cadrer avec un radiologue d'abord
