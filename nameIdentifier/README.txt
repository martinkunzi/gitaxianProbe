Required libraries:

opencv 2.4
numpy
sklearn
skimage
tqdm

si un problème survient à l'installation de ces modules ou de leurs dépendences avec pip, des wheels sont disponibles ici: http://www.lfd.uci.edu/~gohlke/pythonlibs/

Le fichier AllSets.json nécessaires à l'entraînement du classificateur se trouve ici: http://mtgjson.com/
L'entraînement prend entre 20 et 45 minutes, le test entre 5 et 15.
Le fichier cls.pkl fourni est le classificateur, l'entraînement n'est donc pas nécessaire.

Utilisation:
Pour entraîner le classificateur:
    python ocr.py train
Pour tester le classificateur:
    python ocr.py test
Pour avoir un exemple de fonctionnement:
    python ocr.py
Pour utiliser le classificateur
    from ocr import classify
    classify(img)
où img est une image opencv (Matrice numpy)
