Required libraries:

opencv 2.4
numpy
sklearn
skimage
tqdm

si un probl�me survient � l'installation de ces modules ou de leurs d�pendences avec pip, des wheels sont disponibles ici: http://www.lfd.uci.edu/~gohlke/pythonlibs/

Le fichier AllSets.json n�cessaires � l'entra�nement du classificateur se trouve ici: http://mtgjson.com/
L'entra�nement prend entre 20 et 45 minutes, le test entre 5 et 15.
Le fichier cls.pkl fourni est le classificateur, l'entra�nement n'est donc pas n�cessaire.

Utilisation:
Pour entra�ner le classificateur:
    python ocr.py train
Pour tester le classificateur:
    python ocr.py test
Pour avoir un exemple de fonctionnement:
    python ocr.py
Pour utiliser le classificateur
    from ocr import classify
    classify(img)
o� img est une image opencv (Matrice numpy)
