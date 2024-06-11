# Refactoring Needle RL

```
src
|- models: yolox, gpt, ...
|- env: ...
```

x env: simple_env et generalized_env
- logger.py, trainer.py : remplacé par `supervised_needle`, `rl_needle`, le logger peut peut-être rester en place vu que les courbes à logger sont plus ou moins les mêmes / l'eval est pareil?
- il faudra nettoyer le logger quand même
    - Nouvelle façon de gérer les metriques: le trainer fait l'eval des models au moment du test time, puis appelle le logger pour envoyer sur visdom

- reinforce à adapter pour qu'il ressemble au trainer d'avant
- callbacks à adapter pour qu'il puissent être utilisés également par reinforce

* rajouter un fichier pour gérer les metriques, plutôt que de les stocker dans le VisdomPlotter
* tests pour vérifier si le training démarre correctement
* tout déplacer dans un dossier jolineedle pour faire un vrai package python (plus tard)
* doc
