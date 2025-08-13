# init
references.bib (exporter librairie zotero)
journal.csl (template citation du journal)

cmb shit B pour buils paper + supplements

figures in /figures

# GIT
## first-time (inside existing folder)
git init
git remote add origin git@github.com:garind-888/camilla-manuscript.git
git add .
git commit -m "first commit"
git branch -M main
git push -u origin main
git config --global alias.sync '!git add . && git commit -m "sync" && git push'

## daily
git sync      # custom alias pulls, commits everything, pushes

## update from others
git reset --hard && git pull




