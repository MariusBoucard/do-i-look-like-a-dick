import re
import urllib.request



# Ouvrir le fichier texte
with open('TDPDNE3.html', 'r') as f:
    # Lire le contenu du fichier
    content = f.read()

    # Utiliser une expression régulière pour trouver toutes les URLs
    urls = re.findall(r'background:\s*url\(&quot;(.+?)&quot;\)', content)

    # Afficher les URLs trouvées
    print(urls)
    for url in urls :

# URL de l'image à télécharger
        url2 = 'https://www.thisdickpicdoesnotexist.com/'+url

        # Nom de fichier local pour enregistrer l'image
        filename = url.split("/")
        filename = filename[1]
        # Télécharger l'image et enregistrer dans un fichier local
        urllib.request.urlretrieve(url2,"dicks/"+filename)