# VARIABLES DEL FLUX DEL PROGRAMA INICIAL
__DATASET__ = '../datasets/Dataset1.csv'	# especifica la ruta del fitxer del dataset a buscar
__README_DATASET__ = '../datasets/README-Dataset1.txt'	# especifica la ruta del fitxer del README dataset a buscar
__COMPUTE_SILHOUETTE__ = False      # especifica si se ejecuta Silhouette
__COMPUTE_VALIDATION__ = False      # especifica si se ejecuta Validacion

# options = 'Random', '++', default='Random'
__CENTROIDS_INITIALIZATION__ = '++' 	# especifica quina inicialitzacio de centroides vols usar

# options = 'Intra-Cluster Mean', 'Intra-Cluster Weight Mean', default='Intra-Cluster Mean'
__CENTROIDS_COMPUTATION__ = 'Intra-Cluster Mean' 	# especifica quin metode de computacio dels nous centroides utilitazara el programa
__ELBOW_TEST_MAX_K__ = 8				# especifica la K maxima quan sexecuta el elbow test


# VARIABLES SOBRE LA VISUALITZACIO DELS RESULTATS
__PLOTS__ = True					# plots generics del kmeans
__DIMENSION_TO_VISUALIZE__ = [0,1]  # dimensions de les dades que es visualitzen en els plots
__PLOTS_ELBOW_TEST__ = False				# plots de cada execucio del elbow test, recomenable a False, el plot final de les inertias sempre sexecuta quan __PLOTS__ es True
__GIFS__ = True					# generacio de GIFS sobre lexecucio dels kmeans

# IMPORTS
__KMEANS__ = False					# per evitar imports ciclics