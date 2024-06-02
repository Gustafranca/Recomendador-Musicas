import pandas as pd
import numpy as np
#importando data
dados = pd.read_csv("../data/Dados_totais.csv")
dados_generos =  pd.read_csv("../data/data_by_genres.csv")
dados_anos = pd.read_csv("../data/data_by_year.csv")

#Removendo colunas que não influenciaram 
dados.drop(['explicit','key','mode'],axis=1)

dados_generos.drop(['key','mode'],axis=1)

#Recomendando musicas pós anos 2000
dados_anos = dados_anos[dados_anos["year"]>=2000]
dados_anos.drop(['mode','key'],axis=1)

dados_generos = dados_generos.drop(["mode"], axis=1)
dados_generos1 = dados_generos.drop(['genres'],axis=1)


#importando biblioteca e metods do sklearning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

SEED = 1224
np.random.seed(1224)
#(Pipeline) Arranjo de dados ///(StandardScaler) Padronizando os dados /// (PCA) reduz a dimensionalidade do cluster
pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=2, random_state=SEED))])

genre_embedding_pca = pca_pipeline.fit_transform(dados_generos1)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding_pca)

from sklearn.cluster import KMeans
# calculo das disntancias entre elementos
kmeans_pca = KMeans(n_clusters=5, verbose=False, random_state=SEED)

kmeans_pca.fit(projection)

dados_generos['cluster_pca'] = kmeans_pca.predict(projection)
projection['cluster_pca'] = kmeans_pca.predict(projection)

projection['generos'] = dados_generos['genres']

from sklearn.preprocessing import OneHotEncoder
#Usando dummies 
ohe = OneHotEncoder(dtype=int)
colunas_ohe = ohe.fit_transform(dados[['artists']]).toarray()
dados2 = dados.drop('artists', axis=1)

dados_musicas_dummies = pd.concat([dados2, pd.DataFrame(colunas_ohe, columns=ohe.get_feature_names_out(['artists']))], axis=1)
dados_musicas_dummies

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', PCA(n_components=0.7, random_state=SEED))])


music_embedding_pca = pca_pipeline.fit_transform(dados_musicas_dummies.drop(['id','name','artists_song'], axis=1))
projection_m = pd.DataFrame(data=music_embedding_pca)

kmeans_pca_pipeline = KMeans(n_clusters=50, verbose=False, random_state=SEED)

kmeans_pca_pipeline.fit(projection_m)

dados['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)
projection_m['cluster_pca'] = kmeans_pca_pipeline.predict(projection_m)


projection_m['artist'] = dados['artists']
projection_m['song'] = dados['artists_song']

def Recomendador_Musica():
    from sklearn.metrics.pairwise import euclidean_distances
    nome_musica = str(input("Qual música você acabou de ouvir: "))
    try:
    #Buscando elemetos de menor distância do nosso parametro
        cluster = list(projection_m[projection_m['song']== nome_musica]['cluster_pca'])[0]
        
        musicas_recomendadas = projection_m[projection_m['cluster_pca']== cluster][[0, 1, 'song']]
        x_musica = list(projection_m[projection_m['song']== nome_musica][0])[0]
        y_musica = list(projection_m[projection_m['song']== nome_musica][1])[0]

        #distâncias euclidianas
        distancias = euclidean_distances(musicas_recomendadas[[0, 1]], [[x_musica, y_musica]])
        musicas_recomendadas['id'] = dados['id']
        musicas_recomendadas['distancias']= distancias
        recomendada = musicas_recomendadas.sort_values('distancias').head(10)
        
        musicas_validas = recomendada['song'].to_string(index=False)
        
        return musicas_validas
    except IndexError:
        print("The song isn't in our dataset, please try again.")
        return None



ans = Recomendador_Musica()

if ans is not None:
    print("\nHere are some recomendations:")
    print(ans)


