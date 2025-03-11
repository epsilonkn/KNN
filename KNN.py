import math
import random
from matplotlib import pyplot as plt
from sklearn.datasets import fetch_openml
import numpy as np


class KNN(object):

    assert_text = "erreur de dimension dans les données fournies"


    @classmethod
    def calcDistEuclide(cls, origin : tuple, comp : tuple) -> float:
        try :
            assert len(origin) == len(comp), cls.assert_text
            square = 0
            for i in range(len(origin)):
                square += (origin[i] - comp[i])**2
            return math.sqrt(square)
        except AssertionError:
            pass
    

    @classmethod
    def calcDistManhattan(cls, origin : tuple, comp : tuple) -> float:
        try :
            assert len(origin) == len(comp), cls.assert_text
            return sum(abs(a - b) for a, b in zip(origin, comp))
        except AssertionError:
            pass
    

    @classmethod
    def calcDistChebishev(cls, origin : tuple, comp : tuple) -> float:
        try :
            assert len(origin) == len(comp), cls.assert_text
            return max(abs(a - b) for a, b in zip(origin, comp))
        except AssertionError:
            pass



class KNNClassification(KNN):

    

    @classmethod
    def getNeighbors(cls, dataset : list, tags : list, origine : tuple, k : int = 3, calcType : int = 0) -> list:
        """_summary_

        Args:
            dataset (list): liste contenant les données à traiter, sous forme de tuple, le dernier élément du tuple
                                    doit être l'identifiant de la donnée
            tags (list): liste contenant les ID de chaque donnée du dataset
            origine (tuple): donnée de référence dans le calcul
            k (int): nombre de plus proches voisins
            calcType (int): type de calcul à effectuer (0: euclidien, 1: manhattan, 2:Chebyshev)

        Returns:
            list: retourne les k plus proches voisins
        """
        neighbors = []
        distances = []
        for elt in dataset :
            match calcType:
                case 0: 
                    dist = cls.calcDistEuclide(origine, elt)
                case 1: 
                    dist = cls.calcDistManhattan(origine, elt)
                case 2: 
                    dist = cls.calcDistChebishev(origine, elt)
                case _ : 
                    raise ValueError("Type de calcul inconnu")
            distances.append(dist)
        neighbors = list(zip(distances, tags))
        neighbors.sort(key=lambda x: x[0])
        return neighbors[:k]
        

    @classmethod
    def classification(cls, data : list[tuple]) -> list:
        votes = {}
        for elt in data :
            if elt[1] in votes :
                votes[elt[1]] += 1
            else :
                votes[elt[1]] = 1
        max_votes = max(votes.values())
        return [chr(k) for k, v in votes.items() if v == max_votes]
    


class KNNRegression(KNN):

    @classmethod
    def __updateCentroids(cls, centroid, data) :
        newCentroids = []
        for i in range(len(centroid)) :
            newCentroids.append((centroid[i] + sum([j[i] for j in data]))/(len(data) + 1))
        return newCentroids


    def __getCentroids(data : list, k : int) -> list :
        centroides = []
        while len(centroides) < k :
            elt = data[random.randint(0, len(data))]
            if elt not in centroides :
                centroides.append(elt)
        return centroides


    @classmethod
    def regression(cls, test_instance : list, k : int = 3, calcType : int = 0, graph : bool = False, clusters : list = None)  -> list :
        """regression method of the KNN machine learning algorithm

        Args:
            test_instance (list): dataset, must be a list of : list, tuples

            k (int, optional) : number of clusters in the regression, Defaults to 3

            clusters (list, optional) : argument to give defined clusters to the algorithm, Defaults to None

            calcType (int, optional) : way of calculation of the distances in the algorithm -> 0 = euclidian distance, 1 = manhattan distance, 2 = chebyshev distance, Defaults to 0

            graph (bool, optional) : creation of a matplotlib graph if the dimension of the data <= 2, Defaults to False

        Returns:
            list: returns the list of the positions of each cluster center
        """

        assert not(len(test_instance[0]) > 2 and graph == True), "Impossible de générer un graph de dimension supérieure à 2"

        #initialisation des clusters
        centroides = cls.__getCentroids(test_instance, k) if clusters == None else clusters
        centroidesWeights = [[] for _ in range(k)]

        for points in test_instance : 
            if graph == True : 
                plt.scatter(points[0], points[1], c= 'b')
            #calcul de la distance entre le point et les clusters
            match calcType:
                case 0: 
                   distances = [(cls.calcDistEuclide(centr, points), centr) for centr in centroides]
                case 1: 
                    distances = [(cls.calcDistManhattan(centr, points), centr) for centr in centroides]
                case 2: 
                    distances = [(cls.calcDistChebishev(centr, points), centr) for centr in centroides]
                case _ : 
                    raise ValueError("Type de calcul inconnu")
            #recherche du cluster le plus proche et mise à jour de son centre moyen
            distances.sort(key=lambda x: x[0])
            centroidesWeights[centroides.index(distances[0][1])].append(points)
            centroides[centroides.index(distances[0][1])] = cls.__updateCentroids(centroides[centroides.index(distances[0][1])], centroidesWeights[centroides.index(distances[0][1])])

        if graph == True : 
            for centr in centroides :
                plt.scatter(centr[0], centr[1], c= 'r')
            plt.plot()
            plt.show()
        return centroides




if __name__ == "__main__":

    #Exemple de fonctionnement du Kmeans, avec regression en clusters des chiffres, et utilisation du KNN pour trouver le cluster 
    #le plus proche de chaque nouveau chiffre donné

    #taille du jeu de données utilisé pour entrainer le modèle
    IMAGE_NUMBER = 1000
    #taille du jeu de données utilisé pour tester le modèle
    TRAINING_SET_SIZE = 1000


    #fonction de calcul de la précision du modèle
    def getAccuracy(clusters, nums, test_set, nums_set):
        correct = 0
        for i in range(len(test_set)):
            nbr = test_set[i]
            prediction = KNNClassification.getNeighbors(clusters, nums, nbr, k = 1)[0]
            if prediction[1] == nums_set[i]:
                correct += 1

        return correct * 100 / len(test_set)
    

    #on récupère les données
    mnist = fetch_openml("mnist_784") 
    mnist.target = mnist.target.astype(np.int8)

    x = np.array(mnist.data)
    y = np.array(mnist.target)

    #crée des sets sous forme de listes 
    listeArray = []
    for arrays in x[:IMAGE_NUMBER] : 
        listeArray.append(arrays.tolist())

    testArray = []
    for arrays in x[IMAGE_NUMBER:(IMAGE_NUMBER+TRAINING_SET_SIZE)] : 
        testArray.append(arrays.tolist())

    incr = 0
    clusters = []
    nums = []
    while len(clusters) < 10 :
        if y[incr] not in nums :
            clusters.append(x[incr].tolist())
            nums.append(int(y[incr]))
        incr += 1


    #application de la régression
    clusters = KNNRegression.regression(listeArray[:IMAGE_NUMBER], k = 10, clusters= clusters)

    #chiffrage de la précision
    print(f"précision du modèle : {getAccuracy(clusters, nums, testArray, y[IMAGE_NUMBER:(IMAGE_NUMBER+TRAINING_SET_SIZE)])} %")
    print(f"nombre d'images traitées pour entrainer le modèle : {IMAGE_NUMBER}")
    print(f"nombre d'images traitées pour tester le modèle : {TRAINING_SET_SIZE}")



