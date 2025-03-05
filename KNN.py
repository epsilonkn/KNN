import math
import random
from matplotlib import pyplot as plt

class KNN(object):

    @classmethod
    def __calcDistEuclide(cls, origin : tuple, comp : tuple) -> float:
        assert len(origin) == len(comp), "erreur de dimension dans les données fournies"
        squareTotal = 0
        for i in range(len(origin)):
            squareTotal += (origin[i] - comp[i])**2
        return math.sqrt(squareTotal)
    

    @classmethod
    def __calcDistManhattan(cls, origin : tuple, comp : tuple) -> float:
        assert len(origin) == len(comp), "erreur de dimension dans les données fournies"
        return sum(abs(a - b) for a, b in zip(origin, comp))
    

    @classmethod
    def __calcDistChebishev(cls, origin : tuple, comp : tuple) -> float:
        assert len(origin) == len(comp), "erreur de dimension dans les données fournies"
        return max(abs(a - b) for a, b in zip(origin, comp))
    

    @classmethod
    def getNeighbors(cls, test_instance : list, origine : tuple, k : int = 3, calcType : int = 0) -> list:
        """_summary_

        Args:
            test_instance (list): liste contenant les données à traiter, sous forme de tuple, le dernier élément du tuple
                                    doit être le l'identifiant de la donnée
            origine (tuple): donnée de référence dans le calcul
            k (int): nombre de plus proches voisins
            calcType (int): type de calcul à effectuer (0: euclidien, 1: manhattan, 2:Chebyshev)

        Returns:
            list: retourne les k plus proches voisins
        """
        neighbors = []
        for elt in test_instance :
            match calcType:
                case 0: 
                    dist = cls.__calcDistEuclide(origine, elt[:-1])
                case 1: 
                    dist = cls.__calcDistManhattan(origine, elt[:-1])
                case 2: 
                    dist = cls.__calcDistChebishev(origine, elt[:-1])
                case _ : 
                    raise ValueError("Type de calcul inconnu")
            neighbors.append((dist, elt[-1]))
        neighbors.sort(key=lambda x: x[0])
        return neighbors[:k]

    @classmethod
    def regression(cls, data : list[tuple]) -> float:
        mean = 0
        for elt in data :
            mean += elt[0]
        return mean / len(data) if mean > 0 else 0
        

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




liste_tuples = [(random.randint(0, 100), random.randint(0, 100), ord(random.choice("ABCDE"))) for _ in range(100)]
data = KNN.getNeighbors(liste_tuples, (50,50), 5, 1)

print(KNN.regression(data))
print(KNN.classification(data))

for point in liste_tuples:
    plt.scatter(point[0], point[1], c= 'b')
    plt.text(point[0], point[1], f"{chr(point[-1])}", fontsize=8)
plt.scatter(50, 50, c = "r")
plt.plot()
plt.show()


