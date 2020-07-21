import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict


def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

class KMeans:
    # initialize and fit
    def __init__(self, data):
        self.data = data


    def initialize(self, k):
        self.centroids = np.random.normal(size=(k, 2))
        #print(self.centroids,"CENTROIDS")

    # Find the current cluster each data belongs to
    def assignPoints2Clusters(self):
        print("TRUE")
        condition = True

        current_centroids = self.centroids
        count = 0 
        while condition:
            print(current_centroids)
            dic = defaultdict(list)
        # assign each point to its closest centroid
            for i in range(len(self.data)):

                minimum = np.float('inf')

                for j in range(len(current_centroids)):

                    if euclidean_distance(current_centroids[j],self.data[i]) < minimum:

                        min_cent_index = j

                dic[min_cent_index].append(self.data[i])


            #print(dic,"DICTIONARY")
            print(dic.keys(),"KEYS")
        # update the centroid
            updated_centroids = []
            
            for i in range(len(current_centroids)):
                if dic[i] == []:
                    temp = current_centroids[i]
                else:
                    temp = np.sum(dic[i],axis=0)/len(dic[i])
                    temp = list(temp)
                    #print(temp,"TEMP")

                updated_centroids.append(temp)
                
            changed = 0
            for i in range(len(updated_centroids)):
                if updated_centroids[i][0] != current_centroids[i][0] or updated_centroids[i][1] != current_centroids[i][1]:
                    changed = changed + 1 

            #print(current_centroids,"CURRENT")
            #print(updated_centroids,"UPDATED")
            if changed == 0:
                condition = False
            else:
                current_centroids = updated_centroids

            count = count + 1

        print(count, "COUNT \n\n\n\n")


        

            
class DataGeneration:
    def ambers_random_data(self):
        np.random.seed(1)
        x = 2
        data1 = np.random.normal(size=(100, 2)) + [ x, x]
        data2 = np.random.normal(size=(100, 2)) + [ x,-x]
        data3 = np.random.normal(size=(100, 2)) + [-x,-x]
        data4 = np.random.normal(size=(100, 2)) + [-x, x]
        data  = np.concatenate((data1, data2, data3, data4))
        np.random.shuffle(data)
        return data

    def data(self):
        np.random.seed(0)
        data = np.random.normal(size=(400,2))
        return data

#plots
    #def elbow plot:
        #ax.set_xlabel("Number of Clusters")
        #ax.set_ylabel("Distortion")
        #print("Showing the elbow plot. Close the plot window to continue.")
        #plt.show()
    #def kmeans cluster plot: 
        #get num of clusters
        #ax.scatter(
        #ax.legend()
        #plt.show()
if __name__ == "__main__":
    generator = DataGeneration()
    #data = generator.ambers_random_data()
    #print(data.shape)
    #print(np.max(data,axis=0))
    #print(np.min(data,axis=0))
    #myPoints = generator.data()
    #print(myPoints)
    #plt.scatter(data[:,0], data[:,1])
    #plt.show()
    k_means = KMeans(generator.ambers_random_data())
    #print(k_means)
    #k_means.elbow()
    k = input("Choose number of clusters: ")
    k_means.initialize(int(k))
    # updating the clusters 
    k_means.assignPoints2Clusters()
    #k_means.fit()
    #k_means.plot()
