from model.som import SOM

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def main():
    data = pd.read_csv('data/rfm.csv')
    train_data = data.drop(['CustomerID'], axis=1)
    train_data = train_data.values

    som_model = SOM(map_size=(30, 30), n_features=train_data.shape[1], learning_rate=0.5)
    som_model.train(train_data, n_epochs=10)

    som_model.save_weights('data/som_weights.npy')

    res = []

    for i in range(len(data)):
        x = data.loc[i, ['Recency', 'Frequency', 'Monetary']].values
        cluster = som_model.predict(x)
        res.append((data.loc[i, 'CustomerID'], cluster))

    # Encode clusters
    encoder = LabelEncoder()
    clusters = [(customer_id, cluster[1]) for customer_id, cluster in res]
    clusters = pd.DataFrame(clusters, columns=['CustomerID', 'Cluster'])
    clusters['Cluster'] = encoder.fit_transform(clusters['Cluster'])
    print('We have {} clusters'.format(len(clusters['Cluster'].unique())))
    clusters.to_csv('data/clusters.csv', index=False)


if __name__ == '__main__':
    main()
