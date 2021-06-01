# AI for Insurance - How to Personalize Auto Insurance Using Telematics Data With Machine Learning on Google Cloud Platform

Many black-box insurance, also known as telematics insurance, leverage telematics data for offering cheap and flexible premium for their insurers. Based on continuously monitoring driver's behaviour and driving pattern, insurance providers can dynamically change the price. This provides opportunities for individual newly qualified yet careful drivers to demonstrate that they deserve a cheaper price than the one was set for the whole age group. In the UK, [there are 750,000 cars in the UK with black box fitted in 2017](https://www.fairerfinance.com/insights/blog/are-black-boxes-stiffling-competition-in-the-car-insurance-market). In 2016, there were only 25 telematics policies. [This number of more than doubled to 55 in 2017](https://www.thisismoney.co.uk/money/cars/article-7332163/The-number-telematics-insurance-policies-doubled-recent-years.html). The initial targeted drivers were newly qualified young drivers. Since then, black box providers are now targeting a range of drivers. 

Telematics devices allow the insurance providers to track insurers' driving through an app or by having a telematics box fitted in their vehicle. Enormous amount of data streamed by telematics devices can be analysed using cutting-edge machine learning models to help and mine insights into driver's behaviour. With the help of telematics data, owner of the vehicle can be proactively warned on abnormality before it becoming severe.

In this blog, we present an implemented solutions for predicting risk scores by analysing telematics data. We achived the following objectives:
- we designed a Google Cloud based solution architecture for training and serving machine learning on streamed telematics data streaming.
- we trained a set of machine learning models based on pre-collected [publicly available telematics data](https://www.kaggle.com/yunlevin/levin-vehicle-telematics) that are following the OBD-II standard. 
- we trained different kind of machine learning models, include supervised and unsupervised, that can demonstrate the capability of doing:
  - Vehicle predictive maintenance
  - Anormaly detection
  - Assessing risk in speed control

The solution is implemented on the following Google Cloud's services:
- Machine Learning:
  - AI Platform: a fully managed unified platform for training, serving and managing machine learning workflows.
- Data Stores:
  - Big Query: a fully-managed enterprise-level data warehouse that can be queried using SQL at scale of perabyte. In this case, we use BigQuery to store streamed telematics data for being processed and analysed.
  - Cloud Storage: a fully-managed object store. In this case, we use it to store model artefacts. 
  - Cloud SQL(SQL Server): a fully managed MS SQL Server as a backend relational database for Guidewire PolicyCenter(explained later).
- Model Hosting Services:
  - Cloud Run: a severless, fully-managed and highly scalable host environment for containerised application. In this case, we leverage Cloud Run to serve model prediction services run in docker containers. 

### Kafka

We adopt the widely used open source application messaging tool Kafka for buffering streamed-in telematics data that will later be consumed, and stored into BigQuery.

In this blog, we deployed Kafka onto GCP using [Bitnami](https://docs.bitnami.com/google/infrastructure/kafka/). Bitnami provide service for simplifying the deployment of Kafka brokers and ZooKeeper.  

### Guidewire

Guidewire is the widely used software in the P&C insurance industry. For several consecutive year to 2020, it has been recognised as a learder in Gartner Inc's [Magic Quadrant for P&C Core Platforms](https://www.guidewire.com/about-us/news-and-events/press-releases/20201111/guidewire-insurancesuite-positioned-leader-gartner). 

We have deployed Guidewire's Policy Center v10.0.3 a Windows VM hosted on Google Cloud. The Policy Center is configured to store its data in a MS SQL server database fully managed by Google Cloud.

We have integrated the machine learning prediction service into the policy quoting engine of the Policy Center so that it can query machine learning model deployed into Cloud Run containers using REST API for accessing predicted score on drivers's risk level or vehicle's maintenance history. The score can then be used as a basis for adjusting corresponding premium prices originally proposed by any Guidewire's quoting engine. 

The machine learning integrated quoting service can also be triggered remotely through SOAP API.   

### Solution Architecture 

All the functional components mentioned above is composed together to form the following architecture：

![architecture](./figs/architecture.PNG)


## Alternative Approaches:

### Cloud Pub/Sub

Google provides an alternative messaging service that is serverless, highly scalable, and fully-managed. Pub/Sub is a bundled service of Cloud IoT.

The foundamental difference between Kafka and Pub/Sub is in the way that the message delivery is handle
- Pub/Sub offers **at-least-once** message delivery and **best-effort ordering** to existing subscribers. 
- whereas [Kafka **guarantees exactly-once** delivery by default](https://kafka.apache.org/documentation/). Moreover, Kafka [guarantees](https://kafka.apache.org/documentation/#intro_guarantees) that any consumer of a given topic-partition will always read that partition's events **in exactly the same order** as they were written.

Which is better choise strictly depends on the way that downstream applications were designed for consuming and managing the buffered message. 

### Cloud IoT

In the proposed architecture, we did not include [Google Cloud IoT](https://cloud.google.com/solutions/iot), because we did not collect data from any actual telematics devices, instead from pre-collected publicly available data. Cloud IoT Core is a fully managed service on Google Cloud Platform that allows you to easily and securely connect, manage, and ingest data from millions of globally dispersed devices. 

## Sample Data Source

We trained the models using telematics data that follows the OBD-II standard. The telematics data is stored in Google Cloud's BigQuery, which constitutes of the following fields:

- DTC (diagnostic trouble codes), also known as OBDII codes, are your car's system for alerting you of vehicle issues.
- IAT is an acronym for "intake air temperature." The IAT sensor measures the air temperature that is going into your intake manifold. The colder the air, the more dense it is, and the more fuel you need to keep your engine working smoothly.
- MAP is an acronym for "manifold absolute pressure." The MAP sensor is a key component in a speed density fuel injection system, and measures pressure and absence of pressure (vacuum) at the intake manifold.
- KPL (kilometers per liter) is a unit of measurement that says how many units of distance you've travelled for each unit of fuel spent. It's simply an average of the fuel you've spent while driving your automobile, car, SUV, motorcycle, etc.
- MAF data is provided by a sensor used to determine the mass flow rate of air entering a fuel-injected internal combustion engine.    
- RPM, stands for revolutions per minute, and it's used as a measure of how fast any machine is operating at a given time
- Timing Advance (TADV) information transmitted from cell-phones, carried by the car drivers and passengers
- tPos Refers to throttle position
- eLoad, Engine load measures how much air (and fuel) you're sucking into the engine and then compares that value to the theoretical maximum.


## Kafka Setup and Config
- Broker/Producer/Consumer/Zookeeper/Simulators


## ML Modelling

### Predictive Maintenance

Maintenance can be planned and carried out in different ways, and the three common
planning paradigms are corrective, preventive and predictive maintenance ([source](https://hh.diva-portal.org/smash/get/diva2:789498/FULLTEXT01.pdf)).
Corrective maintenance takes place when the fault or failure has occurred, so it often causes unexpected downtime of the services. 
Preventive maintenance or planned maintenance on the other hand is scheduled at certain intervals regardless of the actual status of the vehicles. 
However, the interval is very difficult to be determined precisely given different vehicles can be in very different conditions due to various factors. 
Therefore, predictive maintenance tends to cause either unnecessary repairs when the intervals are too small or cause vehicles not to 
get the needed repairs in time when the intervals are too big.

Compared with corrective maintenance and planned maintenance, predictive maintenance predicts the part and time for maintenance based on the monitoring of the actual 
conditions of the vehicles. 
By doing this, unnecessary replacements in preventive maintenance and unexpected repairs in corrective maintenance can be avoided.

In this study, we trained classification models to predict the vehicles which need to be considered for maintenance using the telematics data. 
Since the original data does not have a label indicating the status of the vehicles, 
we first used clustering to do the outlier detection which is to cluster the telematics data into two groups, i.e., outliers and not outliers, 
and then use the cluster labels as the target column for training the classifiers. 
Finally, we exported the model and deployed it on Cloud Run. 

#### Clustering

For clustering, we use the Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to get the labels. 
This technique is one of the most common clustering algorithms which works based on density of object. 
It works based on two parameters: `epsilon` and `minimum samples`
- `epsilon` determines a specified radius that if includes enough number of points within
- `minimum samples` determine the minimum number of data points we want in a neighborhood to define a cluster

We dropped the columns "battery" and "tAdv" as we found these two columns are not very informative from our exploration. 
The data is shown as below

![data for clustering](figs/data_for_clustering.png)

We then use the t-SNE technique to reduce the dimensionality and visualise the clusters. 


```python
Clus_dataSet2 = cluster_dataset2.copy()
Clus_dataSet2 = np.nan_to_num(Clus_dataSet2)
Clus_dataSet2 = StandardScaler().fit_transform(Clus_dataSet2)

# t-SNE for finding clusters in high-dimensional space
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=500)
tsne_results = tsne.fit_transform(Clus_dataSet2)


df_subset['tsne-2d-one'] = tsne_results[:,0]
df_subset['tsne-2d-two'] = tsne_results[:,1]
plt.figure(figsize=(8,8))
sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    data=df_subset,
    legend="full",
    alpha=0.5
)
```

Here is the visualization of the dataset and from it, we can see several distinctive clusters. 

![cluster visualisation](figs/tsne-visualisation.png)

Then, by changing the hyper-parameter `epsilon` of the DBSCAN clustering models, we tried to find the best match which can 
label the clusters matching the visualisation.

```python
# search for the best eps for this case
min_samples= 10
for eps in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]:
    print('epsilon is: ', eps, '; minmum samples is: ', min_samples)
    dbscan = DBSCAN(eps=eps , min_samples=min_samples)
    dbscan.fit(Clus_dataSet2)
    print(set(dbscan.labels_))
    cluster_dataset2["labels"]=dbscan.labels_
    df_subset["labels"]=dbscan.labels_
    print(cluster_dataset2.labels.value_counts())
    
    
    plt.figure(figsize=(8,8))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue='labels',
        palette=sns.color_palette( n_colors=len(set(dbscan.labels_))),
        data=df_subset,
        legend="full",
        alpha=0.5
    )
    plt.show()
```

It was found that when epsilon = 3.0, the clustering results matches the visualisation of the clusters the best, which is 
shown in the figure below.

![clustering](figs/tsne-visualisation_eps_3.0.png)

Therefore, we set the cluster labels for the model with `epsilon` = 3.0 and `min_samples` = 10 as the target column for training 
the classifiers in the next step.

```python
dbscan = DBSCAN(eps=3.0 , min_samples=min_samples)
dbscan.fit(Clus_dataSet2)
cluster_dataset2["labels"]=dbscan.labels_
```

#### Classification Using the Cluster Labels

We trained our xgboost classifier using the script below. First, we imported the required packages. Then, selected the 
feature columns and target column, split the data into datasets for training and testing, and trained the classifier 
with the training dataset.


```python
try:
    import xgboost
except ImportError as ex:
    print("Error: the xgboost library is not installed.")
    xgboost = None
from sklearn.model_selection import train_test_split
from sklearn import metrics
X = cluster_dataset2.drop(['labels'], axis=1)
y = cluster_dataset2.labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3)
xgb_clr = xgboost.XGBClassifier()
xgb_clr.fit(X_train, y_train)


```

We used the trained model to make predictions for unseen test dataset. 
Then, we printed the confusion matrix and the evaluation metrics.
It was shown that the model can predict the labels for the total 3,000 data entries correctly.

![confusion matrix](figs/confusion_matrix.png)


#### Export and Deploy the Model

After we have trained the model in the above section, 
we can now export the model & serve the model from Cloud Run. We saved the model to a local model file by running the script below.

```python
xgb_clr.save_model('xgb_model.bst')
```

Then, to deploy a model on Cloud Run, we need the following files to be in one directory
- `app.py` - flask app for serving the model
- `Dockerfile` - for building Docker image according to specifics
- `requirements.txt` - a list of dependencies
- `service.yaml` - specification of the configuration of the deployment
- `xgb_model.bst`: the xgboost model file

Following these steps to deploy the model on Cloud Run
1. Make sure the current working directory is the directory containing the above-listed files. 
1. Then, we build a docker image by running 
`docker build -t gcr.io/gft-bike-insurance/predictive-maintenance:latest .`
1. After this, push the image into Google Container Registry (GCR) by running `docker push gcr.io/gft-bike-insurance/predictive-maintenance`
1. We can deploy the model image on Cloud Run using the Cloud Console or the service.yaml file and `gcloud beta run services replace` cli
1. Finally, we get the endpoint of the cloud run service ('https://predictive-maintenance-apnir7ajvq-nw.a.run.app' in this case)

After the model has been deployed on Cloud Run, we can run the following script to make requests.

```python
import subprocess
import json

endpoint = 'https://predictive-maintenance-apnir7ajvq-nw.a.run.app'
one_entry = {'cTemp': 81.0,
             'eLoad': 39.2156862745098,
             'iat': 43.0,
             'imap': 41.0,
             'kpl': 0.0,
             'maf': 2.87,
             'rpm': 791.0,
             'speed': 0.0,
             'tPos': 15.2941176470588}
output = subprocess.check_output("echo $(gcloud auth print-identity-token)", shell=True, text=True)
access_token = "Bearer " + output[:-1]

headers = {'Authorization': access_token}

response = requests.post(endpoint + '/run-predictive-maintenance', 
                         headers=headers, 
                         json=one_entry)
print(response.text)
```

The outcome was 0, meaning that this specific data entry is considered as normal so no special attention is required.

### Competitive Driving Analysis
### Abnormally Detection

## Model Deployment
- Cloud Run

## Generate Quotes from Guidewire
- Query score from Guidewire
- Generate quotes from Guidewire

## Conclusion
