### Telematics Car Insurance on Google Cloud with Machine Learning

Many black-box insurance providers leverage telematics data for offering cheap and flexible premium for their insurers. 

Based on continuously monitoring driver's behaviour and driving pattern, insurance providers can dynamically change the price so that it can be more affordable for a wider population. This give opportunity for newly qualified yet careful drivers to demonstrate the risk level of their driving, which deserves for lower premium price.  

In the UK, [there are 750,000 cars in the UK with black box fitted in 2017](https://www.fairerfinance.com/insights/blog/are-black-boxes-stiffling-competition-in-the-car-insurance-market). In 2016, [there were only 25 telematics policies. This number of more than doubled to 55 in 2017](https://www.thisismoney.co.uk/money/cars/article-7332163/The-number-telematics-insurance-policies-doubled-recent-years.html).

The initial targeted drivers are newly qualified young drivers. Since then, black box providers are now targeting a ranger of drivers. Telematics devices allow the insurance providers to track insurers' driving through an app or by having a black box fitted in their vehicle. The insurer will then usually offer cheaper insurance for careful driving and the policyholder is considered as a lower risk driver. 

Enormous amount of data streamed by telematics devices can be analysed using cutting-edge machine learning models to help and mine insights into driver's behaviour. With the help of telematics data, owner of the vehicle can be proactively warned on abnormality before it becoming severe.

In this blog, we achived the following objectives:
- we designed a Google Cloud based solution architecture for training and serving machine learning on streamed telematics data streaming.
- we trained the machine learning models using telematics data that are following the OBD-II standard. 
- we trained machine models for:
  - Vehicle predictive maintenance
  - Anormaly detection
  - Assessing risk in speed control

The solution is implemented on the following Google Cloud's services:
- Machine Learning:
  - AI Platform: a fully managed unified platform for training, serving, managing machine learning workflows.
- Data Stores:
  - Big Query: a fully-managed enterprise-level data warehouse. Data can be queried using SQL at scale of perabyte. In this case, we use BQ query to store streamed(simulated) in telematics data.
  - Cloud Storage: a fully-managed object store. In this case, model artefacts. 
- Model Hosting Services:
  - Cloud Run: a severless, fully-managed and highly scalable host environment for containerised application. In this case, we leverage Cloud Run to serve model prediction services run in docker containers. 

### Kafka

We also adopt the widely used open source application messaging tool Kafka for consuming and buffering streamed-in telematics data.

In this blog, we deployed Kafka onto GCP using [Bitnami](https://docs.bitnami.com/google/infrastructure/kafka/). Bitnami provide service for simplifying the deployment of Kafka brokers and ZooKeeper.  

### Cloud IoT

In the proposed architecture, we do not include [Google Cloud IoT](https://cloud.google.com/solutions/iot), because we do not collect data from actual telematics devices. Cloud IoT Core is a fully managed service on Google Cloud Platform that allows you to easily and securely connect, manage, and ingest data from millions of globally dispersed devices. 


### Cloud Pub/Sub

Google provides an alternative messaging service that is fully-managed, serverless, highly scalable, and  native to the Google Cloud. Pub/Sub is a bundled service of Cloud IoT.

The foundamental difference between Kafka and Pub/Sub is in the way that the message delivery is handle
- Pub/Sub offers *at-least-once* message delivery and *best-effort ordering* to existing subscribers. 
- whereas [Kafka *guarantees exactly-once* delivery by default](https://kafka.apache.org/documentation/). Moreover, Kafka [guarantees](https://kafka.apache.org/documentation/#intro_guarantees) that any consumer of a given topic-partition will always read that partition's events *in exactly the same order* as they were written.

Which is better choise strictly depends on the way how message is consumed and managed by downstream application. 

###### Architecture: GCP Components



###### Sample Data Source

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



###### Kafka Setup and Config
Broker/Producer/Consumer/Zookeeper/Simulators


###### ML Modelling:
Predictive Maintenance
Competitive Driving Analysis
Abnormaly Detection

###### Model Deployment
Cloud Run

###### Query score from Guidewire
Guidewire

###### Conclusion