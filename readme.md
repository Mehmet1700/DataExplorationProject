# Brain Tumor Detection mit Convolutional Neural Network (CNN)

## Projektrahmen
Dieses Projekt wurde im Rahmen der Vorlesung Data Exploration Project an der DHBW Stuttgart im Studiengang Wirtschaftsinformatik erstellt. Es befasst sich mit der Entwicklung eines Modells zur Erkennung von Gehirntumoren in Bildern mithilfe eines Convolutional Neural Network (CNN). 

Immatrikulationsnummer: 9948958

## Problemstellung
Das Ziel dieses Projekts ist es, ein Modell zu entwickeln, das die Fähigkeit besitzt, Gehirntumore in Bildern zu erkennen. Dies kann dazu beitragen, medizinische Diagnosen zu unterstützen und die Genauigkeit von Untersuchungen zu verbessern.

## Daten
Die Daten stammen aus einem Kaggle-Datensatz, der Bilder von Gehirntumoren und die entsprechenden Klassenzuordnungen enthält. Für unsere Zwecke sind nur die Bilder und ihre Klassen (Tumor oder kein Tumor) relevant. Die Bilder sind in einem separaten Ordner abgelegt.

## Lösung
Wir verwenden ein Convolutional Neural Network (CNN), das speziell für die Verarbeitung von Bildern geeignet ist. Das Modell wird auf den Trainingsdaten trainiert und auf den Testdaten evaluiert, um die Leistung zu bewerten.

## Software
Die Implementierung erfolgt in Python unter Verwendung der folgenden Bibliotheken:
- pandas und numpy für die Datenverarbeitung
- matplotlib für die Visualisierung
- TensorFlow und Keras für das Deep Learning
- MLflow für das Tracking des Machine Learning Lifecycle

## Schritte

1. **Datenexploration und Vorbereitung**
   - Analyse der Daten, einschließlich Klassenverteilung und Datenintegrität.
   - Aufteilung der Daten in Trainings-, Validierungs- und Testdaten.
   - Anpassung der Klassenverteilung durch Oversampling und Erstellen von Bildbatches.

2. **Modellentwicklung**
   - Verwendung des MobileNetV2-Modells als Basis für das CNN.
   - Übertragung des gelernten Modells mit Anpassung der letzten Schicht.
   - Kompilierung des Modells mit Optimierer, Verlustfunktion und Metriken.
   - Training des Modells unter Verwendung der Trainings- und Validierungsdaten.

3. **Modellbewertung und Hyperparameter-Tuning**
   - Bewertung der Modellleistung auf den Testdaten mit Metriken wie Genauigkeit, Präzision, Recall und F1-Score.
   - Durchführung des Hyperparameter-Tunings zur Optimierung der Modellleistung.

4. **MLflow-Integration**
   - Verwendung von MLflow zur Verfolgung von Experimenten und zur Verwaltung von Modellen.
   - Protokollierung von Hyperparametern, Metriken und Modellen während des Trainings und Hyperparameter-Tunings.

## Anwendung
Das trainierte Modell kann verwendet werden, um neue Bilder von Gehirntumoren zu analysieren und Vorhersagen über das Vorhandensein eines Tumors zu treffen. Dies kann Ärzten bei der Diagnose und Behandlung von Patienten unterstützen.

## Quellen

https://www.datacamp.com/tutorial/cnn-tensorflow-python

https://www.kaggle.com/code/leekahwin/brain-tumor-detection-from-mri-images-deep-cn

https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor/data?select=Brain+Tumor.csv

https://mlflow.org/docs/latest/getting-started/intro-quickstart/notebooks/tracking_quickstart.html


