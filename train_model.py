import pickle
import joblib
import pandas as pd
import seaborn as sns
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

# Load dataset
data = pd.read_csv('data/Dataset_HR.csv')
data.head(10)

# Preprocess dataset
data.isnull().sum()

# Jumlah observasi pada tiap label
data['resign'].value_counts()

# Label encoding kolom kecelakaan_kerja
data['kecelakaan_kerja'] = data['kecelakaan_kerja'].map({'tidak':0, 'pernah':1})

# Label encoding kolom promosi
data['promosi'] = data['promosi'].map({'tidak':0, 'ya':1})

# Label encoding kolom divisi
data['divisi'] = data['divisi'].map({'sales':0,
                               'accounting':1,
                               'hr':2,
                               'technical':3,
                               'support':4,
                               'management':5,
                               'IT':6,
                               'product_mng':7,
                               'marketing':8,
                               'RandD':9})

# Label encoding kolom gaji
data['gaji'] = data['gaji'].map({'low':0, 'medium':1, 'high':2})

# Label encoding kolom resign
data['resign'] = data['resign'].map({'tidak':0, 'ya':1})

# Periksa dengan 20 sampel data secara acak
data.sample(20)

# Analisa statistik deskriptif
data.describe().T

# Visualisasi heatmap dalam korelasi antar kolom data
data.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(data.corr(), annot=True)
plt.show()

# Pemilihan fitur
nama_feature = ['tingkat_kepuasan', 'lama_bekerja', 'kecelakaan_kerja', 'gaji', 'jam_kerja_perbulan']

X = data[nama_feature].values
y = data['resign'].values

# Split data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print("Data Training : ", X_train.shape, y_train.shape)
print("Data Testing : ", X_test.shape, y_test.shape)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# Modelling using Decision Tree Classifier
test_model = DecisionTreeClassifier(random_state=42)

# Hyperparameter
params = {'max_depth': list(range(2, 10)),
          'max_leaf_nodes': list(range(2, 10))}

search_best_model = GridSearchCV(test_model, params, cv=20)

# Training model
search_best_model.fit(X_train_scaled, y_train)

# Model dengan hyperparameter terbaik
search_best_model.best_estimator_

# Modelling with Random Forest
model_rf = RandomForestClassifier(max_depth=10, max_leaf_nodes=100, random_state=42)
model_rf.fit(X_train_scaled, y_train)

# Melakukan re-modeling (menyimpan model terbaik dalam variable model)
model = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=9, random_state=42)
model.fit(X_train_scaled, y_train)

# Prediksi data testing dan evaluasi model
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

y_pred_rf = model_rf.predict(X_test_scaled)
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Decision Tree')
plt.show()

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Random Forest')
plt.show()

# Visualisasi Decision Tree
plt.figure(figsize=(25, 20))
plot_tree(model, feature_names = nama_feature, class_names = ['tidak', 'iya'],filled = True)
plt.show()

importances = model_rf.feature_importances_  # For Random Forest
feature_names = ['tingkat_kepuasan', 'lama_bekerja', 'kecelakaan_kerja', 'gaji', 'jam_kerja_perbulan']
for feature, imp in zip(feature_names, importances):
    print(f"{feature}: {imp}")

# Simpan model ke dalam file
with open('models/decision_tree_model.pkl', 'wb') as f:
    pickle.dump(model, f)

joblib.dump(model_rf, 'models/random_forest_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')