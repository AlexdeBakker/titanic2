import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import folium
from streamlit_folium import st_folium
st.write('## Case 1 Titanic')
# Laad de dataset
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")  # Update dit pad indien nodig

# Toon de ruwe data
data = load_data()
st.write('## Ruwe Dataset')
st.write(data)

# Vul ontbrekende waarden in voor Leeftijd met de mediaan, Embarked met de modus, etc.
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)

# Maak extra kenmerken aan indien nodig, zoals het extraheren van 'Titel' uit 'Naam'
data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

st.write('## Schoongemaakte Dataset')
st.write(data.head(10))

# Coördinaten van de locatie waar de Titanic is gezonken
latitude = 41.7325
longitude = -49.9469

# Creëer een Folium-kaart gecentreerd rond de locatie van het zinken
m = folium.Map(location=[latitude, longitude], zoom_start=5)

# Voeg een marker toe op de locatie van het zinken van de Titanic
folium.Marker(
    [latitude, longitude],
    popup="Zinklocatie van de Titanic",
    tooltip="Titanic Zinklocatie",
    icon=folium.Icon(color="red", icon="info-sign")
).add_to(m)

with st.container():
    st.write("## Kaart met Zinklocatie van de Titanic")
    st_folium(m, width=700, height=500)


# Overleving per Klasse
st.write('## Overleving per Klasse')
fig, ax = plt.subplots()
sns.countplot(data=data, x='Pclass', hue='Survived', ax=ax)
st.pyplot(fig)

# Overleving per Geslacht
st.write('## Overleving per Geslacht')
fig, ax = plt.subplots()
sns.countplot(data=data, x='Sex', hue='Survived', ax=ax)
st.pyplot(fig)

# Leeftijdsverdeling per Overleving
st.write('## Leeftijdsverdeling per Overleving')
fig, ax = plt.subplots()
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', kde=True, ax=ax)
st.pyplot(fig)

# Boxplot van Tarief
st.write('## Boxplot van Tarief')
fig, ax = plt.subplots()
ax.boxplot([data["Fare"]])
ax.set_xticklabels(["Passagiers"])
ax.set_ylabel("Prijs per kaartje")
plt.title("Boxplot van Tarief")
st.pyplot(fig)

# Plot 1: Boxplot van Tarief per Klasse
st.write('## Tariefverdeling per Klasse')
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x='Pclass', y='Fare', data=data, ax=ax)
plt.xlabel('Klas')
plt.ylabel('Tarief')
plt.title('Tariefverdeling per Klas')
st.pyplot(fig)

# Plot 2: Scatterplot van Leeftijd vs Tarief
st.write('## Prijs per Kaartje per Leeftijd')
survived = data[data["Survived"] == 1]
not_survived = data[data["Survived"] == 0]

fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(survived["Age"], survived["Fare"], color="Orange", alpha=1, label="Overleefd")
plt.scatter(not_survived["Age"], not_survived["Fare"], alpha=0.6, label="Overleden")
plt.xlabel("Leeftijd")
plt.ylabel("Prijs per kaartje")
plt.legend()
plt.title("Prijs per kaartje per Leeftijd")

# Een gestreepte cirkel aan de plot toevoegen
circle = plt.Circle((30, 10), 18, color='red', fill=False, linewidth=2, linestyle='dashed')
plt.gca().add_patch(circle)

st.pyplot(fig)

# Voeg de eigenschap Gezinsgrootte toe
data['Fam_size'] = data['SibSp'] + data['Parch'] + 1

# Plot 3: Staafdiagram voor Overlevingspercentage per Gezinsgrootte
def bar_chart_compare(dataset, feature1, title="Vergelijkingsstaafdiagram"):
    plt.figure()
    plt.title(title)
    sns.barplot(x=feature1, y='Survived', data=dataset, errorbar=None, color='blue').set_ylabel('Overlevingspercentage')
    st.pyplot(plt.gcf())

bar_chart_compare(data, "Fam_size", title="Overlevingspercentage per Gezinsgrootte")

# Bereid kenmerken en doel voor
X = data[['Pclass', 'Age', 'Fare', 'SibSp', 'Parch']].fillna(0)  # Basiskenmerken als voorbeeld
y = data['Survived']

# Splits de data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train een model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Maak voorspellingen
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write('## Modelnauwkeurigheid')
st.write(f'Nauwkeurigheid van het RandomForest-model: {accuracy:.2f}')

# Sta gebruikersinvoer voor voorspelling toe
st.write('## Voorspel Overleving')
pclass = st.selectbox('Passagiersklasse', [1, 2, 3])
age = st.slider('Leeftijd', 0, 80, 25)
fare = st.slider('Tarief', 0, 500, 50)
sibsp = st.number_input('Broers/Zussen/Partners aan boord', 0, 8, 0)
parch = st.number_input('Ouders/Kinderen aan boord', 0, 6, 0)

# Voorspelling
input_data = pd.DataFrame({'Pclass': [pclass], 'Age': [age], 'Fare': [fare], 'SibSp': [sibsp], 'Parch': [parch]})
prediction = model.predict(input_data)

if prediction == 1:
    st.write('Voorspelling: Overleefd')
else:
    st.write('Voorspelling: Niet overleefd')
 