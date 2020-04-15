from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency



species = pd.read_csv('species_info.csv')
print(species.head())
print(species.dtypes)

species.scientific_name.nunique()
#There are 5541 unique species in the dataset

species.conservation_status.unique()
#The conservation statuses include: Species of Concern, Endangered, Threatened, None

species.fillna('No Intervention', inplace=True)
species.conservation_status.value_counts()
#3.2% of the species population have a conservation status and the majority are 'Species of Concern' with 2.7%

protection_counts = species.groupby('conservation_status').scientific_name.nunique().reset_index().sort_values(by='scientific_name')

#plotting bar chart
plt.figure(figsize=(10,4))
ax = plt.subplot()
plt.bar(range(len(protection_counts)), protection_counts.scientific_name.values)
ax.set_xticks(range(len(protection_counts)))
ax.set_xticklabels(protection_counts.conservation_status)
plt.ylabel('Number of Species')
plt.title('Conservation Status by Species')
plt.show()

#Are certain types of species more likely to be endangered?
species['is_protected'] = species.conservation_status != 'No Intervention'
category_pivot = category_counts.pivot(columns = 'is_protected', index = 'category', values = 'scientific_name').reset_index()
category_pivot.columns = ['category', 'not_protected', 'protected']
category_pivot['percent_protected'] = (100 * category_pivot.protected / (category_pivot.not_protected + category_pivot.protected)).round(2).astype(str) + '%'
category_pivot
#Mammals and Birds seam to be most likely to have an endangered species

#Significance test: Are mammals more likely to be endangered than birds? -> chi2
contingency = [[30, 146], [75, 413]]
tstat, pval, f, freq = chi2_contingency(contingency)
print(pval)
#No significant difference that mammals are more likely to be endangered than birds

#Reptile vs. Mammal?
contingency2 = [[30, 146], [5, 73]]
tstat, pval, f, freq = chi2_contingency(contingency2)
print(pval)


# Entering new observations with observations.csv
observations = pd.read_csv('observations.csv')
print(observations.head())

#Analyse sheep population
species['is_sheep'] = species.common_names.apply(lambda x: 'Sheep' in x)
sheep_species = species[species.is_sheep & (species.category == 'Mammal')]
sheep_observations = observations.merge(sheep_species)
obs_by_park = sheep_observations.groupby('park_name').observations.sum().reset_index()
obs_by_park


#Visualise with bar chart
plt.figure(figsize=(16, 4))
ax = plt.subplot()
plt.bar(range(len(obs_by_park)),
        obs_by_park.observations.values)
ax.set_xticks(range(len(obs_by_park)))
ax.set_xticklabels(obs_by_park.park_name.values)
plt.ylabel('Number of Observations')
plt.title('Observations of Sheep per Week')
plt.show()
#Yellowstone has the largest amount of sightings for sheep, Great Smoky Mountains the lowest