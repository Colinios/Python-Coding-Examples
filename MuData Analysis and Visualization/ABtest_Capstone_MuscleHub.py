### MuscleHub AB Test - Capstone Project ###

#Inspecting and importing SQL databases
import sql_query

sql_query('''
SELECT *
FROM visits
LIMIT 5
''')

sql_query('''
SELECT *
FROM fitness_tests
LIMIT 5
''')

sql_query('''
SELECT *
FROM applications
LIMIT 5
''')

sql_query('''
SELECT *
FROM purchases
LIMIT 5
''')

df = sql_query('''
SELECT visits.first_name, visits.last_name, visits.gender, visits.email, visits.visit_date, fitness_tests.fitness_test_date, applications.application_date, purchases.purchase_date
FROM visits
LEFT JOIN fitness_tests
ON fitness_tests.first_name = visits.first_name
AND fitness_tests.last_name = visits.last_name
AND fitness_tests.email = visits.email
LEFT JOIN applications
ON applications.first_name = visits.first_name
AND applications.last_name = visits.last_name
AND applications.email = visits.email
LEFT JOIN purchases
ON purchases.first_name = visits.first_name
AND purchases.last_name = visits.last_name
AND purchases.email = visits.email
WHERE visits.visit_date >= '7-1-17'
''')

## Analysing the A and B groups with Python
import pandas as pd
from matplotlib import pyplot as plt

df['ab_test_group'] = df.fitness_test_date.apply(lambda x: 'A' if pd.notnull(x) else 'B')
ab_counts = df.groupby('ab_test_group').first_name.count().reset_index()
ab_counts

plt.pie(ab_counts.first_name.values, labels = ['A', 'B'], autopct = '%d%%', shadow = True, colors = ['darkred', 'darkgreen'])
plt.axis('equal')
plt.show()
plt.savefig('ab_test_pie_chart.png')
#Both A and B test groups seem to be around the same size

#How many of each group applied?
df['is_application'] = df.application_date.apply(lambda x: 'Application' if pd.notnull(x) else 'No Application')
app_counts = df.groupby(['ab_test_group', 'is_application']).first_name.count().reset_index()
app_counts

app_pivot = app_counts.pivot(columns = 'is_application', index = 'ab_test_group', values = 'first_name').reset_index()

app_pivot['Total'] = app_pivot['Application'] + app_pivot['No Application']
app_pivot['Percent with Application'] = (100 * app_pivot['Application'] / app_pivot['Total']).round(1).astype(str) + '%'
app_pivot
#Test group B seems more likely to file an application with 13% (vs. 10%)
#Is this a significant difference? Test with Chi2 (95% significance):
from scipy.stats import chi2_contingency
contingency = [[250, 2254], [325, 2175]]
tstat, pval, f, freq = chi2_contingency(contingency)
pval
#Hypothesis rejected, difference between A and B group is significant. Applications are more likely without a fitness test at the beginning.

#How many purchased a membership after applying? And what kind of customers are they?
df['is_member'] = df.purchase_date.apply(lambda x: 'Member' if pd.notnull(x) else 'Not Member')
just_apps = df[df.is_application == 'Application']

member = just_apps.groupby(['ab_test_group', 'is_member']).first_name.count().reset_index()
member_pivot = member.pivot(columns = 'is_member', index = 'ab_test_group', values = 'first_name').reset_index()
member_pivot['Total'] = member_pivot['Member'] + member_pivot['Not Member']
member_pivot['Percent Purchase'] = (100* member_pivot['Member'] / member_pivot['Total']).round(1).astype(str) + '%'
member_pivot

#80% of Group A (with fitness test at beginning) purchased a membership, vs. 77% of Group B (without the test)
#Are the two results significantly different?	Chi2 test (95% significance):
contingency = [[200, 2304], [250, 2250]]
tstat, pval, f, freq = chi2_contingency(contingency)
pval
#Not significantly different: There is no effect on purchases of memberships if applicants conducted a fitness test or not

#More important, what percent of all visitors purchased a membership?
final_member = df.groupby(['is_member', 'ab_test_group']).first_name.count().reset_index()
final_member_pivot = final_member.pivot(columns = 'is_member', index = 'ab_test_group', values = 'first_name').reset_index()
final_member_pivot['Total'] = final_member_pivot['Member'] + final_member_pivot['Not Member']
final_member_pivot['Percent Purchase'] = (100* final_member_pivot['Member'] / final_member_pivot['Total']).round(1).astype(str) + '%'
final_member_pivot
#More purchases by Group B, is it significantly different?	Chi2 test (95% significance):
contingency = [[200, 2304], [250, 2250]]
tstat, pval, f, freq = chi2_contingency(contingency)
pval
#Yes! Group B (without test) purchased significantly more memberships in the end. 
#Therefore MuscleHub fitness studio should abolish the introductory fitness test, and perhaps make it voluntary

