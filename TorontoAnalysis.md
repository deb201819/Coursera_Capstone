
# Segmenting and Clustering Neighborhoods in Toronto

##### Thank You for reviewing my work!
##### This Notebook is for Answer 1 of:
##### Peer-graded Assignment: Segmenting and Clustering Neighborhoods in Toronto


```python
import pandas as pd
from bs4 import BeautifulSoup
import requests

from geopy.geocoders import Nominatim

import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans

import folium #map rendering library
```

#### Scrap List of postal codes of Canada wiki page content by using BeautifulSoup


```python
source = requests.get('https://en.wikipedia.org/wiki/List_of_postal_codes_of_Canada:_M').text
soup = BeautifulSoup(source,'lxml')
```

#### Find the first table on the Wikipedia page and iterate through tags for required information


```python
table_can_zipinfo = soup.find('table')
colvals = table_can_zipinfo.find_all('td')

elem_cnt = len(colvals)

postcode = []
borough = []
neighborhood = []

for i in range(0, elem_cnt, 3):
    postcode.append(colvals[i].text.strip())
    borough.append(colvals[i+1].text.strip())
    neighborhood.append(colvals[i+2].text.strip())
```

#### Build the dataframe from the list of values


```python
df_can_postcode = pd.DataFrame(data=[postcode, borough, neighborhood]).transpose()
df_can_postcode.columns = ['Postcode', 'Borough', 'Neighborhood']
```

#### Cleansing and Transforming the data
###### 1. Only process the cells that have an assigned borough. Ignore cells with a borough that is Not assigned
###### 2. If a cell has a borough but a Not assigned neighborhood, then the neighborhood will be the same as the borough


```python
df_can_postcode.drop(df_can_postcode[df_can_postcode['Borough'] == 'Not assigned'].index, inplace=True)
df_can_postcode.loc[df_can_postcode.Neighborhood == 'Not assigned', "Neighborhood"] = df_can_postcode.Borough
df_can_postcode.reset_index(drop=True,inplace=True)
```

#### Visualizing first few rows of the dataframe 


```python
df_can_postcode.head(12)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Postcode</th>
      <th>Borough</th>
      <th>Neighborhood</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M7A</td>
      <td>Queen's Park</td>
      <td>Queen's Park</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M9A</td>
      <td>Etobicoke</td>
      <td>Islington Avenue</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills North</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M4B</td>
      <td>East York</td>
      <td>Woodbine Gardens</td>
    </tr>
  </tbody>
</table>
</div>



#### Printing the number of rows of your dataframe


```python
df_can_postcode.shape
```




    (212, 3)






## Thank You for reviewing my work!
