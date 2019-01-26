
# Segmenting and Clustering Neighborhoods in Toronto

##### Thank You for reviewing my work!
##### This Notebook is for Answer 3 of:
##### Peer-graded Assignment: Segmenting and Clustering Neighborhoods in Toronto


```python
import pandas as pd
from bs4 import BeautifulSoup
import requests

import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.cluster import KMeans

import numpy as np # library to handle data in a vectorized manner

import json # library to handle JSON files

from geopy.geocoders import Nominatim # convert an address into latitude and longitude values

import requests # library to handle requests
from pandas.io.json import json_normalize # tranform JSON file into a pandas dataframe

# Matplotlib and associated plotting modules
import matplotlib.cm as cm
import matplotlib.colors as colors

# import k-means from clustering stage
from sklearn.cluster import KMeans

import folium # map rendering library

print('Libraries imported.')
```

    Libraries imported.


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



#### Read the Geospatial csv file and inner join it with original dataframe


```python
df_latlong = pd.read_csv('http://cocl.us/Geospatial_data')
df_latlong.columns = ['Postcode', 'Latitude', 'Longitude']
```

#### Having a quick look at the latitude and longitude data


```python
df_latlong.head(3)
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
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M1B</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M1C</td>
      <td>43.784535</td>
      <td>-79.160497</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M1E</td>
      <td>43.763573</td>
      <td>-79.188711</td>
    </tr>
  </tbody>
</table>
</div>



### Joining the latitude and longitude data with Original Dataframe


```python
df_toronto = pd.merge(df_can_postcode, df_latlong, on=['Postcode'], how='inner')
```


```python
df_toronto.head(12)
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
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>M3A</td>
      <td>North York</td>
      <td>Parkwoods</td>
      <td>43.753259</td>
      <td>-79.329656</td>
    </tr>
    <tr>
      <th>1</th>
      <td>M4A</td>
      <td>North York</td>
      <td>Victoria Village</td>
      <td>43.725882</td>
      <td>-79.315572</td>
    </tr>
    <tr>
      <th>2</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>3</th>
      <td>M5A</td>
      <td>Downtown Toronto</td>
      <td>Regent Park</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>4</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Heights</td>
      <td>43.718518</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>5</th>
      <td>M6A</td>
      <td>North York</td>
      <td>Lawrence Manor</td>
      <td>43.718518</td>
      <td>-79.464763</td>
    </tr>
    <tr>
      <th>6</th>
      <td>M7A</td>
      <td>Queen's Park</td>
      <td>Queen's Park</td>
      <td>43.662301</td>
      <td>-79.389494</td>
    </tr>
    <tr>
      <th>7</th>
      <td>M9A</td>
      <td>Etobicoke</td>
      <td>Islington Avenue</td>
      <td>43.667856</td>
      <td>-79.532242</td>
    </tr>
    <tr>
      <th>8</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Rouge</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>9</th>
      <td>M1B</td>
      <td>Scarborough</td>
      <td>Malvern</td>
      <td>43.806686</td>
      <td>-79.194353</td>
    </tr>
    <tr>
      <th>10</th>
      <td>M3B</td>
      <td>North York</td>
      <td>Don Mills North</td>
      <td>43.745906</td>
      <td>-79.352188</td>
    </tr>
    <tr>
      <th>11</th>
      <td>M4B</td>
      <td>East York</td>
      <td>Woodbine Gardens</td>
      <td>43.706397</td>
      <td>-79.309937</td>
    </tr>
  </tbody>
</table>
</div>



#### Displaying shape of our dataframe


```python
neighborhoods = df_toronto [['Borough', 'Neighborhood', 'Latitude', 'Longitude']].copy()
print('The dataframe has {} boroughs and {} neighborhoods.'.format(
        len(neighborhoods['Borough'].unique()),
        neighborhoods.shape[0]
    )
)
```

    The dataframe has 11 boroughs and 212 neighborhoods.


### Use geopy library to get the latitude and longitude values of Toronto


```python
address = 'Toronto,Canada'

geolocator = Nominatim(user_agent="trn_explorer")
location = geolocator.geocode(address)
latitude_x = location.latitude
longitude_y = location.longitude
print('The geograpical coordinate of Toronto are {}, {}.'.format(latitude_x, longitude_y))
```

    The geograpical coordinate of Toronto are 43.653963, -79.387207.


#### Create a map of Toronto with neighborhoods superimposed on top


```python
# create map of Toronto using latitude and longitude values
map_toronto = folium.Map(location=[latitude_x, longitude_y], zoom_start=10)

# add markers to map
for lat, lng, bor, nei in zip(df_toronto['Latitude'], df_toronto['Longitude'], df_toronto['Borough'], df_toronto['Neighborhood']):
    
    label = '{}, {}'.format(nei, bor)
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='yellow',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.8,
        ).add_to(map_toronto)  
    
map_toronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOScsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjUzOTYzLC03OS4zODcyMDddLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTAsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzkzMTMzYWZlN2NmNjRlZDg5MzljYTNkNTkzZTNhNWIwID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iNzMxMDVjY2M1NWM0ODJlODcwYTY5MWRmOGU2MGFmZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1MzI1ODYsLTc5LjMyOTY1NjVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YjhmZDA4ZjUxNDY0ZTgxYjU5MmQ4N2EzYWUxYTAzZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNTA1MTQ1YWM4MTc0Zjg3OGU1ODBlYTZkZDc5MTM1OSA9ICQoJzxkaXYgaWQ9Imh0bWxfYTUwNTE0NWFjODE3NGY4NzhlNTgwZWE2ZGQ3OTEzNTkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmt3b29kcywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmI4ZmQwOGY1MTQ2NGU4MWI1OTJkODdhM2FlMWEwM2Quc2V0Q29udGVudChodG1sX2E1MDUxNDVhYzgxNzRmODc4ZTU4MGVhNmRkNzkxMzU5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2I3MzEwNWNjYzU1YzQ4MmU4NzBhNjkxZGY4ZTYwYWZlLmJpbmRQb3B1cChwb3B1cF82YjhmZDA4ZjUxNDY0ZTgxYjU5MmQ4N2EzYWUxYTAzZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YWRiZGZiYzUyZWQ0YTBlYWY5ZmFhODJkOGNiYWI0YiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg4MjI5OTk5OTk5NSwtNzkuMzE1NTcxNTk5OTk5OThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kODk4ZWE0OGU3MjM0MWUxYTVjNjk4YzUxMmJiM2YzZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84YmMyNjFmY2M0MTE0NjgzOWVhYjZjODY4Zjc3MWRkOCA9ICQoJzxkaXYgaWQ9Imh0bWxfOGJjMjYxZmNjNDExNDY4MzllYWI2Yzg2OGY3NzFkZDgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZpY3RvcmlhIFZpbGxhZ2UsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q4OThlYTQ4ZTcyMzQxZTFhNWM2OThjNTEyYmIzZjNmLnNldENvbnRlbnQoaHRtbF84YmMyNjFmY2M0MTE0NjgzOWVhYjZjODY4Zjc3MWRkOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YWRiZGZiYzUyZWQ0YTBlYWY5ZmFhODJkOGNiYWI0Yi5iaW5kUG9wdXAocG9wdXBfZDg5OGVhNDhlNzIzNDFlMWE1YzY5OGM1MTJiYjNmM2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjUwNTg1MGVkODQ0NDc1MzlhNDU5M2M5MzNiM2U5YjkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjdlYTlmZjg3NmY4NGFmYTkyZGMwYzJkMDA5MWQyNTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2RiZjdmYWNhYWU5NDdjMGEwYTg4NTVhZDEyNjVmMzIgPSAkKCc8ZGl2IGlkPSJodG1sXzNkYmY3ZmFjYWFlOTQ3YzBhMGE4ODU1YWQxMjY1ZjMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y3ZWE5ZmY4NzZmODRhZmE5MmRjMGMyZDAwOTFkMjU2LnNldENvbnRlbnQoaHRtbF8zZGJmN2ZhY2FhZTk0N2MwYTBhODg1NWFkMTI2NWYzMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yNTA1ODUwZWQ4NDQ0NzUzOWE0NTkzYzkzM2IzZTliOS5iaW5kUG9wdXAocG9wdXBfZjdlYTlmZjg3NmY4NGFmYTkyZGMwYzJkMDA5MWQyNTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGQ5ODFiNDU1ODliNGRiMGI2MGJlMTZiYTBkNTA3YzMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGIwZGNiNGRmYjdiNDk3M2I3YzUzZjdlNDVlNjZlZjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2UwYTYxYTU5YmU5NDA4ZDhiYTIwMDdjMjAwZTI4MmUgPSAkKCc8ZGl2IGlkPSJodG1sXzdlMGE2MWE1OWJlOTQwOGQ4YmEyMDA3YzIwMGUyODJlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SZWdlbnQgUGFyaywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOGIwZGNiNGRmYjdiNDk3M2I3YzUzZjdlNDVlNjZlZjYuc2V0Q29udGVudChodG1sXzdlMGE2MWE1OWJlOTQwOGQ4YmEyMDA3YzIwMGUyODJlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RkOTgxYjQ1NTg5YjRkYjBiNjBiZTE2YmEwZDUwN2MzLmJpbmRQb3B1cChwb3B1cF84YjBkY2I0ZGZiN2I0OTczYjdjNTNmN2U0NWU2NmVmNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYzdkNmE3M2Y0ZWY0ZWI5Yjk0NjczNmI0ZjA5MDY5NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxODUxNzk5OTk5OTk5NiwtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYTdhOTEyYWZhZGM0NGY2OWZhNjE3Y2ZjNTFmODI0YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZWMzOGQxZDE2ZWI0MjI5YTk3YmUzZTI3NjJiMjcwMCA9ICQoJzxkaXYgaWQ9Imh0bWxfMGVjMzhkMWQxNmViNDIyOWE5N2JlM2UyNzYyYjI3MDAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxhd3JlbmNlIEhlaWdodHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNhN2E5MTJhZmFkYzQ0ZjY5ZmE2MTdjZmM1MWY4MjRjLnNldENvbnRlbnQoaHRtbF8wZWMzOGQxZDE2ZWI0MjI5YTk3YmUzZTI3NjJiMjcwMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zYzdkNmE3M2Y0ZWY0ZWI5Yjk0NjczNmI0ZjA5MDY5Ni5iaW5kUG9wdXAocG9wdXBfM2E3YTkxMmFmYWRjNDRmNjlmYTYxN2NmYzUxZjgyNGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTNkYTQyYmI5ZTI4NDM1NzhmNzc2YWVmNTNiOGNlMDYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTg1MTc5OTk5OTk5OTYsLTc5LjQ2NDc2MzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTFmNDQzNWE0YWI5NDQwZmI2MzBlMTA0NzEzMzVhNjMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjZjMTkwZjE3ZjRlNDM4Yjk5NjdhNTZmYTVhZTAyNzkgPSAkKCc8ZGl2IGlkPSJodG1sXzY2YzE5MGYxN2Y0ZTQzOGI5OTY3YTU2ZmE1YWUwMjc5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MYXdyZW5jZSBNYW5vciwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTFmNDQzNWE0YWI5NDQwZmI2MzBlMTA0NzEzMzVhNjMuc2V0Q29udGVudChodG1sXzY2YzE5MGYxN2Y0ZTQzOGI5OTY3YTU2ZmE1YWUwMjc5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzEzZGE0MmJiOWUyODQzNTc4Zjc3NmFlZjUzYjhjZTA2LmJpbmRQb3B1cChwb3B1cF8xMWY0NDM1YTRhYjk0NDBmYjYzMGUxMDQ3MTMzNWE2Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZGIwMDk1YjUzODQ0NzFiYWNkOTAxZDg4N2VmZTc4YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjMwMTUsLTc5LjM4OTQ5MzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mY2YxMmFmMmM4ODc0YmEyYjdhNzIwZGQwNDNhZGJkNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMmExODVlYmExMjE0Yzg0OWY3NmFlOWU5NDg2ZmMyOSA9ICQoJzxkaXYgaWQ9Imh0bWxfMDJhMTg1ZWJhMTIxNGM4NDlmNzZhZTllOTQ4NmZjMjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlF1ZWVuJiMzOTtzIFBhcmssIFF1ZWVuJiMzOTtzIFBhcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZjZjEyYWYyYzg4NzRiYTJiN2E3MjBkZDA0M2FkYmQ0LnNldENvbnRlbnQoaHRtbF8wMmExODVlYmExMjE0Yzg0OWY3NmFlOWU5NDg2ZmMyOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZGIwMDk1YjUzODQ0NzFiYWNkOTAxZDg4N2VmZTc4Yy5iaW5kUG9wdXAocG9wdXBfZmNmMTJhZjJjODg3NGJhMmI3YTcyMGRkMDQzYWRiZDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzM2ZjNkMDMwYmU3NGQ3YThhMzVlYWFjM2UzNDI0ZDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc4NTU2LC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA3Yzc3NDU0MjdiMjQyMTE4NjY3ZWMwZGVmZDRlYTkwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NlMTlmZmY4YzljNzQ2YWViOTA1OGUzOTBiMGIyYzY4ID0gJCgnPGRpdiBpZD0iaHRtbF9jZTE5ZmZmOGM5Yzc0NmFlYjkwNThlMzkwYjBiMmM2OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SXNsaW5ndG9uIEF2ZW51ZSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wN2M3NzQ1NDI3YjI0MjExODY2N2VjMGRlZmQ0ZWE5MC5zZXRDb250ZW50KGh0bWxfY2UxOWZmZjhjOWM3NDZhZWI5MDU4ZTM5MGIwYjJjNjgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzM2ZjNkMDMwYmU3NGQ3YThhMzVlYWFjM2UzNDI0ZDguYmluZFBvcHVwKHBvcHVwXzA3Yzc3NDU0MjdiMjQyMTE4NjY3ZWMwZGVmZDRlYTkwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJlZDhlOGI5ZGZmNjRjMGQ4NTBjMDQxYTQyODFjYzc2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODA2Njg2Mjk5OTk5OTk2LC03OS4xOTQzNTM0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ2OTU4MTM0YWNhNDQxMTg4MGZkMDk5ZTc3NGE3OTU5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RjMjIxMDljMTlhZjQ0YmI5MThiNGZmN2JmNWIwYzBkID0gJCgnPGRpdiBpZD0iaHRtbF9kYzIyMTA5YzE5YWY0NGJiOTE4YjRmZjdiZjViMGMwZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um91Z2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80Njk1ODEzNGFjYTQ0MTE4ODBmZDA5OWU3NzRhNzk1OS5zZXRDb250ZW50KGh0bWxfZGMyMjEwOWMxOWFmNDRiYjkxOGI0ZmY3YmY1YjBjMGQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmVkOGU4YjlkZmY2NGMwZDg1MGMwNDFhNDI4MWNjNzYuYmluZFBvcHVwKHBvcHVwXzQ2OTU4MTM0YWNhNDQxMTg4MGZkMDk5ZTc3NGE3OTU5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M3ZDI4NzczN2E2NTQ5MTBhZjE3OTM0NTY4YTlkYmI1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuODA2Njg2Mjk5OTk5OTk2LC03OS4xOTQzNTM0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcxNjYxMDdlNjNjZTQwNzA4OTRiNjM0N2UxMDU0NGY2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA3ZTRkMjMxOTYyYTQ0YzU5ZGYxNjU2ZGYzN2Q1Y2YxID0gJCgnPGRpdiBpZD0iaHRtbF8wN2U0ZDIzMTk2MmE0NGM1OWRmMTY1NmRmMzdkNWNmMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFsdmVybiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzcxNjYxMDdlNjNjZTQwNzA4OTRiNjM0N2UxMDU0NGY2LnNldENvbnRlbnQoaHRtbF8wN2U0ZDIzMTk2MmE0NGM1OWRmMTY1NmRmMzdkNWNmMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jN2QyODc3MzdhNjU0OTEwYWYxNzkzNDU2OGE5ZGJiNS5iaW5kUG9wdXAocG9wdXBfNzE2NjEwN2U2M2NlNDA3MDg5NGI2MzQ3ZTEwNTQ0ZjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDg0ZDYxMjlmMjgxNDc2ZWIxZDk2M2U2MDk0MWIzZmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NDU5MDU3OTk5OTk5OTYsLTc5LjM1MjE4OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FkMmQ4YjkzYmY5YjRlNWJhMmI3NTk5ZmNmY2MwNGM4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzFhODVmOTQ5NGQzOTQwODBhMWM3NTJmY2Q2NDMwOWQxID0gJCgnPGRpdiBpZD0iaHRtbF8xYTg1Zjk0OTRkMzk0MDgwYTFjNzUyZmNkNjQzMDlkMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG9uIE1pbGxzIE5vcnRoLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZDJkOGI5M2JmOWI0ZTViYTJiNzU5OWZjZmNjMDRjOC5zZXRDb250ZW50KGh0bWxfMWE4NWY5NDk0ZDM5NDA4MGExYzc1MmZjZDY0MzA5ZDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDg0ZDYxMjlmMjgxNDc2ZWIxZDk2M2U2MDk0MWIzZmEuYmluZFBvcHVwKHBvcHVwX2FkMmQ4YjkzYmY5YjRlNWJhMmI3NTk5ZmNmY2MwNGM4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM1NGRjYjg1NDM1YjRmYzhhZWYzOTRiYTgzZGVkMDMzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2Mzk3MiwtNzkuMzA5OTM3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWYwZDI0N2IzYmRhNDJlNmFlMzM0ZGZmMjMyNTI3YjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTE0MjViZDQ3ZTA3NDRhMDhiMzliMDMwOWU4YTk5YTIgPSAkKCc8ZGl2IGlkPSJodG1sXzkxNDI1YmQ0N2UwNzQ0YTA4YjM5YjAzMDllOGE5OWEyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Xb29kYmluZSBHYXJkZW5zLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzFmMGQyNDdiM2JkYTQyZTZhZTMzNGRmZjIzMjUyN2IyLnNldENvbnRlbnQoaHRtbF85MTQyNWJkNDdlMDc0NGEwOGIzOWIwMzA5ZThhOTlhMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNTRkY2I4NTQzNWI0ZmM4YWVmMzk0YmE4M2RlZDAzMy5iaW5kUG9wdXAocG9wdXBfMWYwZDI0N2IzYmRhNDJlNmFlMzM0ZGZmMjMyNTI3YjIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTM4YzE0N2E2ZjBlNGQxOWI3NzY2MWZlM2VmNmNhYTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDYzOTcyLC03OS4zMDk5MzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iZTJjY2I2NDAxMDc0ZWMyYWY0YzhlZTBiMzE1ZDFjYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMzcxNDYwNDBiMDc0YWE2OTdiYzE3NWRhNTA5N2YxMCA9ICQoJzxkaXYgaWQ9Imh0bWxfYjM3MTQ2MDQwYjA3NGFhNjk3YmMxNzVkYTUwOTdmMTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlBhcmt2aWV3IEhpbGwsIEVhc3QgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmUyY2NiNjQwMTA3NGVjMmFmNGM4ZWUwYjMxNWQxY2Muc2V0Q29udGVudChodG1sX2IzNzE0NjA0MGIwNzRhYTY5N2JjMTc1ZGE1MDk3ZjEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUzOGMxNDdhNmYwZTRkMTliNzc2NjFmZTNlZjZjYWE3LmJpbmRQb3B1cChwb3B1cF9iZTJjY2I2NDAxMDc0ZWMyYWY0YzhlZTBiMzE1ZDFjYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NTExNzVhZGE2MjY0YTViYjM4YWQ3MjMyOGM1MjVmMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NzE2MTgsLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTFlMzVhMjMxNGM4NGI0ZWE2NWI4YWJiMWUzZWZlOGEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODg3NmNjMGExNjUzNDhhM2JiMGFhNDRhNjEzN2VhMjcgPSAkKCc8ZGl2IGlkPSJodG1sXzg4NzZjYzBhMTY1MzQ4YTNiYjBhYTQ0YTYxMzdlYTI3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SeWVyc29uLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xMWUzNWEyMzE0Yzg0YjRlYTY1YjhhYmIxZTNlZmU4YS5zZXRDb250ZW50KGh0bWxfODg3NmNjMGExNjUzNDhhM2JiMGFhNDRhNjEzN2VhMjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTUxMTc1YWRhNjI2NGE1YmIzOGFkNzIzMjhjNTI1ZjIuYmluZFBvcHVwKHBvcHVwXzExZTM1YTIzMTRjODRiNGVhNjViOGFiYjFlM2VmZThhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzMwNGNiZDY3ZDk1MTRmMGJiY2ViMGViYmQ0ZGFmOGQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3MTYxOCwtNzkuMzc4OTM3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNjNmODVkZjkyMDc0YTYxOWEzNzNjODcxNjc3ZTIwNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83NGY2Y2M4Y2EzODQ0MDZiOTY1YmViNGE0MTBlNjFhMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzRmNmNjOGNhMzg0NDA2Yjk2NWJlYjRhNDEwZTYxYTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhcmRlbiBEaXN0cmljdCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzYzZjg1ZGY5MjA3NGE2MTlhMzczYzg3MTY3N2UyMDYuc2V0Q29udGVudChodG1sXzc0ZjZjYzhjYTM4NDQwNmI5NjViZWI0YTQxMGU2MWExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMwNGNiZDY3ZDk1MTRmMGJiY2ViMGViYmQ0ZGFmOGQyLmJpbmRQb3B1cChwb3B1cF9jNjNmODVkZjkyMDc0YTYxOWEzNzNjODcxNjc3ZTIwNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NmViZTcwODk1YTE0MmIyYWI1ODZlODNmOTYwYjIyYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwOTU3NywtNzkuNDQ1MDcyNTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NGZmZGYwMzM0NmM0ODNiYTQyNmQ2YWU4YzM5NDk2YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNzM2MDU1ODFkMTc0OGI2OTBiNTFlNTJhMGM2YzdmOSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTczNjA1NTgxZDE3NDhiNjkwYjUxZTUyYTBjNmM3ZjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdsZW5jYWlybiwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDRmZmRmMDMzNDZjNDgzYmE0MjZkNmFlOGMzOTQ5NmIuc2V0Q29udGVudChodG1sX2U3MzYwNTU4MWQxNzQ4YjY5MGI1MWU1MmEwYzZjN2Y5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc2ZWJlNzA4OTVhMTQyYjJhYjU4NmU4M2Y5NjBiMjJhLmJpbmRQb3B1cChwb3B1cF80NGZmZGYwMzM0NmM0ODNiYTQyNmQ2YWU4YzM5NDk2Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNTdiMDNmMDY3MDA0NTE3YmI0NjhhNDUwNGViNjM3NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDk0MzIsLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTA5OTE3ZWIzZDhkNDAzYTkzZGQzYmMzNmRmY2IxZjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNjdlNDJkNGIwOGNmNDc2YmJlNTBkOTUzYTY4ZmE4MzIgPSAkKCc8ZGl2IGlkPSJodG1sXzY3ZTQyZDRiMDhjZjQ3NmJiZTUwZDk1M2E2OGZhODMyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DbG92ZXJkYWxlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEwOTkxN2ViM2Q4ZDQwM2E5M2RkM2JjMzZkZmNiMWY0LnNldENvbnRlbnQoaHRtbF82N2U0MmQ0YjA4Y2Y0NzZiYmU1MGQ5NTNhNjhmYTgzMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNTdiMDNmMDY3MDA0NTE3YmI0NjhhNDUwNGViNjM3Ni5iaW5kUG9wdXAocG9wdXBfMTA5OTE3ZWIzZDhkNDAzYTkzZGQzYmMzNmRmY2IxZjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTNlMWU2NmFiMDMxNGUxMjgxZDgyY2YxNmY5ZjNlMjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA5NDMyLC03OS41NTQ3MjQ0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ViOGYxNTY1YzA2MjRjZmJiMDFjNDRjOGEzYTVlNzdmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRmMjEwNjQzOWRiNjRiNzk5YjNlYTU5MGExNTM5NWY2ID0gJCgnPGRpdiBpZD0iaHRtbF80ZjIxMDY0MzlkYjY0Yjc5OWIzZWE1OTBhMTUzOTVmNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SXNsaW5ndG9uLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ViOGYxNTY1YzA2MjRjZmJiMDFjNDRjOGEzYTVlNzdmLnNldENvbnRlbnQoaHRtbF80ZjIxMDY0MzlkYjY0Yjc5OWIzZWE1OTBhMTUzOTVmNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81M2UxZTY2YWIwMzE0ZTEyODFkODJjZjE2ZjlmM2UyNS5iaW5kUG9wdXAocG9wdXBfZWI4ZjE1NjVjMDYyNGNmYmIwMWM0NGM4YTNhNWU3N2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGY0NDIzOGJkNzZhNDc5YTlmYTYxMjg3YTM2NzFmYzAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA5NDMyLC03OS41NTQ3MjQ0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM5MDZjNjk5MTRjMzQ3ZTZiMDlmNDU1YTI4NmU2NDQ3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE3MTE4ODBlODU1MjQ3NTdiN2M1ZGQ4MDlmM2U2YTE3ID0gJCgnPGRpdiBpZD0iaHRtbF8xNzExODgwZTg1NTI0NzU3YjdjNWRkODA5ZjNlNmExNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWFydGluIEdyb3ZlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM5MDZjNjk5MTRjMzQ3ZTZiMDlmNDU1YTI4NmU2NDQ3LnNldENvbnRlbnQoaHRtbF8xNzExODgwZTg1NTI0NzU3YjdjNWRkODA5ZjNlNmExNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kZjQ0MjM4YmQ3NmE0NzlhOWZhNjEyODdhMzY3MWZjMC5iaW5kUG9wdXAocG9wdXBfMzkwNmM2OTkxNGMzNDdlNmIwOWY0NTVhMjg2ZTY0NDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTk5ODEzOGU3MWFhNDU3ZTgzNzU5NzQwZTViOTRlNTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA5NDMyLC03OS41NTQ3MjQ0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QyNjgzNDcwZjZmNzQ0YzBiZmQ0YmM4NzIxZjhiM2JjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JjNGZkOTIwYjQ4ZDRhOTViODJmMWFmZWYxM2FiOWRmID0gJCgnPGRpdiBpZD0iaHRtbF9iYzRmZDkyMGI0OGQ0YTk1YjgyZjFhZmVmMTNhYjlkZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UHJpbmNlc3MgR2FyZGVucywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMjY4MzQ3MGY2Zjc0NGMwYmZkNGJjODcyMWY4YjNiYy5zZXRDb250ZW50KGh0bWxfYmM0ZmQ5MjBiNDhkNGE5NWI4MmYxYWZlZjEzYWI5ZGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTk5ODEzOGU3MWFhNDU3ZTgzNzU5NzQwZTViOTRlNTUuYmluZFBvcHVwKHBvcHVwX2QyNjgzNDcwZjZmNzQ0YzBiZmQ0YmM4NzIxZjhiM2JjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QxNmM0MzlkODM4YzRlZjJiNzI0YWRlNDI2YTQwNmFkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwOTQzMiwtNzkuNTU0NzI0NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lYzU5ZWZhZGZiZDM0MTgwYTE1NmNiYmRiN2YxYTE4MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kM2EwMWU4MDRhMzM0Nzk4OTllYjJjMTFjMjEzNjE2ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZDNhMDFlODA0YTMzNDc5ODk5ZWIyYzExYzIxMzYxNmQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldlc3QgRGVhbmUgUGFyaywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYzU5ZWZhZGZiZDM0MTgwYTE1NmNiYmRiN2YxYTE4Mi5zZXRDb250ZW50KGh0bWxfZDNhMDFlODA0YTMzNDc5ODk5ZWIyYzExYzIxMzYxNmQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDE2YzQzOWQ4MzhjNGVmMmI3MjRhZGU0MjZhNDA2YWQuYmluZFBvcHVwKHBvcHVwX2VjNTllZmFkZmJkMzQxODBhMTU2Y2JiZGI3ZjFhMTgyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2EzYWRjMTJlYTNjMzQ0MmZiMjg5MjJjOThmYjBiOTJmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzg0NTM1MSwtNzkuMTYwNDk3MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yODQxMGI5OTM3NDI0ZTM1YTBmZTU0ODg1YTMzMDE4OCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MjBhZDliZWJmZGE0YmFmOWQxNDQzMzBhNjBjNmY0MyA9ICQoJzxkaXYgaWQ9Imh0bWxfODIwYWQ5YmViZmRhNGJhZjlkMTQ0MzMwYTYwYzZmNDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpZ2hsYW5kIENyZWVrLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjg0MTBiOTkzNzQyNGUzNWEwZmU1NDg4NWEzMzAxODguc2V0Q29udGVudChodG1sXzgyMGFkOWJlYmZkYTRiYWY5ZDE0NDMzMGE2MGM2ZjQzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2EzYWRjMTJlYTNjMzQ0MmZiMjg5MjJjOThmYjBiOTJmLmJpbmRQb3B1cChwb3B1cF8yODQxMGI5OTM3NDI0ZTM1YTBmZTU0ODg1YTMzMDE4OCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl81MzExODFkYThlODY0NmYzYTgzMmU2YzNkYjEzN2Y0NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4NDUzNTEsLTc5LjE2MDQ5NzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWMzNWNhZDJjOWUyNDg2MGE3NTAyZmJmOTAyM2FlYTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjk4MWYzMDU3MTNiNDZhYzk5NTU0NTdhZDZjNDJmZmEgPSAkKCc8ZGl2IGlkPSJodG1sXzI5ODFmMzA1NzEzYjQ2YWM5OTU1NDU3YWQ2YzQyZmZhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb3VnZSBIaWxsLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWMzNWNhZDJjOWUyNDg2MGE3NTAyZmJmOTAyM2FlYTguc2V0Q29udGVudChodG1sXzI5ODFmMzA1NzEzYjQ2YWM5OTU1NDU3YWQ2YzQyZmZhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUzMTE4MWRhOGU4NjQ2ZjNhODMyZTZjM2RiMTM3ZjQ1LmJpbmRQb3B1cChwb3B1cF9hYzM1Y2FkMmM5ZTI0ODYwYTc1MDJmYmY5MDIzYWVhOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wMzU4YjI4M2JiNDI0NGQ2YmQ0MjA2NTlhMGYzMTIxNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4NDUzNTEsLTc5LjE2MDQ5NzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzg5Y2NlZGQ4NGZiNDA3OWFhNmNkMWRjMzc2NGRmYTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDA4NzBlMzE3YzE5NGZjOTg5MmE0ZDBmYjMyOGU1YmMgPSAkKCc8ZGl2IGlkPSJodG1sXzAwODcwZTMxN2MxOTRmYzk4OTJhNGQwZmIzMjhlNWJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Qb3J0IFVuaW9uLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzg5Y2NlZGQ4NGZiNDA3OWFhNmNkMWRjMzc2NGRmYTcuc2V0Q29udGVudChodG1sXzAwODcwZTMxN2MxOTRmYzk4OTJhNGQwZmIzMjhlNWJjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzAzNThiMjgzYmI0MjQ0ZDZiZDQyMDY1OWEwZjMxMjE1LmJpbmRQb3B1cChwb3B1cF8zODljY2VkZDg0ZmI0MDc5YWE2Y2QxZGMzNzY0ZGZhNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNTMyMzIxNzFhNjM0NTFiYjdkMTU5MjE5OTdiMDRlZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg5OTcwMDAwMDAxLC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NDQ1NDJjZmNjYTY0MzMwOGEzNTBlZmFjYTI3ZWZkNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMTZkN2RmZmU1NDM0NzRhOThkYTI3NTVmNWM0NGExOCA9ICQoJzxkaXYgaWQ9Imh0bWxfMTE2ZDdkZmZlNTQzNDc0YTk4ZGEyNzU1ZjVjNDRhMTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZsZW1pbmdkb24gUGFyaywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDQ0NTQyY2ZjY2E2NDMzMDhhMzUwZWZhY2EyN2VmZDcuc2V0Q29udGVudChodG1sXzExNmQ3ZGZmZTU0MzQ3NGE5OGRhMjc1NWY1YzQ0YTE4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM1MzIzMjE3MWE2MzQ1MWJiN2QxNTkyMTk5N2IwNGVlLmJpbmRQb3B1cChwb3B1cF80NDQ1NDJjZmNjYTY0MzMwOGEzNTBlZmFjYTI3ZWZkNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOTZjZDcxODI3NzQ0NmRlOGNlMjAzMjZlNTVmYjQxNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcyNTg5OTcwMDAwMDAxLC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82NGZlNGUyNmFmNGM0Yzc0ODc4Y2E3ZDMxZjY2N2NlYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lYzFiZjllNmNmYTc0YzkzOTIwNTBiMzU0OWM3YzExYSA9ICQoJzxkaXYgaWQ9Imh0bWxfZWMxYmY5ZTZjZmE3NGM5MzkyMDUwYjM1NDljN2MxMWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvbiBNaWxscyBTb3V0aCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjRmZTRlMjZhZjRjNGM3NDg3OGNhN2QzMWY2NjdjZWIuc2V0Q29udGVudChodG1sX2VjMWJmOWU2Y2ZhNzRjOTM5MjA1MGIzNTQ5YzdjMTFhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q5NmNkNzE4Mjc3NDQ2ZGU4Y2UyMDMyNmU1NWZiNDE0LmJpbmRQb3B1cChwb3B1cF82NGZlNGUyNmFmNGM0Yzc0ODc4Y2E3ZDMxZjY2N2NlYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMTE1N2MyZTE4ZjU0MzczODk3N2M5MzM2OGU5Mzc3NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5NTM0MzkwMDAwMDAwNSwtNzkuMzE4Mzg4N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc4NWQ2M2E1MzRkZTRkNjBhYmI2YmM3NWM2YWRhZGE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJhZTU0MWVkMGFkOTRjZDNiZWUxZjQxYWZlZjZjYzQyID0gJCgnPGRpdiBpZD0iaHRtbF8yYWU1NDFlZDBhZDk0Y2QzYmVlMWY0MWFmZWY2Y2M0MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V29vZGJpbmUgSGVpZ2h0cywgRWFzdCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ODVkNjNhNTM0ZGU0ZDYwYWJiNmJjNzVjNmFkYWRhNi5zZXRDb250ZW50KGh0bWxfMmFlNTQxZWQwYWQ5NGNkM2JlZTFmNDFhZmVmNmNjNDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjExNTdjMmUxOGY1NDM3Mzg5NzdjOTMzNjhlOTM3NzQuYmluZFBvcHVwKHBvcHVwXzc4NWQ2M2E1MzRkZTRkNjBhYmI2YmM3NWM2YWRhZGE2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MwMzcxMGNiZDRkYTQ2ODliODJkNjJmODcxN2I2MWY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNDkzOSwtNzkuMzc1NDE3OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYzNWI4MTUzYmM2NDQxNDhhYzQ4NTdjYjJiODRhODMwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQwZTQ4ZmUyODA5ZjQ4ZmJhMjM5MjRkYzVjNWM0OGI4ID0gJCgnPGRpdiBpZD0iaHRtbF80MGU0OGZlMjgwOWY0OGZiYTIzOTI0ZGM1YzVjNDhiOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYzNWI4MTUzYmM2NDQxNDhhYzQ4NTdjYjJiODRhODMwLnNldENvbnRlbnQoaHRtbF80MGU0OGZlMjgwOWY0OGZiYTIzOTI0ZGM1YzVjNDhiOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMDM3MTBjYmQ0ZGE0Njg5YjgyZDYyZjg3MTdiNjFmNC5iaW5kUG9wdXAocG9wdXBfNjM1YjgxNTNiYzY0NDE0OGFjNDg1N2NiMmI4NGE4MzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2U3MjM2MTE2OWY4NGZiYzg1OTBlOGEwYTA2YjYxOWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTM3ODEzLC03OS40MjgxOTE0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Y1ZTdlN2ZhMjQ5ZDQxZmM4MDA4ZWFiZDhkM2NkZDE0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2UxYTE2ZThkNmFlMTQxMzhhNWMyYWMyYjdjNzk3NTU0ID0gJCgnPGRpdiBpZD0iaHRtbF9lMWExNmU4ZDZhZTE0MTM4YTVjMmFjMmI3Yzc5NzU1NCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SHVtZXdvb2QtQ2VkYXJ2YWxlLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNWU3ZTdmYTI0OWQ0MWZjODAwOGVhYmQ4ZDNjZGQxNC5zZXRDb250ZW50KGh0bWxfZTFhMTZlOGQ2YWUxNDEzOGE1YzJhYzJiN2M3OTc1NTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2U3MjM2MTE2OWY4NGZiYzg1OTBlOGEwYTA2YjYxOWQuYmluZFBvcHVwKHBvcHVwX2Y1ZTdlN2ZhMjQ5ZDQxZmM4MDA4ZWFiZDhkM2NkZDE0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlhYTlhNTkzNWUzODQ4ZGViYWE0ZTQ2ZWE5NGNmOGZhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQzNTE1MiwtNzkuNTc3MjAwNzk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hYTJlZmExMDlkMjU0YWUyYWE0ODk0Zjk2NDI3NjhmYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9kZjk5MTUwZGE1Nzk0NDRhOWFmYzZmMWFhMGVhNzdkZCA9ICQoJzxkaXYgaWQ9Imh0bWxfZGY5OTE1MGRhNTc5NDQ0YTlhZmM2ZjFhYTBlYTc3ZGQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJsb29yZGFsZSBHYXJkZW5zLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FhMmVmYTEwOWQyNTRhZTJhYTQ4OTRmOTY0Mjc2OGZiLnNldENvbnRlbnQoaHRtbF9kZjk5MTUwZGE1Nzk0NDRhOWFmYzZmMWFhMGVhNzdkZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YWE5YTU5MzVlMzg0OGRlYmFhNGU0NmVhOTRjZjhmYS5iaW5kUG9wdXAocG9wdXBfYWEyZWZhMTA5ZDI1NGFlMmFhNDg5NGY5NjQyNzY4ZmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTVhNjRhMzJmYWI1NDRjMDljMTliMWZlYWRiNTQwNWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDM1MTUyLC03OS41NzcyMDA3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNhOGFkY2I5YmMxMjRhMDY5MGQ5OGFiOTNjMmM1MTJjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZiMGIwYThlOWFkYzQ3MjViOTU0YzQ3M2FmNTAyMjlmID0gJCgnPGRpdiBpZD0iaHRtbF9mYjBiMGE4ZTlhZGM0NzI1Yjk1NGM0NzNhZjUwMjI5ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RXJpbmdhdGUsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfM2E4YWRjYjliYzEyNGEwNjkwZDk4YWI5M2MyYzUxMmMuc2V0Q29udGVudChodG1sX2ZiMGIwYThlOWFkYzQ3MjViOTU0YzQ3M2FmNTAyMjlmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE1YTY0YTMyZmFiNTQ0YzA5YzE5YjFmZWFkYjU0MDVlLmJpbmRQb3B1cChwb3B1cF8zYThhZGNiOWJjMTI0YTA2OTBkOThhYjkzYzJjNTEyYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zYjczMWY3ZWUwZTg0MmUwYTdlM2FhNDdlYWY0M2U5OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MzUxNTIsLTc5LjU3NzIwMDc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDQ0MDljZGFhZWM0NDczZmJmYzRhZTkxNWRlMTQ1MDAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfM2MzYzQ3OTkxMmU5NGFmZWIwNDgwYjgyNGYyNWFiNTMgPSAkKCc8ZGl2IGlkPSJodG1sXzNjM2M0Nzk5MTJlOTRhZmViMDQ4MGI4MjRmMjVhYjUzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NYXJrbGFuZCBXb29kLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzA0NDA5Y2RhYWVjNDQ3M2ZiZmM0YWU5MTVkZTE0NTAwLnNldENvbnRlbnQoaHRtbF8zYzNjNDc5OTEyZTk0YWZlYjA0ODBiODI0ZjI1YWI1Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zYjczMWY3ZWUwZTg0MmUwYTdlM2FhNDdlYWY0M2U5OC5iaW5kUG9wdXAocG9wdXBfMDQ0MDljZGFhZWM0NDczZmJmYzRhZTkxNWRlMTQ1MDApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGU3NGU1YzIwYWZmNDBmMmEwNDIxYWNhNDlmODIyZDAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDM1MTUyLC03OS41NzcyMDA3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIyOGViMDNhNzYxNjQ4MjU5ODc0NjYxODNiMjY3OTdiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EzNDE0OTdlMzMwMDRjOTg4MTNkMDFlMDIxNTAwNWI1ID0gJCgnPGRpdiBpZD0iaHRtbF9hMzQxNDk3ZTMzMDA0Yzk4ODEzZDAxZTAyMTUwMDViNSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T2xkIEJ1cm5oYW10aG9ycGUsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjI4ZWIwM2E3NjE2NDgyNTk4NzQ2NjE4M2IyNjc5N2Iuc2V0Q29udGVudChodG1sX2EzNDE0OTdlMzMwMDRjOTg4MTNkMDFlMDIxNTAwNWI1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RlNzRlNWMyMGFmZjQwZjJhMDQyMWFjYTQ5ZjgyMmQwLmJpbmRQb3B1cChwb3B1cF8yMjhlYjAzYTc2MTY0ODI1OTg3NDY2MTgzYjI2Nzk3Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jMTBhYzBjZGRjMWY0NGE2OTBkNmFlYWYwYzk1ZjJiYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2MzU3MjYsLTc5LjE4ODcxMTVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMDhiMzRkZTJlODY0NWI4YTA1YTM0Yzk0OTZkZWMxMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ZTZkMzI4YTdiZGM0ZWEwYjg2YmViZTUxYzdhOGQyZSA9ICQoJzxkaXYgaWQ9Imh0bWxfOGU2ZDMyOGE3YmRjNGVhMGI4NmJlYmU1MWM3YThkMmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkd1aWxkd29vZCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YwOGIzNGRlMmU4NjQ1YjhhMDVhMzRjOTQ5NmRlYzExLnNldENvbnRlbnQoaHRtbF84ZTZkMzI4YTdiZGM0ZWEwYjg2YmViZTUxYzdhOGQyZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMTBhYzBjZGRjMWY0NGE2OTBkNmFlYWYwYzk1ZjJiYi5iaW5kUG9wdXAocG9wdXBfZjA4YjM0ZGUyZTg2NDViOGEwNWEzNGM5NDk2ZGVjMTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTRjZmU1MDIzNzQ4NGYyOThmMGVlNDQwMWU1ZWQxZjUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NjM1NzI2LC03OS4xODg3MTE1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmIwMTBmOGYzNmM3NDIxYWFhOGY3YzZjMzI2MTZiZTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGU3ZDNhMjYyNTE0NGQzOWIzMjIxZGYwYjQyNWY3MjAgPSAkKCc8ZGl2IGlkPSJodG1sXzBlN2QzYTI2MjUxNDRkMzliMzIyMWRmMGI0MjVmNzIwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3JuaW5nc2lkZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZiMDEwZjhmMzZjNzQyMWFhYThmN2M2YzMyNjE2YmUzLnNldENvbnRlbnQoaHRtbF8wZTdkM2EyNjI1MTQ0ZDM5YjMyMjFkZjBiNDI1ZjcyMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NGNmZTUwMjM3NDg0ZjI5OGYwZWU0NDAxZTVlZDFmNS5iaW5kUG9wdXAocG9wdXBfZmIwMTBmOGYzNmM3NDIxYWFhOGY3YzZjMzI2MTZiZTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTY1Mjk1YTIxNzBmNGI4ZGI3NmIxOTFlMjc3MjZiZDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NjM1NzI2LC03OS4xODg3MTE1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWZiMmJmZGUyNjRjNDgzNTk1NWExYTkxNjUyZmVkNzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYThkNGUxMmVjMDViNDM0ZjhmYjExZjJhNGIxNjZlMTkgPSAkKCc8ZGl2IGlkPSJodG1sX2E4ZDRlMTJlYzA1YjQzNGY4ZmIxMWYyYTRiMTY2ZTE5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XZXN0IEhpbGwsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81ZmIyYmZkZTI2NGM0ODM1OTU1YTFhOTE2NTJmZWQ3NC5zZXRDb250ZW50KGh0bWxfYThkNGUxMmVjMDViNDM0ZjhmYjExZjJhNGIxNjZlMTkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTY1Mjk1YTIxNzBmNGI4ZGI3NmIxOTFlMjc3MjZiZDUuYmluZFBvcHVwKHBvcHVwXzVmYjJiZmRlMjY0YzQ4MzU5NTVhMWE5MTY1MmZlZDc0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk0M2Q0MGZiZmZhOTQxYTk4ZDg1ZGU2OTY1YmM1NTY2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc2MzU3Mzk5OTk5OTksLTc5LjI5MzAzMTJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MzA3MzMzOGI5MTM0N2JhOWY4NGQyZDMyZmMxOTFkYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lM2MyZGY1YmI0MDU0YmRiYjYyNzZlYzJjNWI3NzcxMyA9ICQoJzxkaXYgaWQ9Imh0bWxfZTNjMmRmNWJiNDA1NGJkYmI2Mjc2ZWMyYzViNzc3MTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBCZWFjaGVzLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYzMDczMzM4YjkxMzQ3YmE5Zjg0ZDJkMzJmYzE5MWRhLnNldENvbnRlbnQoaHRtbF9lM2MyZGY1YmI0MDU0YmRiYjYyNzZlYzJjNWI3NzcxMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NDNkNDBmYmZmYTk0MWE5OGQ4NWRlNjk2NWJjNTU2Ni5iaW5kUG9wdXAocG9wdXBfNjMwNzMzMzhiOTEzNDdiYTlmODRkMmQzMmZjMTkxZGEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjU3ODg0N2E4MTIzNDdjOGFlZTU0NTQyOWVhYWE5N2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDQ3NzA3OTk5OTk5OTYsLTc5LjM3MzMwNjRdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jN2ZiMzhiYTcyZDU0ZGZkOWM2OWY0MjFlMDQ4OWY1NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xY2ExNWJiZjM1MDg0NmIzOWRhMDNiZDExYTRhYTM3YyA9ICQoJzxkaXYgaWQ9Imh0bWxfMWNhMTViYmYzNTA4NDZiMzlkYTAzYmQxMWE0YWEzN2MiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcmN6eSBQYXJrLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jN2ZiMzhiYTcyZDU0ZGZkOWM2OWY0MjFlMDQ4OWY1Ni5zZXRDb250ZW50KGh0bWxfMWNhMTViYmYzNTA4NDZiMzlkYTAzYmQxMWE0YWEzN2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjU3ODg0N2E4MTIzNDdjOGFlZTU0NTQyOWVhYWE5N2EuYmluZFBvcHVwKHBvcHVwX2M3ZmIzOGJhNzJkNTRkZmQ5YzY5ZjQyMWUwNDg5ZjU2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjZmRlM2M5MjdmNTQ1NDY4NDM0MDRkNzY2YWZlZTYzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5MDI1NiwtNzkuNDUzNTEyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmU1ZGFkMzMzZGJjNDdlMThlNGMyZjRmYzg1OWVjM2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWUxMzhkOTFkNmQ2NGE0NzkwZjM1Y2JlMThlNGQzOWQgPSAkKCc8ZGl2IGlkPSJodG1sXzllMTM4ZDkxZDZkNjRhNDc5MGYzNWNiZTE4ZTRkMzlkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYWxlZG9uaWEtRmFpcmJhbmtzLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZTVkYWQzMzNkYmM0N2UxOGU0YzJmNGZjODU5ZWMzYy5zZXRDb250ZW50KGh0bWxfOWUxMzhkOTFkNmQ2NGE0NzkwZjM1Y2JlMThlNGQzOWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGNmZGUzYzkyN2Y1NDU0Njg0MzQwNGQ3NjZhZmVlNjMuYmluZFBvcHVwKHBvcHVwXzJlNWRhZDMzM2RiYzQ3ZTE4ZTRjMmY0ZmM4NTllYzNjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U5MjE5NWIxYWRjOTRlMjdiZDRhYWU3YWE5MDI5OGU3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzcwOTkyMSwtNzkuMjE2OTE3NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kZWRmZWI4ZTZlZjg0NTc0YjU5ZTE0OTJkOTE0ZjgzYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83MDM1ZTk1N2VlM2I0YTY4YmFmOTZhM2ZkNzJhZjY5ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzAzNWU5NTdlZTNiNGE2OGJhZjk2YTNmZDcyYWY2OWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldvYnVybiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RlZGZlYjhlNmVmODQ1NzRiNTllMTQ5MmQ5MTRmODNhLnNldENvbnRlbnQoaHRtbF83MDM1ZTk1N2VlM2I0YTY4YmFmOTZhM2ZkNzJhZjY5ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lOTIxOTViMWFkYzk0ZTI3YmQ0YWFlN2FhOTAyOThlNy5iaW5kUG9wdXAocG9wdXBfZGVkZmViOGU2ZWY4NDU3NGI1OWUxNDkyZDkxNGY4M2EpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWY5MjkyNmFmMDc1NDZkY2JiZWFhZDNkYWQ3MWExZmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDkwNjA0LC03OS4zNjM0NTE3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDAxNDNlNzA1ODRiNGFjZWIzZTdjNjVlNWJmNGU4MWEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNThmNzQzNjM2OGQ4NDZiODhjNTg1ZjE4ZTNhYWRkMmQgPSAkKCc8ZGl2IGlkPSJodG1sXzU4Zjc0MzYzNjhkODQ2Yjg4YzU4NWYxOGUzYWFkZDJkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MZWFzaWRlLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzAwMTQzZTcwNTg0YjRhY2ViM2U3YzY1ZTViZjRlODFhLnNldENvbnRlbnQoaHRtbF81OGY3NDM2MzY4ZDg0NmI4OGM1ODVmMThlM2FhZGQyZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81ZjkyOTI2YWYwNzU0NmRjYmJlYWFkM2RhZDcxYTFmZC5iaW5kUG9wdXAocG9wdXBfMDAxNDNlNzA1ODRiNGFjZWIzZTdjNjVlNWJmNGU4MWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2ExYTI3NGJmNjA5NGNhNDhhYjQwOGQ1NDMzOWE4OTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTc5NTI0LC03OS4zODczODI2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTMwMDcyMmFmOWM4NDk4OGI2YWRjOWZlNTAyOTIzZjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjA1MDNiNzI0ODQ4NDU5YWExNTY4ODIxNWJjNDg0NjUgPSAkKCc8ZGl2IGlkPSJodG1sXzIwNTAzYjcyNDg0ODQ1OWFhMTU2ODgyMTViYzQ4NDY1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DZW50cmFsIEJheSBTdHJlZXQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2EzMDA3MjJhZjljODQ5ODhiNmFkYzlmZTUwMjkyM2Y2LnNldENvbnRlbnQoaHRtbF8yMDUwM2I3MjQ4NDg0NTlhYTE1Njg4MjE1YmM0ODQ2NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jYTFhMjc0YmY2MDk0Y2E0OGFiNDA4ZDU0MzM5YTg5OS5iaW5kUG9wdXAocG9wdXBfYTMwMDcyMmFmOWM4NDk4OGI2YWRjOWZlNTAyOTIzZjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODMyNDdkMzc2NGZmNGM3OWI3NzhiMzZlZDNjYmUxZDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njk1NDIsLTc5LjQyMjU2MzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xNjIyYmJiYTFjMjk0YmE2OTM5ZmNiNWM2MDAwYTA4YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lM2Y2ZjQ5ZjUyMGM0M2ExOGJlMzNiZmY5OTVhNTQwYSA9ICQoJzxkaXYgaWQ9Imh0bWxfZTNmNmY0OWY1MjBjNDNhMThiZTMzYmZmOTk1YTU0MGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocmlzdGllLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xNjIyYmJiYTFjMjk0YmE2OTM5ZmNiNWM2MDAwYTA4YS5zZXRDb250ZW50KGh0bWxfZTNmNmY0OWY1MjBjNDNhMThiZTMzYmZmOTk1YTU0MGEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODMyNDdkMzc2NGZmNGM3OWI3NzhiMzZlZDNjYmUxZDMuYmluZFBvcHVwKHBvcHVwXzE2MjJiYmJhMWMyOTRiYTY5MzlmY2I1YzYwMDBhMDhhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ2OGQ2ZGM2NTE0YTQ3YTRhYTQ0ZGM1NjNiMDY3NDU0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzczMTM2LC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk1NDI2YWFiNWQ2NDQzZWFhOTJkM2ZmOGUxMTE3MDc0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EzMDAyNGQ3MjE1ZjRhODliMDE4MjdjOTQ1ZTlhNGIxID0gJCgnPGRpdiBpZD0iaHRtbF9hMzAwMjRkNzIxNWY0YTg5YjAxODI3Yzk0NWU5YTRiMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2VkYXJicmFlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTU0MjZhYWI1ZDY0NDNlYWE5MmQzZmY4ZTExMTcwNzQuc2V0Q29udGVudChodG1sX2EzMDAyNGQ3MjE1ZjRhODliMDE4MjdjOTQ1ZTlhNGIxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ2OGQ2ZGM2NTE0YTQ3YTRhYTQ0ZGM1NjNiMDY3NDU0LmJpbmRQb3B1cChwb3B1cF85NTQyNmFhYjVkNjQ0M2VhYTkyZDNmZjhlMTExNzA3NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNGQxOTEwOTg3Yjk0YzI1YWVhNGM1NDdiMTJmMGQ4ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgwMzc2MjIsLTc5LjM2MzQ1MTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jM2I4ZThlNzQxMjc0MTM3OTIxYjg3NzdmNWNiYmI0ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYWFkM2Q5NDA2YzY0MDE0ODdhNmJkYThhODVjODM2ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMmFhZDNkOTQwNmM2NDAxNDg3YTZiZGE4YTg1YzgzNmYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpbGxjcmVzdCBWaWxsYWdlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jM2I4ZThlNzQxMjc0MTM3OTIxYjg3NzdmNWNiYmI0ZC5zZXRDb250ZW50KGh0bWxfMmFhZDNkOTQwNmM2NDAxNDg3YTZiZGE4YTg1YzgzNmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzRkMTkxMDk4N2I5NGMyNWFlYTRjNTQ3YjEyZjBkOGUuYmluZFBvcHVwKHBvcHVwX2MzYjhlOGU3NDEyNzQxMzc5MjFiODc3N2Y1Y2JiYjRkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M1ZmRjN2MzODU0YTQ2YTU5ZGJjMzY5NjQzZjBjZGViID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU0MzI4MywtNzkuNDQyMjU5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY1N2YxYjVkMGRlZTQyMjJiM2E1OTM4MzE5MWFhNTk5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzExYjk0N2I0OWMyOTQ4M2I5ODUzZGYxMGJmMjIyNjg2ID0gJCgnPGRpdiBpZD0iaHRtbF8xMWI5NDdiNDljMjk0ODNiOTg1M2RmMTBiZjIyMjY4NiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgTWFub3IsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY1N2YxYjVkMGRlZTQyMjJiM2E1OTM4MzE5MWFhNTk5LnNldENvbnRlbnQoaHRtbF8xMWI5NDdiNDljMjk0ODNiOTg1M2RmMTBiZjIyMjY4Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jNWZkYzdjMzg1NGE0NmE1OWRiYzM2OTY0M2YwY2RlYi5iaW5kUG9wdXAocG9wdXBfNjU3ZjFiNWQwZGVlNDIyMmIzYTU5MzgzMTkxYWE1OTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmI4YTBkZmEyNGU2NGM4NGEwNWE4ZDEzYjFiMGFkYTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTQzMjgzLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzQyYWYwNjQ1ZTljNDNiMmJmZGUyYjA1OWFiNDI3MzEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzdhZjM5MTc5ZGM1NDk0ODg5YzcwZDUyN2YzY2RlNmQgPSAkKCc8ZGl2IGlkPSJodG1sXzc3YWYzOTE3OWRjNTQ5NDg4OWM3MGQ1MjdmM2NkZTZkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcgTm9ydGgsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM0MmFmMDY0NWU5YzQzYjJiZmRlMmIwNTlhYjQyNzMxLnNldENvbnRlbnQoaHRtbF83N2FmMzkxNzlkYzU0OTQ4ODljNzBkNTI3ZjNjZGU2ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYjhhMGRmYTI0ZTY0Yzg0YTA1YThkMTNiMWIwYWRhMC5iaW5kUG9wdXAocG9wdXBfMzQyYWYwNjQ1ZTljNDNiMmJmZGUyYjA1OWFiNDI3MzEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWRhYjBhYTE4NWFiNDU3ZmE2ODNkMDM5NzZkYWE5Y2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTQzMjgzLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzRlNzJhYTFhNzk2NDdmNDg5N2M5NjU0OGFlMzcyYjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmZhZWZkN2M0N2M4NGJmODk5ZDlkY2Q3Nzg3ZDJhMDAgPSAkKCc8ZGl2IGlkPSJodG1sXzJmYWVmZDdjNDdjODRiZjg5OWQ5ZGNkNzc4N2QyYTAwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XaWxzb24gSGVpZ2h0cywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzRlNzJhYTFhNzk2NDdmNDg5N2M5NjU0OGFlMzcyYjQuc2V0Q29udGVudChodG1sXzJmYWVmZDdjNDdjODRiZjg5OWQ5ZGNkNzc4N2QyYTAwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzlkYWIwYWExODVhYjQ1N2ZhNjgzZDAzOTc2ZGFhOWNlLmJpbmRQb3B1cChwb3B1cF83NGU3MmFhMWE3OTY0N2Y0ODk3Yzk2NTQ4YWUzNzJiNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kOGRhNmI1ODk0NzI0ZTZkOWMyY2I0NmRmYTIxMzc5OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcwNTM2ODksLTc5LjM0OTM3MTkwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjRjZGMwMWFlMzdkNDc1NTliN2ExM2I2MjEzNzcwOTEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZWFjZjk2NzkwZWIwNGU1ZDk5MDdlMjMwMjUyMzM0MDkgPSAkKCc8ZGl2IGlkPSJodG1sX2VhY2Y5Njc5MGViMDRlNWQ5OTA3ZTIzMDI1MjMzNDA5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaG9ybmNsaWZmZSBQYXJrLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2I0Y2RjMDFhZTM3ZDQ3NTU5YjdhMTNiNjIxMzc3MDkxLnNldENvbnRlbnQoaHRtbF9lYWNmOTY3OTBlYjA0ZTVkOTkwN2UyMzAyNTIzMzQwOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kOGRhNmI1ODk0NzI0ZTZkOWMyY2I0NmRmYTIxMzc5OC5iaW5kUG9wdXAocG9wdXBfYjRjZGMwMWFlMzdkNDc1NTliN2ExM2I2MjEzNzcwOTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDI4ZmUyNTMwMDM4NDBhM2JhOWM5OGI3NTM4MjRiYWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdjMWMyODE3NzBiYzQxMTk4NjkzMDk1OGQzNTI0YWJiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I5NjlmY2RmMjI3ZTQxNmViMmUyNDcxNmNkYWRjZjE3ID0gJCgnPGRpdiBpZD0iaHRtbF9iOTY5ZmNkZjIyN2U0MTZlYjJlMjQ3MTZjZGFkY2YxNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWRlbGFpZGUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdjMWMyODE3NzBiYzQxMTk4NjkzMDk1OGQzNTI0YWJiLnNldENvbnRlbnQoaHRtbF9iOTY5ZmNkZjIyN2U0MTZlYjJlMjQ3MTZjZGFkY2YxNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MjhmZTI1MzAwMzg0MGEzYmE5Yzk4Yjc1MzgyNGJhZS5iaW5kUG9wdXAocG9wdXBfN2MxYzI4MTc3MGJjNDExOTg2OTMwOTU4ZDM1MjRhYmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmJhYTljNTdmYzdjNDNiMzk5YTI2NzU4M2JkNjI4NjAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU4ODg4Zjg0MjA5NTQ1YWU5ZjkzMmU3MDc0ZGQ4NDAyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhjMGQzYjM2M2U1MDRmMzRiYjM1NTQzM2Y1NzI1NTcxID0gJCgnPGRpdiBpZD0iaHRtbF84YzBkM2IzNjNlNTA0ZjM0YmIzNTU0MzNmNTcyNTU3MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZywgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTg4ODhmODQyMDk1NDVhZTlmOTMyZTcwNzRkZDg0MDIuc2V0Q29udGVudChodG1sXzhjMGQzYjM2M2U1MDRmMzRiYjM1NTQzM2Y1NzI1NTcxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZiYWE5YzU3ZmM3YzQzYjM5OWEyNjc1ODNiZDYyODYwLmJpbmRQb3B1cChwb3B1cF81ODg4OGY4NDIwOTU0NWFlOWY5MzJlNzA3NGRkODQwMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85MWZmYWMzNDUzMzM0NzVlYjlhYjQxNTQzOWM1ZjkyMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzQ2ZmYyY2E1Y2Y4NGExZjg2ZjhkNzFkZjQyNjc4MmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMmZlMWMwYWQ3YjM3NGYxMTllMWZhOTBhM2ZmNGUyOTcgPSAkKCc8ZGl2IGlkPSJodG1sXzJmZTFjMGFkN2IzNzRmMTE5ZTFmYTkwYTNmZjRlMjk3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SaWNobW9uZCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzQ2ZmYyY2E1Y2Y4NGExZjg2ZjhkNzFkZjQyNjc4MmMuc2V0Q29udGVudChodG1sXzJmZTFjMGFkN2IzNzRmMTE5ZTFmYTkwYTNmZjRlMjk3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkxZmZhYzM0NTMzMzQ3NWViOWFiNDE1NDM5YzVmOTIwLmJpbmRQb3B1cChwb3B1cF9jNDZmZjJjYTVjZjg0YTFmODZmOGQ3MWRmNDI2NzgyYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82YTg1ZmQ5YjM3ZDE0MzM0YWZmNmJjMjFkZDNkYmY1MCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTAwNTEwMDAwMDAxLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzMwYjA4ZmY3ODZmNDRlYTliNzg2MWY0OTIwM2I3NmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNGM1MjE5NzI5M2M0NGQ4NDk1MTViZmJmOTViODhjYTUgPSAkKCc8ZGl2IGlkPSJodG1sXzRjNTIxOTcyOTNjNDRkODQ5NTE1YmZiZjk1Yjg4Y2E1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3ZlcmNvdXJ0IFZpbGxhZ2UsIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzMwYjA4ZmY3ODZmNDRlYTliNzg2MWY0OTIwM2I3NmYuc2V0Q29udGVudChodG1sXzRjNTIxOTcyOTNjNDRkODQ5NTE1YmZiZjk1Yjg4Y2E1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZhODVmZDliMzdkMTQzMzRhZmY2YmMyMWRkM2RiZjUwLmJpbmRQb3B1cChwb3B1cF8zMzBiMDhmZjc4NmY0NGVhOWI3ODYxZjQ5MjAzYjc2Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMmZlMWNhMWFhOWU0ODhkOTQ1NzI2NDUzNGYzMjY0NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTAwNTEwMDAwMDAxLC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTIwOTliNzRiZGVmNDFhYjg1M2JiYjVkOWJkMTUzYWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTI0ZjA5MDA4Yzg3NDU4M2I2NTlkNjMyZWIyYmNjN2QgPSAkKCc8ZGl2IGlkPSJodG1sXzUyNGYwOTAwOGM4NzQ1ODNiNjU5ZDYzMmViMmJjYzdkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EdWZmZXJpbiwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85MjA5OWI3NGJkZWY0MWFiODUzYmJiNWQ5YmQxNTNhYy5zZXRDb250ZW50KGh0bWxfNTI0ZjA5MDA4Yzg3NDU4M2I2NTlkNjMyZWIyYmNjN2QpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTJmZTFjYTFhYTllNDg4ZDk0NTcyNjQ1MzRmMzI2NDUuYmluZFBvcHVwKHBvcHVwXzkyMDk5Yjc0YmRlZjQxYWI4NTNiYmI1ZDliZDE1M2FjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2YwNjdkODc4MjgyZjRmMTM4NGFiYTY0NzNlZjE5OTgyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzQ0NzM0MiwtNzkuMjM5NDc2MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wNDcxNDkzMjgxY2I0YjRmOGZiMDBjNTUyMTNkNTQ4MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTQ4ZTk4MmZkY2U0YjUyOWQyMGUyOTY1YzBiOTNjMCA9ICQoJzxkaXYgaWQ9Imh0bWxfN2E0OGU5ODJmZGNlNGI1MjlkMjBlMjk2NWMwYjkzYzAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNjYXJib3JvdWdoIFZpbGxhZ2UsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNDcxNDkzMjgxY2I0YjRmOGZiMDBjNTUyMTNkNTQ4MS5zZXRDb250ZW50KGh0bWxfN2E0OGU5ODJmZGNlNGI1MjlkMjBlMjk2NWMwYjkzYzApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjA2N2Q4NzgyODJmNGYxMzg0YWJhNjQ3M2VmMTk5ODIuYmluZFBvcHVwKHBvcHVwXzA0NzE0OTMyODFjYjRiNGY4ZmIwMGM1NTIxM2Q1NDgxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQxZjJkMjVhZThhMTQ0ZjQ4NDk4NGVlZWY2NzUxYWQ4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzc4NTE3NSwtNzkuMzQ2NTU1N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZhMWMyMDIwZmU5OTRjMTk5ZjllMTM1N2YxYzA1YjhiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkwMmZjMzEwYzRmMzQwMTM5MzMzYThlM2ViMTZhMWQxID0gJCgnPGRpdiBpZD0iaHRtbF85MDJmYzMxMGM0ZjM0MDEzOTMzM2E4ZTNlYjE2YTFkMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RmFpcnZpZXcsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZhMWMyMDIwZmU5OTRjMTk5ZjllMTM1N2YxYzA1YjhiLnNldENvbnRlbnQoaHRtbF85MDJmYzMxMGM0ZjM0MDEzOTMzM2E4ZTNlYjE2YTFkMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MWYyZDI1YWU4YTE0NGY0ODQ5ODRlZWVmNjc1MWFkOC5iaW5kUG9wdXAocG9wdXBfNmExYzIwMjBmZTk5NGMxOTlmOWUxMzU3ZjFjMDViOGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODcwNzlkMWZhN2Q4NDM3YWFjYTBhYTNiOGU2M2YxYjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Nzg1MTc1LC03OS4zNDY1NTU3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjQyNThmMWExY2MzNGY2MzlmYWUzODFjNDI4YTNhNmUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWRjODc0NjZhNzAyNGFhYmFiZGMzYjdkMzY3NzhkMjQgPSAkKCc8ZGl2IGlkPSJodG1sXzFkYzg3NDY2YTcwMjRhYWJhYmRjM2I3ZDM2Nzc4ZDI0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IZW5yeSBGYXJtLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mNDI1OGYxYTFjYzM0ZjYzOWZhZTM4MWM0MjhhM2E2ZS5zZXRDb250ZW50KGh0bWxfMWRjODc0NjZhNzAyNGFhYmFiZGMzYjdkMzY3NzhkMjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODcwNzlkMWZhN2Q4NDM3YWFjYTBhYTNiOGU2M2YxYjIuYmluZFBvcHVwKHBvcHVwX2Y0MjU4ZjFhMWNjMzRmNjM5ZmFlMzgxYzQyOGEzYTZlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2UyYTRiZmVjMWRlZjRkMGE4YzlhZThjNjhkNTRhMTY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzc4NTE3NSwtNzkuMzQ2NTU1N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZjYzNmY2ZhYWUyODQ3MDY4YWM1YWYwMjhmNTBiNTE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQ2NWRkMzMxNGQ0MDRlM2E5ZmZjODNiOWU2ZWU0Y2E0ID0gJCgnPGRpdiBpZD0iaHRtbF80NjVkZDMzMTRkNDA0ZTNhOWZmYzgzYjllNmVlNGNhNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+T3Jpb2xlLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82Y2MzZmNmYWFlMjg0NzA2OGFjNWFmMDI4ZjUwYjUxNi5zZXRDb250ZW50KGh0bWxfNDY1ZGQzMzE0ZDQwNGUzYTlmZmM4M2I5ZTZlZTRjYTQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTJhNGJmZWMxZGVmNGQwYThjOWFlOGM2OGQ1NGExNjQuYmluZFBvcHVwKHBvcHVwXzZjYzNmY2ZhYWUyODQ3MDY4YWM1YWYwMjhmNTBiNTE2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M4NTcxZWI4NDc2NjQ2YzNhYWQ5NzYzODU0N2MyZTFiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzY3OTgwMywtNzkuNDg3MjYxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMzAzMTMzYTA4MmM0OWUzYjMyZGIwMDJhYzUxNGE5MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZGI0Y2FlOTViYWE0MTFjOGFjM2M0NTQ5MDA2Mzk2NCA9ICQoJzxkaXYgaWQ9Imh0bWxfZmRiNGNhZTk1YmFhNDExYzhhYzNjNDU0OTAwNjM5NjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk5vcnRod29vZCBQYXJrLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jMzAzMTMzYTA4MmM0OWUzYjMyZGIwMDJhYzUxNGE5My5zZXRDb250ZW50KGh0bWxfZmRiNGNhZTk1YmFhNDExYzhhYzNjNDU0OTAwNjM5NjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzg1NzFlYjg0NzY2NDZjM2FhZDk3NjM4NTQ3YzJlMWIuYmluZFBvcHVwKHBvcHVwX2MzMDMxMzNhMDgyYzQ5ZTNiMzJkYjAwMmFjNTE0YTkzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QwZWM4ZDg5M2ZmOTRkYmViMjNhMWJjMDM1OTY3MGEzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzY3OTgwMywtNzkuNDg3MjYxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MTUyYWNmMjY4OWI0Yzg3YTEyODRmZGExYjkwYTM5ZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mOTUxZjZhYWExMGE0YWJiYmQ3ZjNmMmRiMGZjODAxZSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjk1MWY2YWFhMTBhNGFiYmJkN2YzZjJkYjBmYzgwMWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPllvcmsgVW5pdmVyc2l0eSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzE1MmFjZjI2ODliNGM4N2ExMjg0ZmRhMWI5MGEzOWYuc2V0Q29udGVudChodG1sX2Y5NTFmNmFhYTEwYTRhYmJiZDdmM2YyZGIwZmM4MDFlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QwZWM4ZDg5M2ZmOTRkYmViMjNhMWJjMDM1OTY3MGEzLmJpbmRQb3B1cChwb3B1cF83MTUyYWNmMjY4OWI0Yzg3YTEyODRmZGExYjkwYTM5Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMTgyMTI0MzNlYjU0N2JjOWNhZTI1MDU3MDQ1MTEzZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NTM0NywtNzkuMzM4MTA2NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzVhMTk1N2M3ZmQ2NDQ3NTFiMTQ5NmU4MGExZGQ2NWVhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JhNDdlYzc4MjQyZDQxMWJhMjJiYmZkNGMzNjJmY2E4ID0gJCgnPGRpdiBpZD0iaHRtbF9iYTQ3ZWM3ODI0MmQ0MTFiYTIyYmJmZDRjMzYyZmNhOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBUb3JvbnRvLCBFYXN0IFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVhMTk1N2M3ZmQ2NDQ3NTFiMTQ5NmU4MGExZGQ2NWVhLnNldENvbnRlbnQoaHRtbF9iYTQ3ZWM3ODI0MmQ0MTFiYTIyYmJmZDRjMzYyZmNhOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMTgyMTI0MzNlYjU0N2JjOWNhZTI1MDU3MDQ1MTEzZS5iaW5kUG9wdXAocG9wdXBfNWExOTU3YzdmZDY0NDc1MWIxNDk2ZTgwYTFkZDY1ZWEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGFkYWY5OGE4N2RmNDY3ZWE1YmNlZDY4ZWY2MDkzN2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzhkM2JkMjNiZWJiMTQ2YWI5MmEyMDljNjgwMWY3ZjE0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzg5NzBmMmExMDdkOTRhNDM4NDQ3Y2Q0MDNmMjNiYTIwID0gJCgnPGRpdiBpZD0iaHRtbF84OTcwZjJhMTA3ZDk0YTQzODQ0N2NkNDAzZjIzYmEyMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IEVhc3QsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzhkM2JkMjNiZWJiMTQ2YWI5MmEyMDljNjgwMWY3ZjE0LnNldENvbnRlbnQoaHRtbF84OTcwZjJhMTA3ZDk0YTQzODQ0N2NkNDAzZjIzYmEyMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YWRhZjk4YTg3ZGY0NjdlYTViY2VkNjhlZjYwOTM3ZS5iaW5kUG9wdXAocG9wdXBfOGQzYmQyM2JlYmIxNDZhYjkyYTIwOWM2ODAxZjdmMTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTc1NTVlM2NjMjcwNGVhOGEyNjIwMjY2NzU1MDEwY2YgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU5NzFlMjIxNjg1ZDQzYTI4NTQ1ZmExMDExNjMzNzhiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZjMzYzZGU0YTQ5NDRkMGFiMGEzNzAzYjU0OWM1NGU2ID0gJCgnPGRpdiBpZD0iaHRtbF9mYzM2M2RlNGE0OTQ0ZDBhYjBhMzcwM2I1NDljNTRlNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBJc2xhbmRzLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81OTcxZTIyMTY4NWQ0M2EyODU0NWZhMTAxMTYzMzc4Yi5zZXRDb250ZW50KGh0bWxfZmMzNjNkZTRhNDk0NGQwYWIwYTM3MDNiNTQ5YzU0ZTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTc1NTVlM2NjMjcwNGVhOGEyNjIwMjY2NzU1MDEwY2YuYmluZFBvcHVwKHBvcHVwXzU5NzFlMjIxNjg1ZDQzYTI4NTQ1ZmExMDExNjMzNzhiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQwNDFiYmM0OTYzZjRmY2NiZTBkMTBmY2ExY2FjOWEwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MzY2OWIxOTJhYmE0YjM5YTc2ZTQ5MTRkNjJjMjRlZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80MjU1YTg4ZmJkOGQ0ODY2ODUwZjdmMTAxMWM1ZmViMiA9ICQoJzxkaXYgaWQ9Imh0bWxfNDI1NWE4OGZiZDhkNDg2Njg1MGY3ZjEwMTFjNWZlYjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaW9uIFN0YXRpb24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzUzNjY5YjE5MmFiYTRiMzlhNzZlNDkxNGQ2MmMyNGVkLnNldENvbnRlbnQoaHRtbF80MjU1YTg4ZmJkOGQ0ODY2ODUwZjdmMTAxMWM1ZmViMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MDQxYmJjNDk2M2Y0ZmNjYmUwZDEwZmNhMWNhYzlhMC5iaW5kUG9wdXAocG9wdXBfNTM2NjliMTkyYWJhNGIzOWE3NmU0OTE0ZDYyYzI0ZWQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjM4MzM2OTk5OWZjNDI4Y2JhNmQxNzFjYTI0NmU4ZjkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDc5MjY3MDAwMDAwMDYsLTc5LjQxOTc0OTddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85Njc3NmM1NWFhZWQ0ZmU1OTE3MWNlOTI1Y2E1NzU2ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZjcxMThjNmY4NGI0NGM3YTQ4ZDdhNzExOGY2NWNjYSA9ICQoJzxkaXYgaWQ9Imh0bWxfMmY3MTE4YzZmODRiNDRjN2E0OGQ3YTcxMThmNjVjY2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkxpdHRsZSBQb3J0dWdhbCwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85Njc3NmM1NWFhZWQ0ZmU1OTE3MWNlOTI1Y2E1NzU2ZC5zZXRDb250ZW50KGh0bWxfMmY3MTE4YzZmODRiNDRjN2E0OGQ3YTcxMThmNjVjY2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjM4MzM2OTk5OWZjNDI4Y2JhNmQxNzFjYTI0NmU4ZjkuYmluZFBvcHVwKHBvcHVwXzk2Nzc2YzU1YWFlZDRmZTU5MTcxY2U5MjVjYTU3NTZkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2YxMTg1MThjNWQxNDRiN2Y4YTlmOTRjYWJlNWQxMGM1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3OTI2NzAwMDAwMDA2LC03OS40MTk3NDk3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZTAzZjllYTY2NDkwNGU2OTgzZGRhMDgyODBhNTBmODQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWZmYWViZDk2NmIwNDVlMGFlNDU5NDBjYmEzNWVlNGIgPSAkKCc8ZGl2IGlkPSJodG1sXzFmZmFlYmQ5NjZiMDQ1ZTBhZTQ1OTQwY2JhMzVlZTRiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UcmluaXR5LCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UwM2Y5ZWE2NjQ5MDRlNjk4M2RkYTA4MjgwYTUwZjg0LnNldENvbnRlbnQoaHRtbF8xZmZhZWJkOTY2YjA0NWUwYWU0NTk0MGNiYTM1ZWU0Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMTE4NTE4YzVkMTQ0YjdmOGE5Zjk0Y2FiZTVkMTBjNS5iaW5kUG9wdXAocG9wdXBfZTAzZjllYTY2NDkwNGU2OTgzZGRhMDgyODBhNTBmODQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmRkYWU3ZGJhODQ2NDI5YmIyN2M0MzMyYWQxYTRlYjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mjc5MjkyLC03OS4yNjIwMjk0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QwMjljN2NiNmQzYzRmZTA5MjlkNTA1MjljY2IxMTM0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EzMjg2ZjcwNmJkMTRkODI4MjUxNmI2M2E3MTc2ZWU2ID0gJCgnPGRpdiBpZD0iaHRtbF9hMzI4NmY3MDZiZDE0ZDgyODI1MTZiNjNhNzE3NmVlNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RWFzdCBCaXJjaG1vdW50IFBhcmssIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMDI5YzdjYjZkM2M0ZmUwOTI5ZDUwNTI5Y2NiMTEzNC5zZXRDb250ZW50KGh0bWxfYTMyODZmNzA2YmQxNGQ4MjgyNTE2YjYzYTcxNzZlZTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmRkYWU3ZGJhODQ2NDI5YmIyN2M0MzMyYWQxYTRlYjYuYmluZFBvcHVwKHBvcHVwX2QwMjljN2NiNmQzYzRmZTA5MjlkNTA1MjljY2IxMTM0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzkxZjVmNmE4MjllMzRkYTZhMThiZDNiZTE3Y2FhMjkyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI3OTI5MiwtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85ZDEzYWNmNmY4YjE0ZDFiOTcwOTYzMTc2ZmJhNzRlNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83OTlkYTM5ZDRhOTk0MjU3OTEzNmNjODE1YTc4OGZlZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNzk5ZGEzOWQ0YTk5NDI1NzkxMzZjYzgxNWE3ODhmZWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPklvbnZpZXcsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ZDEzYWNmNmY4YjE0ZDFiOTcwOTYzMTc2ZmJhNzRlNi5zZXRDb250ZW50KGh0bWxfNzk5ZGEzOWQ0YTk5NDI1NzkxMzZjYzgxNWE3ODhmZWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTFmNWY2YTgyOWUzNGRhNmExOGJkM2JlMTdjYWEyOTIuYmluZFBvcHVwKHBvcHVwXzlkMTNhY2Y2ZjhiMTRkMWI5NzA5NjMxNzZmYmE3NGU2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NlNWE2MmZhOTZhODQwM2NiNGIzY2Q0OGEzODRkYzg5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI3OTI5MiwtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lZGM5NDQzMzYyMjg0MTZkOGQ1NDJiN2U2NDIyZTk5MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NDJlODI3OTEwZGM0ZDllODY3ZGNjN2ExMTU2YjQ5NSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTQyZTgyNzkxMGRjNGQ5ZTg2N2RjYzdhMTE1NmI0OTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktlbm5lZHkgUGFyaywgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VkYzk0NDMzNjIyODQxNmQ4ZDU0MmI3ZTY0MjJlOTkxLnNldENvbnRlbnQoaHRtbF85NDJlODI3OTEwZGM0ZDllODY3ZGNjN2ExMTU2YjQ5NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZTVhNjJmYTk2YTg0MDNjYjRiM2NkNDhhMzg0ZGM4OS5iaW5kUG9wdXAocG9wdXBfZWRjOTQ0MzM2MjI4NDE2ZDhkNTQyYjdlNjQyMmU5OTEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDMwYWJjYzU3NzE0NGRiYWFjZGQ2MGZlZGU2NTMyNmQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODY5NDczLC03OS4zODU5NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYjZlNDcyMmY0NWU0M2UwODYyMGFhNzU4ZWVjZjYyMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81ZjU4MjIyYmNiNDc0N2MwYWQzMDkxZmY1MzYyYzdhMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNWY1ODIyMmJjYjQ3NDdjMGFkMzA5MWZmNTM2MmM3YTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJheXZpZXcgVmlsbGFnZSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2I2ZTQ3MjJmNDVlNDNlMDg2MjBhYTc1OGVlY2Y2MjMuc2V0Q29udGVudChodG1sXzVmNTgyMjJiY2I0NzQ3YzBhZDMwOTFmZjUzNjJjN2ExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzAzMGFiY2M1NzcxNDRkYmFhY2RkNjBmZWRlNjUzMjZkLmJpbmRQb3B1cChwb3B1cF9jYjZlNDcyMmY0NWU0M2UwODYyMGFhNzU4ZWVjZjYyMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9iODBhZDFlZjdkNDE0MzAxODllZjEzNDE4ZGY5ZWM5NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczNzQ3MzIwMDAwMDAwNCwtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81MTllNDdhZGQ0YTA0MTQ1ODY5NDgzY2UyZGJmN2VmNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85NDY4Njk3ZDI2NDY0Y2RlOTI1ODkxZjljMWY4MGNhZSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTQ2ODY5N2QyNjQ2NGNkZTkyNTg5MWY5YzFmODBjYWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNGQiBUb3JvbnRvLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MTllNDdhZGQ0YTA0MTQ1ODY5NDgzY2UyZGJmN2VmNC5zZXRDb250ZW50KGh0bWxfOTQ2ODY5N2QyNjQ2NGNkZTkyNTg5MWY5YzFmODBjYWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYjgwYWQxZWY3ZDQxNDMwMTg5ZWYxMzQxOGRmOWVjOTYuYmluZFBvcHVwKHBvcHVwXzUxOWU0N2FkZDRhMDQxNDU4Njk0ODNjZTJkYmY3ZWY0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2IzMDFjMTljNGNlYjQzY2NiMTYyYjlhZGRlMzU0YWZiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM3NDczMjAwMDAwMDA0LC03OS40NjQ3NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg1NDUxOWE0MTNkNjQzMzE4NWMxMzNiYmYzZWVmZWE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAzNDgwYjUzYzBkZDQ0MGM4MGJlMzY3ZDlmMzhmZDU3ID0gJCgnPGRpdiBpZD0iaHRtbF8wMzQ4MGI1M2MwZGQ0NDBjODBiZTM2N2Q5ZjM4ZmQ1NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3IEVhc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg1NDUxOWE0MTNkNjQzMzE4NWMxMzNiYmYzZWVmZWE2LnNldENvbnRlbnQoaHRtbF8wMzQ4MGI1M2MwZGQ0NDBjODBiZTM2N2Q5ZjM4ZmQ1Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMzAxYzE5YzRjZWI0M2NjYjE2MmI5YWRkZTM1NGFmYi5iaW5kUG9wdXAocG9wdXBfODU0NTE5YTQxM2Q2NDMzMTg1YzEzM2JiZjNlZWZlYTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWYzOWM4MjQ1NWZjNDg5Mzk1NDA0NWY0NzI0MDBjZWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mMTlhMGEwYTI3NDM0NjFkOTZhZGFjZmQ1NTI2YzRmYiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZjlhYjc2ZDZjMWQ0YzM1YjI5MjY0YTVjMmY5MWUyMyA9ICQoJzxkaXYgaWQ9Imh0bWxfMmY5YWI3NmQ2YzFkNGMzNWIyOTI2NGE1YzJmOTFlMjMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBEYW5mb3J0aCBXZXN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YxOWEwYTBhMjc0MzQ2MWQ5NmFkYWNmZDU1MjZjNGZiLnNldENvbnRlbnQoaHRtbF8yZjlhYjc2ZDZjMWQ0YzM1YjI5MjY0YTVjMmY5MWUyMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xZjM5YzgyNDU1ZmM0ODkzOTU0MDQ1ZjQ3MjQwMGNlYS5iaW5kUG9wdXAocG9wdXBfZjE5YTBhMGEyNzQzNDYxZDk2YWRhY2ZkNTUyNmM0ZmIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTdmZTNlMjFhMmM1NDc5MmI1MzQ5Njg0OTk1NWZlYWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Nzk1NTcxLC03OS4zNTIxODhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMmI2YjA1OTNiZDc0NzliYTEzMjEwZDhiMjcxYjE2ZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xZTgzNTI2OTRjZDY0YzRjYWI4OGNlYjYyNTU3N2U4OCA9ICQoJzxkaXYgaWQ9Imh0bWxfMWU4MzUyNjk0Y2Q2NGM0Y2FiODhjZWI2MjU1NzdlODgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJpdmVyZGFsZSwgRWFzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMmI2YjA1OTNiZDc0NzliYTEzMjEwZDhiMjcxYjE2ZS5zZXRDb250ZW50KGh0bWxfMWU4MzUyNjk0Y2Q2NGM0Y2FiODhjZWI2MjU1NzdlODgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTdmZTNlMjFhMmM1NDc5MmI1MzQ5Njg0OTk1NWZlYWQuYmluZFBvcHVwKHBvcHVwXzMyYjZiMDU5M2JkNzQ3OWJhMTMyMTBkOGIyNzFiMTZlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg4ZTdiNTE5NTEyZTRlYTI4ZmRlNGI5ODU4OGFjNGQ5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ3MTc2OCwtNzkuMzgxNTc2NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MWZkNjg1ZThhOWU0OWM1YTU4OWEyNjIxNjAyNjFiNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iYjdiNGM4NTNmMDI0YjY5YmM5MDFhZTU0N2E1MDMxYyA9ICQoJzxkaXYgaWQ9Imh0bWxfYmI3YjRjODUzZjAyNGI2OWJjOTAxYWU1NDdhNTAzMWMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRlc2lnbiBFeGNoYW5nZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjFmZDY4NWU4YTllNDljNWE1ODlhMjYyMTYwMjYxYjcuc2V0Q29udGVudChodG1sX2JiN2I0Yzg1M2YwMjRiNjliYzkwMWFlNTQ3YTUwMzFjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg4ZTdiNTE5NTEyZTRlYTI4ZmRlNGI5ODU4OGFjNGQ5LmJpbmRQb3B1cChwb3B1cF82MWZkNjg1ZThhOWU0OWM1YTU4OWEyNjIxNjAyNjFiNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MzU2NTZmOWUyY2M0NDQ1ODRjZTYwNmRiYWQ3NDdjMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDEwYjMyNTljOGU2NGI3Yzk3YWU4Zjg1NWJlODE3NDcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2QzNTY1NzU2Yzg1NGIwN2IzM2I2YWQ5OTUyYzA1MzYgPSAkKCc8ZGl2IGlkPSJodG1sXzdkMzU2NTc1NmM4NTRiMDdiMzNiNmFkOTk1MmMwNTM2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ub3JvbnRvIERvbWluaW9uIENlbnRyZSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDEwYjMyNTljOGU2NGI3Yzk3YWU4Zjg1NWJlODE3NDcuc2V0Q29udGVudChodG1sXzdkMzU2NTc1NmM4NTRiMDdiMzNiNmFkOTk1MmMwNTM2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgzNTY1NmY5ZTJjYzQ0NDU4NGNlNjA2ZGJhZDc0N2MxLmJpbmRQb3B1cChwb3B1cF80MTBiMzI1OWM4ZTY0YjdjOTdhZThmODU1YmU4MTc0Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80NTI0MjE0NzgyMjU0NzczYjQzNzM3MzFiMGM4ZmVhMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjg0NzIsLTc5LjQyODE5MTQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjZiOWNiOTVjNThiNDU2NGI4YTU1MjA2MmQzNWU5ZTggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzg4MmUwYTlmMGQzNGQwYjlhY2FlNDNiZjEyYzJhMTcgPSAkKCc8ZGl2IGlkPSJodG1sXzM4ODJlMGE5ZjBkMzRkMGI5YWNhZTQzYmYxMmMyYTE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ccm9ja3RvbiwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNmI5Y2I5NWM1OGI0NTY0YjhhNTUyMDYyZDM1ZTllOC5zZXRDb250ZW50KGh0bWxfMzg4MmUwYTlmMGQzNGQwYjlhY2FlNDNiZjEyYzJhMTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDUyNDIxNDc4MjI1NDc3M2I0MzczNzMxYjBjOGZlYTEuYmluZFBvcHVwKHBvcHVwX2I2YjljYjk1YzU4YjQ1NjRiOGE1NTIwNjJkMzVlOWU4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY2ZTYxZmYzNzFhMDQ1ODliNTAxZjVhMGNlYWY3YzhkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2ODQ3MiwtNzkuNDI4MTkxNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jMmUxZTg2ZjZjNGY0NzNlYWFiOGNmOGI1ZWQ5NmE1MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNTdkYWQyMmJiNDY0NDQyYWVhYWMyMmZmNmQ2YThmOSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTU3ZGFkMjJiYjQ2NDQ0MmFlYWFjMjJmZjZkNmE4ZjkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkV4aGliaXRpb24gUGxhY2UsIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzJlMWU4NmY2YzRmNDczZWFhYjhjZjhiNWVkOTZhNTIuc2V0Q29udGVudChodG1sXzE1N2RhZDIyYmI0NjQ0NDJhZWFhYzIyZmY2ZDZhOGY5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzY2ZTYxZmYzNzFhMDQ1ODliNTAxZjVhMGNlYWY3YzhkLmJpbmRQb3B1cChwb3B1cF9jMmUxZTg2ZjZjNGY0NzNlYWFiOGNmOGI1ZWQ5NmE1Mik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNzk5Y2I4YmYyM2I0MzcxOGQ1MjI3MDQwMDZhZGY0NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjg0NzIsLTc5LjQyODE5MTQwMDAwMDAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMGM2ZGE4NGVjMmJjNDk2ODgwN2YwYTBmMGY1MzdiNjcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjA1MGI4NDdlN2Y5NDUzNDhkMjgzNTdhYzRkMjQyYTcgPSAkKCc8ZGl2IGlkPSJodG1sX2YwNTBiODQ3ZTdmOTQ1MzQ4ZDI4MzU3YWM0ZDI0MmE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrZGFsZSBWaWxsYWdlLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBjNmRhODRlYzJiYzQ5Njg4MDdmMGEwZjBmNTM3YjY3LnNldENvbnRlbnQoaHRtbF9mMDUwYjg0N2U3Zjk0NTM0OGQyODM1N2FjNGQyNDJhNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNzk5Y2I4YmYyM2I0MzcxOGQ1MjI3MDQwMDZhZGY0Ni5iaW5kUG9wdXAocG9wdXBfMGM2ZGE4NGVjMmJjNDk2ODgwN2YwYTBmMGY1MzdiNjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjUzMmYxZDI5Y2M2NDk4MWE2NDhhZjAzMDhjOTJhYjcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTExMTE3MDAwMDAwMDQsLTc5LjI4NDU3NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82MmQ3NWJkODVhYTU0NmVmOWYzZjE4Mjc0NjE5ZDJlZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMWJmNDczYjRiNWY0NWY1ODM3ZTAwYTFjNzg4NDc4NCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzFiZjQ3M2I0YjVmNDVmNTgzN2UwMGExYzc4ODQ3ODQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsYWlybGVhLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjJkNzViZDg1YWE1NDZlZjlmM2YxODI3NDYxOWQyZWQuc2V0Q29udGVudChodG1sX2MxYmY0NzNiNGI1ZjQ1ZjU4MzdlMDBhMWM3ODg0Nzg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y1MzJmMWQyOWNjNjQ5ODFhNjQ4YWYwMzA4YzkyYWI3LmJpbmRQb3B1cChwb3B1cF82MmQ3NWJkODVhYTU0NmVmOWYzZjE4Mjc0NjE5ZDJlZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84ZGM5NjhkMzJkODA0MDUxOTgzOTM0MzE1MmJjZGZhYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMTExMTcwMDAwMDAwNCwtNzkuMjg0NTc3Ml0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA3MzI3NmRjNDE4YTQwNTc5ZGUzNzI5NzhlODA3ZTVjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZlNWE0OWI5ZWJkYzQwZTlhNGNmYmNiNWNmZjdhMjU4ID0gJCgnPGRpdiBpZD0iaHRtbF9mZTVhNDliOWViZGM0MGU5YTRjZmJjYjVjZmY3YTI1OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R29sZGVuIE1pbGUsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNzMyNzZkYzQxOGE0MDU3OWRlMzcyOTc4ZTgwN2U1Yy5zZXRDb250ZW50KGh0bWxfZmU1YTQ5YjllYmRjNDBlOWE0Y2ZiY2I1Y2ZmN2EyNTgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGRjOTY4ZDMyZDgwNDA1MTk4MzkzNDMxNTJiY2RmYWMuYmluZFBvcHVwKHBvcHVwXzA3MzI3NmRjNDE4YTQwNTc5ZGUzNzI5NzhlODA3ZTVjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk5YzE5MzkxNjE4ZDQ4ZjlhZmMzZjcyNTYzMDgwNThkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExMTExNzAwMDAwMDA0LC03OS4yODQ1NzcyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWRjOTQxNzU5OGUwNDIyNzhjYWRhZGQxMWViZjYyZjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTJjYzk3YTIzY2ZmNGFlYTlkYWFmN2QzYjNkZTc0ZTQgPSAkKCc8ZGl2IGlkPSJodG1sXzUyY2M5N2EyM2NmZjRhZWE5ZGFhZjdkM2IzZGU3NGU0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5PYWtyaWRnZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FkYzk0MTc1OThlMDQyMjc4Y2FkYWRkMTFlYmY2MmY0LnNldENvbnRlbnQoaHRtbF81MmNjOTdhMjNjZmY0YWVhOWRhYWY3ZDNiM2RlNzRlNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85OWMxOTM5MTYxOGQ0OGY5YWZjM2Y3MjU2MzA4MDU4ZC5iaW5kUG9wdXAocG9wdXBfYWRjOTQxNzU5OGUwNDIyNzhjYWRhZGQxMWViZjYyZjQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTQxZTliMTQ0MjhiNDM0ZTliZjdiYWIxZDQ1ZWVjZWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTc0OTAyLC03OS4zNzQ3MTQwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA2MmYxZDA1MTM0NTQ3NTM5YzI2YTBmYWI2MWFkZDgxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU5MTcyM2Q0MTRhYTQ0NDFhOTI2N2Y4YTgxNmMwYmNlID0gJCgnPGRpdiBpZD0iaHRtbF81OTE3MjNkNDE0YWE0NDQxYTkyNjdmOGE4MTZjMGJjZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U2lsdmVyIEhpbGxzLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wNjJmMWQwNTEzNDU0NzUzOWMyNmEwZmFiNjFhZGQ4MS5zZXRDb250ZW50KGh0bWxfNTkxNzIzZDQxNGFhNDQ0MWE5MjY3ZjhhODE2YzBiY2UpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTQxZTliMTQ0MjhiNDM0ZTliZjdiYWIxZDQ1ZWVjZWIuYmluZFBvcHVwKHBvcHVwXzA2MmYxZDA1MTM0NTQ3NTM5YzI2YTBmYWI2MWFkZDgxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVjNmQyMTcyNzk4NjQzMGViMzFiNjQzYzc1MWYyYmQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU3NDkwMiwtNzkuMzc0NzE0MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYTdmZTczZmFkNGQ0OTk3ODVmNzFlMWMxYzJhYzliOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNTcwMDZkNDVmMmU0NmNmYTI3YzI4YmIwNGU3MjQzMiA9ICQoJzxkaXYgaWQ9Imh0bWxfYzU3MDA2ZDQ1ZjJlNDZjZmEyN2MyOGJiMDRlNzI0MzIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPllvcmsgTWlsbHMsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNhN2ZlNzNmYWQ0ZDQ5OTc4NWY3MWUxYzFjMmFjOWI4LnNldENvbnRlbnQoaHRtbF9jNTcwMDZkNDVmMmU0NmNmYTI3YzI4YmIwNGU3MjQzMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81YzZkMjE3Mjc5ODY0MzBlYjMxYjY0M2M3NTFmMmJkMi5iaW5kUG9wdXAocG9wdXBfM2E3ZmU3M2ZhZDRkNDk5Nzg1ZjcxZTFjMWMyYWM5YjgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzNjM2E2YjM4ZDlhNDFlMGE1NDE5OTE1ODRmZDk1ZjIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MzkwMTQ2LC03OS41MDY5NDM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfN2MyOGIzYjMyMTc0NDkxY2I1NzRjNzViMjBlZjlkNTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTc1OTBkODY4Y2NkNDA4OWFhYWU4ZTlhNDI5ZmU4NTUgPSAkKCc8ZGl2IGlkPSJodG1sXzU3NTkwZDg2OGNjZDQwODlhYWFlOGU5YTQyOWZlODU1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcgV2VzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfN2MyOGIzYjMyMTc0NDkxY2I1NzRjNzViMjBlZjlkNTYuc2V0Q29udGVudChodG1sXzU3NTkwZDg2OGNjZDQwODlhYWFlOGU5YTQyOWZlODU1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzMzYzNhNmIzOGQ5YTQxZTBhNTQxOTkxNTg0ZmQ5NWYyLmJpbmRQb3B1cChwb3B1cF83YzI4YjNiMzIxNzQ0OTFjYjU3NGM3NWIyMGVmOWQ1Nik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85ZTljNzRhYzU0ODA0MzY5YTllZmYyNTRkMDM2N2ZjYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2ODk5ODUsLTc5LjMxNTU3MTU5OTk5OTk4XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWNjYzMwNmVkZWM4NGI0YWEzNDFkOTIxOGYyNGRkYWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTYxMzA5YTU3NzgyNDlhNjgyYmZhZTBiM2E4YTk3YjUgPSAkKCc8ZGl2IGlkPSJodG1sX2E2MTMwOWE1Nzc4MjQ5YTY4MmJmYWUwYjNhOGE5N2I1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgQmVhY2hlcyBXZXN0LCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FjY2MzMDZlZGVjODRiNGFhMzQxZDkyMThmMjRkZGFiLnNldENvbnRlbnQoaHRtbF9hNjEzMDlhNTc3ODI0OWE2ODJiZmFlMGIzYThhOTdiNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZTljNzRhYzU0ODA0MzY5YTllZmYyNTRkMDM2N2ZjYS5iaW5kUG9wdXAocG9wdXBfYWNjYzMwNmVkZWM4NGI0YWEzNDFkOTIxOGYyNGRkYWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmVlYjk5N2E3NzI4NGVhMThlMWI3OGE4YjVjODI3NDcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njg5OTg1LC03OS4zMTU1NzE1OTk5OTk5OF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E5OTExYzkwYjc5MTQ5ZmU5YmU4ZGJhZTNlMjQwYjk3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQwNzRlOGUxNGFiYTQwYjhhZTA2M2QyNmZmOGRmYjVlID0gJCgnPGRpdiBpZD0iaHRtbF80MDc0ZThlMTRhYmE0MGI4YWUwNjNkMjZmZjhkZmI1ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SW5kaWEgQmF6YWFyLCBFYXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2E5OTExYzkwYjc5MTQ5ZmU5YmU4ZGJhZTNlMjQwYjk3LnNldENvbnRlbnQoaHRtbF80MDc0ZThlMTRhYmE0MGI4YWUwNjNkMjZmZjhkZmI1ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mZWViOTk3YTc3Mjg0ZWExOGUxYjc4YThiNWM4Mjc0Ny5iaW5kUG9wdXAocG9wdXBfYTk5MTFjOTBiNzkxNDlmZTliZThkYmFlM2UyNDBiOTcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTZlODI3ZWJjYjUyNDE2OGEwOTU4OTlmNjJiMmIyN2QgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YwMGIwOGYyOTY2YjQ5YjA4NGVjZWM5ZDk3OTRjODdjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIwNWU0ODI0MGUwNjQ4MTY4YjMyYzcwZjMxZWYzOWRlID0gJCgnPGRpdiBpZD0iaHRtbF8yMDVlNDgyNDBlMDY0ODE2OGIzMmM3MGYzMWVmMzlkZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q29tbWVyY2UgQ291cnQsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YwMGIwOGYyOTY2YjQ5YjA4NGVjZWM5ZDk3OTRjODdjLnNldENvbnRlbnQoaHRtbF8yMDVlNDgyNDBlMDY0ODE2OGIzMmM3MGYzMWVmMzlkZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NmU4MjdlYmNiNTI0MTY4YTA5NTg5OWY2MmIyYjI3ZC5iaW5kUG9wdXAocG9wdXBfZjAwYjA4ZjI5NjZiNDliMDg0ZWNlYzlkOTc5NGM4N2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTljY2I2OGYwM2U2NDcxMjk4NWQ0MmI2NTdjNDhkOTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2FkYmVkMDk5NmVmNDQ0MDU4NTA2MjM4NTM0ZDY5YmJjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMxODhmMDc5MjY2ZjQ0NTg5NTYzYzdlMGIzYTUwMjNmID0gJCgnPGRpdiBpZD0iaHRtbF8zMTg4ZjA3OTI2NmY0NDU4OTU2M2M3ZTBiM2E1MDIzZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VmljdG9yaWEgSG90ZWwsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FkYmVkMDk5NmVmNDQ0MDU4NTA2MjM4NTM0ZDY5YmJjLnNldENvbnRlbnQoaHRtbF8zMTg4ZjA3OTI2NmY0NDU4OTU2M2M3ZTBiM2E1MDIzZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81OWNjYjY4ZjAzZTY0NzEyOTg1ZDQyYjY1N2M0OGQ5NS5iaW5kUG9wdXAocG9wdXBfYWRiZWQwOTk2ZWY0NDQwNTg1MDYyMzg1MzRkNjliYmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYWY3ODgwNGRlNDBmNGRhYWIwYmMxYTVmY2FjZjQ3YTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTM3NTYyMDAwMDAwMDYsLTc5LjQ5MDA3MzhdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YTcyNTk3N2E5YWQ0NTY5YjA4Zjg0MmEyMzU3ZjQ0YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hZWEyYmU4OWU0NWE0NWEyYmNiYTBiZDc5N2ZiYTU5ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYWVhMmJlODllNDVhNDVhMmJjYmEwYmQ3OTdmYmE1OWUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hcGxlIExlYWYgUGFyaywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfN2E3MjU5NzdhOWFkNDU2OWIwOGY4NDJhMjM1N2Y0NGIuc2V0Q29udGVudChodG1sX2FlYTJiZTg5ZTQ1YTQ1YTJiY2JhMGJkNzk3ZmJhNTllKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FmNzg4MDRkZTQwZjRkYWFiMGJjMWE1ZmNhY2Y0N2E3LmJpbmRQb3B1cChwb3B1cF83YTcyNTk3N2E5YWQ0NTY5YjA4Zjg0MmEyMzU3ZjQ0Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yYjVjZDJhMTE1MDk0M2Y2YWQ5ZGU3ZDhiZjI0YTJmMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMzc1NjIwMDAwMDAwNiwtNzkuNDkwMDczOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkyN2ZhYjFhOWFkNDQwNDViMDQ3YTgwZWY1ZGQ2NzZmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE2NDQyODE3NjNmNTRkMGNiY2VjMTRiYmQ4MGJlNjdjID0gJCgnPGRpdiBpZD0iaHRtbF8xNjQ0MjgxNzYzZjU0ZDBjYmNlYzE0YmJkODBiZTY3YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggUGFyaywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTI3ZmFiMWE5YWQ0NDA0NWIwNDdhODBlZjVkZDY3NmYuc2V0Q29udGVudChodG1sXzE2NDQyODE3NjNmNTRkMGNiY2VjMTRiYmQ4MGJlNjdjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJiNWNkMmExMTUwOTQzZjZhZDlkZTdkOGJmMjRhMmYxLmJpbmRQb3B1cChwb3B1cF85MjdmYWIxYTlhZDQ0MDQ1YjA0N2E4MGVmNWRkNjc2Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83ZWQyYmEyYzk0NjE0NWFjODJjZTIwZDU2MTJmOTIxMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxMzc1NjIwMDAwMDAwNiwtNzkuNDkwMDczOF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzlmZWI5ZjRhZmIyNjQ1NjA5YmZiN2RkZmY3OWE1MzQ2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzAxNjgzYTM1YzIzMzQ5Zjc4NmE3MjllMGZkZDI0YWM4ID0gJCgnPGRpdiBpZD0iaHRtbF8wMTY4M2EzNWMyMzM0OWY3ODZhNzI5ZTBmZGQyNGFjOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VXB3b29kIFBhcmssIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlmZWI5ZjRhZmIyNjQ1NjA5YmZiN2RkZmY3OWE1MzQ2LnNldENvbnRlbnQoaHRtbF8wMTY4M2EzNWMyMzM0OWY3ODZhNzI5ZTBmZGQyNGFjOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZWQyYmEyYzk0NjE0NWFjODJjZTIwZDU2MTJmOTIxMC5iaW5kUG9wdXAocG9wdXBfOWZlYjlmNGFmYjI2NDU2MDliZmI3ZGRmZjc5YTUzNDYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZGIyZjdjYTY1NzQxNDBhN2I1ZDQ3MWIxZGYyNjIxYjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTYzMDMzLC03OS41NjU5NjMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzc1ZWIzNTNmZGRmOTRhODM4OTYxMDI4YTdlMTg3NjBmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk2M2ExZjI5MWEwMzQ5MTVhOTEzMDAyODJmYjlkN2YxID0gJCgnPGRpdiBpZD0iaHRtbF85NjNhMWYyOTFhMDM0OTE1YTkxMzAwMjgyZmI5ZDdmMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SHVtYmVyIFN1bW1pdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzVlYjM1M2ZkZGY5NGE4Mzg5NjEwMjhhN2UxODc2MGYuc2V0Q29udGVudChodG1sXzk2M2ExZjI5MWEwMzQ5MTVhOTEzMDAyODJmYjlkN2YxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RiMmY3Y2E2NTc0MTQwYTdiNWQ0NzFiMWRmMjYyMWIzLmJpbmRQb3B1cChwb3B1cF83NWViMzUzZmRkZjk0YTgzODk2MTAyOGE3ZTE4NzYwZik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82Y2VmNTlmNGM5NmM0MmFmOWJjNjk4YzJlNTdlMzBkMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNjMxNiwtNzkuMjM5NDc2MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kOTU3YmI4NmIwZTI0ZmRlODBlNDE4MTA2NjJlNDFkNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yYzc2NGRhOTc5NjA0YzczOTgzYTJmNWNiOGUxOThlYiA9ICQoJzxkaXYgaWQ9Imh0bWxfMmM3NjRkYTk3OTYwNGM3Mzk4M2EyZjVjYjhlMTk4ZWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNsaWZmY3Jlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kOTU3YmI4NmIwZTI0ZmRlODBlNDE4MTA2NjJlNDFkNS5zZXRDb250ZW50KGh0bWxfMmM3NjRkYTk3OTYwNGM3Mzk4M2EyZjVjYjhlMTk4ZWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmNlZjU5ZjRjOTZjNDJhZjliYzY5OGMyZTU3ZTMwZDIuYmluZFBvcHVwKHBvcHVwX2Q5NTdiYjg2YjBlMjRmZGU4MGU0MTgxMDY2MmU0MWQ1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZjZjdmNDg3MTc1OTRlYTA4NTljY2FmMzM2OTQxNDEyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzE2MzE2LC03OS4yMzk0NzYwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzljZjA4YmQzOGM4ZjQ2MWZiOWIyNGJiYjU2OGZlMGM1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EzOGRkMjY3YzM2MDQ1MTVhMzNmMTU1YmZjOWY3MjU1ID0gJCgnPGRpdiBpZD0iaHRtbF9hMzhkZDI2N2MzNjA0NTE1YTMzZjE1NWJmYzlmNzI1NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2xpZmZzaWRlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOWNmMDhiZDM4YzhmNDYxZmI5YjI0YmJiNTY4ZmUwYzUuc2V0Q29udGVudChodG1sX2EzOGRkMjY3YzM2MDQ1MTVhMzNmMTU1YmZjOWY3MjU1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZjZjdmNDg3MTc1OTRlYTA4NTljY2FmMzM2OTQxNDEyLmJpbmRQb3B1cChwb3B1cF85Y2YwOGJkMzhjOGY0NjFmYjliMjRiYmI1NjhmZTBjNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83MWFkYmEzNmJiOGY0NTJiOWE0MTc1ZDhlOGQzYTdlOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjcxNjMxNiwtNzkuMjM5NDc2MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYTc5MDc3NmMwZWQ0ZGFlOTRlNTJkZTRmNDFmZjBiZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTYwOWEyNjhhYmI0YjI0OTg0NmI3OTE1OTViODRkMiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2E2MDlhMjY4YWJiNGIyNDk4NDZiNzkxNTk1Yjg0ZDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNjYXJib3JvdWdoIFZpbGxhZ2UgV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NhNzkwNzc2YzBlZDRkYWU5NGU1MmRlNGY0MWZmMGJkLnNldENvbnRlbnQoaHRtbF83YTYwOWEyNjhhYmI0YjI0OTg0NmI3OTE1OTViODRkMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MWFkYmEzNmJiOGY0NTJiOWE0MTc1ZDhlOGQzYTdlOS5iaW5kUG9wdXAocG9wdXBfY2E3OTA3NzZjMGVkNGRhZTk0ZTUyZGU0ZjQxZmYwYmQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNmExMTY2Mjc2N2RlNGEzM2E0YjAyZTRjZjM2NzMyMTEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODkwNTMsLTc5LjQwODQ5Mjc5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGUzMGRlYTM3MDA1NDFhNThlNmRmZGFjYzEzNGMyMzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjNlZWIwOWQ4ODgzNGQ2M2ExYTgwMmM5NDY4M2ZhOTAgPSAkKCc8ZGl2IGlkPSJodG1sX2IzZWViMDlkODg4MzRkNjNhMWE4MDJjOTQ2ODNmYTkwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5OZXd0b25icm9vaywgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGUzMGRlYTM3MDA1NDFhNThlNmRmZGFjYzEzNGMyMzguc2V0Q29udGVudChodG1sX2IzZWViMDlkODg4MzRkNjNhMWE4MDJjOTQ2ODNmYTkwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZhMTE2NjI3NjdkZTRhMzNhNGIwMmU0Y2YzNjczMjExLmJpbmRQb3B1cChwb3B1cF80ZTMwZGVhMzcwMDU0MWE1OGU2ZGZkYWNjMTM0YzIzOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNmY4YjUzOTIyNzk0ZjIyYmFmNjYxYjNlMjQ2NGMxYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc4OTA1MywtNzkuNDA4NDkyNzk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wZDljZDFjMDAxOGU0YTE5OGFhNDc4MTE5OTAyNGE1MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTBhZjA3OGEwOWU0NWI1YmY5MGRlYTYwYmYyMjQ4NyA9ICQoJzxkaXYgaWQ9Imh0bWxfN2EwYWYwNzhhMDllNDViNWJmOTBkZWE2MGJmMjI0ODciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPldpbGxvd2RhbGUsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzBkOWNkMWMwMDE4ZTRhMTk4YWE0NzgxMTk5MDI0YTUzLnNldENvbnRlbnQoaHRtbF83YTBhZjA3OGEwOWU0NWI1YmY5MGRlYTYwYmYyMjQ4Nyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNmY4YjUzOTIyNzk0ZjIyYmFmNjYxYjNlMjQ2NGMxYS5iaW5kUG9wdXAocG9wdXBfMGQ5Y2QxYzAwMThlNGExOThhYTQ3ODExOTkwMjRhNTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWJmZTQ3N2JmOWNkNGExMmI2MmYxNTU3NDkwM2I5MDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mjg0OTY0LC03OS40OTU2OTc0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q5ZWJmNjEyNDY2ZjRiNTJiZmRkODA3NjE4YjczNDljID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2E1NGM0MjJiYWQxYTRmM2U4Mjc2NzNhZjgzZTdhOWVmID0gJCgnPGRpdiBpZD0iaHRtbF9hNTRjNDIyYmFkMWE0ZjNlODI3NjczYWY4M2U3YTllZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RG93bnN2aWV3IENlbnRyYWwsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q5ZWJmNjEyNDY2ZjRiNTJiZmRkODA3NjE4YjczNDljLnNldENvbnRlbnQoaHRtbF9hNTRjNDIyYmFkMWE0ZjNlODI3NjczYWY4M2U3YTllZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYmZlNDc3YmY5Y2Q0YTEyYjYyZjE1NTc0OTAzYjkwOS5iaW5kUG9wdXAocG9wdXBfZDllYmY2MTI0NjZmNGI1MmJmZGQ4MDc2MThiNzM0OWMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZmZiZTE5OThkMWE3NDhkZjljNTY4YWYxOTgzMjNjM2QgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTk1MjU1LC03OS4zNDA5MjNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zOTJmNWQ3NTA2ODU0ZDRmYmJkMTMxMWZiZTc0NWU1NiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yOTlmYWZmNTBjODA0NTg1YWNmYzFhMzQyNWUzM2UwMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjk5ZmFmZjUwYzgwNDU4NWFjZmMxYTM0MjVlMzNlMDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0dWRpbyBEaXN0cmljdCwgRWFzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zOTJmNWQ3NTA2ODU0ZDRmYmJkMTMxMWZiZTc0NWU1Ni5zZXRDb250ZW50KGh0bWxfMjk5ZmFmZjUwYzgwNDU4NWFjZmMxYTM0MjVlMzNlMDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZmZiZTE5OThkMWE3NDhkZjljNTY4YWYxOTgzMjNjM2QuYmluZFBvcHVwKHBvcHVwXzM5MmY1ZDc1MDY4NTRkNGZiYmQxMzExZmJlNzQ1ZTU2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZjNDc2YmRhZTU4NDQ0ODY5NGNjOTFlMTY0ZmIyMmFiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzMzMjgyNSwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE5ZDk1ZDM4ODY2YTRkZTU5YTVhMjEyODQ0ZjU3MjkyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M3YjgyZmMyNjkyYTRlZTc4NWI0ZjJkMTA1NzNmZTc4ID0gJCgnPGRpdiBpZD0iaHRtbF9jN2I4MmZjMjY5MmE0ZWU3ODViNGYyZDEwNTczZmU3OCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVkZm9yZCBQYXJrLCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xOWQ5NWQzODg2NmE0ZGU1OWE1YTIxMjg0NGY1NzI5Mi5zZXRDb250ZW50KGh0bWxfYzdiODJmYzI2OTJhNGVlNzg1YjRmMmQxMDU3M2ZlNzgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmM0NzZiZGFlNTg0NDQ4Njk0Y2M5MWUxNjRmYjIyYWIuYmluZFBvcHVwKHBvcHVwXzE5ZDk1ZDM4ODY2YTRkZTU5YTVhMjEyODQ0ZjU3MjkyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQzZmNhYzBjOWYxMzQwMmViNTZjMGEwMGIxYzcwODBjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzMzMjgyNSwtNzkuNDE5NzQ5N10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQxZTg2ZWY3NTQ2MjQ3MGFiOGM3ZTFmOTEyZGU0MmNlID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMxNTQ0ZjMzYmY0MzQwOWJiMTI0NmM0YWU4ZGUwZjI2ID0gJCgnPGRpdiBpZD0iaHRtbF8zMTU0NGYzM2JmNDM0MDliYjEyNDZjNGFlOGRlMGYyNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgTWFub3IgRWFzdCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDFlODZlZjc1NDYyNDcwYWI4YzdlMWY5MTJkZTQyY2Uuc2V0Q29udGVudChodG1sXzMxNTQ0ZjMzYmY0MzQwOWJiMTI0NmM0YWU4ZGUwZjI2KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzZmNhYzBjOWYxMzQwMmViNTZjMGEwMGIxYzcwODBjLmJpbmRQb3B1cChwb3B1cF80MWU4NmVmNzU0NjI0NzBhYjhjN2UxZjkxMmRlNDJjZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMmU1NTg0MzExYzI0MGZhYmU0Y2Q5MGI4NTYzMWI0YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MTExNTgsLTc5LjQ3NjAxMzI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOGRkMzg4NzI3MmYyNDI4NWJlMDhlMmIwOGJmOTI4NzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOTQzYmE0M2IwOTQ2NGIwOGJmMzA5MzU5YWIyYzU3ZDAgPSAkKCc8ZGl2IGlkPSJodG1sXzk0M2JhNDNiMDk0NjRiMDhiZjMwOTM1OWFiMmM1N2QwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EZWwgUmF5LCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84ZGQzODg3MjcyZjI0Mjg1YmUwOGUyYjA4YmY5Mjg3NC5zZXRDb250ZW50KGh0bWxfOTQzYmE0M2IwOTQ2NGIwOGJmMzA5MzU5YWIyYzU3ZDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTJlNTU4NDMxMWMyNDBmYWJlNGNkOTBiODU2MzFiNGMuYmluZFBvcHVwKHBvcHVwXzhkZDM4ODcyNzJmMjQyODViZTA4ZTJiMDhiZjkyODc0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNjZDgxYTQzN2U0ZTQ5N2Q4ZTk1N2UyMzdkZThmN2Q3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkxMTE1OCwtNzkuNDc2MDEzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hY2UzNjU5NjA2OTA0ZDlmYWJkNjI2NWY5NzUwZmUwZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lYTZjMDZkNzcwMzA0Y2NlYTAzOGNiNmNlOWU2ZmY1YSA9ICQoJzxkaXYgaWQ9Imh0bWxfZWE2YzA2ZDc3MDMwNGNjZWEwMzhjYjZjZTllNmZmNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktlZWxlc2RhbGUsIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2FjZTM2NTk2MDY5MDRkOWZhYmQ2MjY1Zjk3NTBmZTBmLnNldENvbnRlbnQoaHRtbF9lYTZjMDZkNzcwMzA0Y2NlYTAzOGNiNmNlOWU2ZmY1YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zY2Q4MWE0MzdlNGU0OTdkOGU5NTdlMjM3ZGU4ZjdkNy5iaW5kUG9wdXAocG9wdXBfYWNlMzY1OTYwNjkwNGQ5ZmFiZDYyNjVmOTc1MGZlMGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTVlOTEzY2UwY2ZjNGQzYzgwZDJlYzJhODVlYmIyYzIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTExMTU4LC03OS40NzYwMTMyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzMyN2E3NGIyZmI0MjQ4YTlhOTRmZWRmMGYwMDk4ODc1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y5ZmYyYzY5OGQ4ZTRhZDM5ODZlNzI0MDcyZDJiMzFlID0gJCgnPGRpdiBpZD0iaHRtbF9mOWZmMmM2OThkOGU0YWQzOTg2ZTcyNDA3MmQyYjMxZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TW91bnQgRGVubmlzLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMjdhNzRiMmZiNDI0OGE5YTk0ZmVkZjBmMDA5ODg3NS5zZXRDb250ZW50KGh0bWxfZjlmZjJjNjk4ZDhlNGFkMzk4NmU3MjQwNzJkMmIzMWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTVlOTEzY2UwY2ZjNGQzYzgwZDJlYzJhODVlYmIyYzIuYmluZFBvcHVwKHBvcHVwXzMyN2E3NGIyZmI0MjQ4YTlhOTRmZWRmMGYwMDk4ODc1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzYzOTNjODBiNmU5ZTQ3YmE4ZmVmMGEwYjVhNjk5NTkwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkxMTE1OCwtNzkuNDc2MDEzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NGNmOWY2MmRmY2M0MWZkYTQ4NjI4MWYzZWFmMGYwYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNjRjNzNmZDJjMGY0YjJmOGRlYTZhY2JkMGEyZjRhZiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjY0YzczZmQyYzBmNGIyZjhkZWE2YWNiZDBhMmY0YWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlNpbHZlcnRob3JuLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80NGNmOWY2MmRmY2M0MWZkYTQ4NjI4MWYzZWFmMGYwYS5zZXRDb250ZW50KGh0bWxfYjY0YzczZmQyYzBmNGIyZjhkZWE2YWNiZDBhMmY0YWYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjM5M2M4MGI2ZTllNDdiYThmZWYwYTBiNWE2OTk1OTAuYmluZFBvcHVwKHBvcHVwXzQ0Y2Y5ZjYyZGZjYzQxZmRhNDg2MjgxZjNlYWYwZjBhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2E5OGEwNjdhOTUwYjRjZmJiZjgwYzgwYzA5NDg2NmExID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI0NzY1OSwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iN2QxYTk3MThkMGM0ZDE4YjM4N2U5OWFiNTZiNjlkNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MzJkNjEwOWM4MjM0ZGI2YjFjOWQ3ZjJkMWExOWRjOSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTMyZDYxMDljODIzNGRiNmIxYzlkN2YyZDFhMTlkYzkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkVtZXJ5LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iN2QxYTk3MThkMGM0ZDE4YjM4N2U5OWFiNTZiNjlkNi5zZXRDb250ZW50KGh0bWxfNTMyZDYxMDljODIzNGRiNmIxYzlkN2YyZDFhMTlkYzkpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTk4YTA2N2E5NTBiNGNmYmJmODBjODBjMDk0ODY2YTEuYmluZFBvcHVwKHBvcHVwX2I3ZDFhOTcxOGQwYzRkMThiMzg3ZTk5YWI1NmI2OWQ2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ZlNmUzZGE0NDk4OTQ3NWJhMGE3NmEzMzE2NjRkYWY4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI0NzY1OSwtNzkuNTMyMjQyNDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNzljMTUxNzQ2YWU0ZTg3OTMxZTVmOWViMjI2MjU3YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MDQ4NjE5NmFjY2Y0Yjg2YWU2YjE1ZTY0OTNkODc0NyA9ICQoJzxkaXYgaWQ9Imh0bWxfOTA0ODYxOTZhY2NmNGI4NmFlNmIxNWU2NDkzZDg3NDciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlcmxlYSwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjc5YzE1MTc0NmFlNGU4NzkzMWU1ZjllYjIyNjI1N2Muc2V0Q29udGVudChodG1sXzkwNDg2MTk2YWNjZjRiODZhZTZiMTVlNjQ5M2Q4NzQ3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ZlNmUzZGE0NDk4OTQ3NWJhMGE3NmEzMzE2NjRkYWY4LmJpbmRQb3B1cChwb3B1cF9iNzljMTUxNzQ2YWU0ZTg3OTMxZTVmOWViMjI2MjU3Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85OWI1NTRiYjI1ZDU0NmNlYTIyODE3MTNhYWQ0NmI4OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY5MjY1NzAwMDAwMDAwNCwtNzkuMjY0ODQ4MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2Q4MWM3MzhkYTRiYTQ5ZjlhODk4NjE1Y2E0MmI0YjllID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2VmODE3MzI4ODk2ODRmN2I5MmQ2MDY1NTUyYmMzOTBjID0gJCgnPGRpdiBpZD0iaHRtbF9lZjgxNzMyODg5Njg0ZjdiOTJkNjA2NTU1MmJjMzkwYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmlyY2ggQ2xpZmYsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kODFjNzM4ZGE0YmE0OWY5YTg5ODYxNWNhNDJiNGI5ZS5zZXRDb250ZW50KGh0bWxfZWY4MTczMjg4OTY4NGY3YjkyZDYwNjU1NTJiYzM5MGMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTliNTU0YmIyNWQ1NDZjZWEyMjgxNzEzYWFkNDZiODguYmluZFBvcHVwKHBvcHVwX2Q4MWM3MzhkYTRiYTQ5ZjlhODk4NjE1Y2E0MmI0YjllKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRkYWJjYjY2Y2JlNzRiZTliMWEwMDMxNjViZGVlOGQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjkyNjU3MDAwMDAwMDA0LC03OS4yNjQ4NDgxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjU5NmYxMzVjZDhhNDBhNmIyZmQzNDFiNzNlMmU4MzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjAzMTEzOTkwZjljNDA2MDk1NTg0MzZhNDMwZDJiMzcgPSAkKCc8ZGl2IGlkPSJodG1sXzIwMzExMzk5MGY5YzQwNjA5NTU4NDM2YTQzMGQyYjM3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DbGlmZnNpZGUgV2VzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Y1OTZmMTM1Y2Q4YTQwYTZiMmZkMzQxYjczZTJlODM0LnNldENvbnRlbnQoaHRtbF8yMDMxMTM5OTBmOWM0MDYwOTU1ODQzNmE0MzBkMmIzNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80ZGFiY2I2NmNiZTc0YmU5YjFhMDAzMTY1YmRlZThkMi5iaW5kUG9wdXAocG9wdXBfZjU5NmYxMzVjZDhhNDBhNmIyZmQzNDFiNzNlMmU4MzQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWYyOTdjOGU2OTNhNGQ4ZDg3Mjc5MmRkM2QwOWE1MGYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NzAxMTk5LC03OS40MDg0OTI3OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcyZDlhNDE3ZmQ4ZjRhZmM4ZjYyM2VmMjRhNjY5OWM2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzliM2UxM2VlNzc5MzQ0NzliYjU5MzkwMDQzYzZhNmZjID0gJCgnPGRpdiBpZD0iaHRtbF85YjNlMTNlZTc3OTM0NDc5YmI1OTM5MDA0M2M2YTZmYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2lsbG93ZGFsZSBTb3V0aCwgTm9ydGggWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzJkOWE0MTdmZDhmNGFmYzhmNjIzZWYyNGE2Njk5YzYuc2V0Q29udGVudChodG1sXzliM2UxM2VlNzc5MzQ0NzliYjU5MzkwMDQzYzZhNmZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzFmMjk3YzhlNjkzYTRkOGQ4NzI3OTJkZDNkMDlhNTBmLmJpbmRQb3B1cChwb3B1cF83MmQ5YTQxN2ZkOGY0YWZjOGY2MjNlZjI0YTY2OTljNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jZDIwZjZkNDQ4Mzk0YjVlYTY5YjZkZTZjM2U4N2IwMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc2MTYzMTMsLTc5LjUyMDk5OTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGM0MDRlZDMxODkxNDY1OThlMGQ2MWY0Y2M2OTI1ZTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTM1YjAyNTAwOTgzNGJjM2E4YzY2OTdhOTI3ODhiYzggPSAkKCc8ZGl2IGlkPSJodG1sX2EzNWIwMjUwMDk4MzRiYzNhOGM2Njk3YTkyNzg4YmM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Eb3duc3ZpZXcgTm9ydGh3ZXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80YzQwNGVkMzE4OTE0NjU5OGUwZDYxZjRjYzY5MjVlNS5zZXRDb250ZW50KGh0bWxfYTM1YjAyNTAwOTgzNGJjM2E4YzY2OTdhOTI3ODhiYzgpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2QyMGY2ZDQ0ODM5NGI1ZWE2OWI2ZGU2YzNlODdiMDAuYmluZFBvcHVwKHBvcHVwXzRjNDA0ZWQzMTg5MTQ2NTk4ZTBkNjFmNGNjNjkyNWU1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU1N2MxN2JhMGQxMTQ2ZmFiY2MxYTk1ZTllMmRiMTM2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzI4MDIwNSwtNzkuMzg4NzkwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzI1ZTJiMmQ2YTQyMzRmN2I5MTBlMWI0YzViYzRkZmU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzE1YjM0ZDFlMDM5OTQwYzU4MmZjYjY0YjY5NTAwMWY3ID0gJCgnPGRpdiBpZD0iaHRtbF8xNWIzNGQxZTAzOTk0MGM1ODJmY2I2NGI2OTUwMDFmNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TGF3cmVuY2UgUGFyaywgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yNWUyYjJkNmE0MjM0ZjdiOTEwZTFiNGM1YmM0ZGZlNC5zZXRDb250ZW50KGh0bWxfMTViMzRkMWUwMzk5NDBjNTgyZmNiNjRiNjk1MDAxZjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTU3YzE3YmEwZDExNDZmYWJjYzFhOTVlOWUyZGIxMzYuYmluZFBvcHVwKHBvcHVwXzI1ZTJiMmQ2YTQyMzRmN2I5MTBlMWI0YzViYzRkZmU0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhiMDEzNTNmZjU0YTRkNWZhMTkyMWIwODJmMmY0NDVjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzExNjk0OCwtNzkuNDE2OTM1NTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jZjEyYzlkM2JlZWM0YWE1OTQ5YTA4MDU0YTAyMjJhMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zY2E5MGQ0NjQ1Nzk0ZGM4YTQwZDE2NDVjOTdhYmMwMSA9ICQoJzxkaXYgaWQ9Imh0bWxfM2NhOTBkNDY0NTc5NGRjOGE0MGQxNjQ1Yzk3YWJjMDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VsYXduLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NmMTJjOWQzYmVlYzRhYTU5NDlhMDgwNTRhMDIyMmEyLnNldENvbnRlbnQoaHRtbF8zY2E5MGQ0NjQ1Nzk0ZGM4YTQwZDE2NDVjOTdhYmMwMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84YjAxMzUzZmY1NGE0ZDVmYTE5MjFiMDgyZjJmNDQ1Yy5iaW5kUG9wdXAocG9wdXBfY2YxMmM5ZDNiZWVjNGFhNTk0OWEwODA1NGEwMjIyYTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWQwMTlkMGJhZjY5NDc3OThlZDViYTk2MjU2ZGJiYjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzMxODUyOTk5OTk5OSwtNzkuNDg3MjYxOTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMjk2MjEwNjgwOTU0Mjk5YjYxODg1ZGRiMDM1YTA0OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMmEzYWQ1MmNjMzg0OTliODlmY2QwMjNlODU1Y2UxMSA9ICQoJzxkaXYgaWQ9Imh0bWxfMTJhM2FkNTJjYzM4NDk5Yjg5ZmNkMDIzZTg1NWNlMTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBKdW5jdGlvbiBOb3J0aCwgWW9yazwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjI5NjIxMDY4MDk1NDI5OWI2MTg4NWRkYjAzNWEwNDkuc2V0Q29udGVudChodG1sXzEyYTNhZDUyY2MzODQ5OWI4OWZjZDAyM2U4NTVjZTExKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VkMDE5ZDBiYWY2OTQ3Nzk4ZWQ1YmE5NjI1NmRiYmI4LmJpbmRQb3B1cChwb3B1cF9iMjk2MjEwNjgwOTU0Mjk5YjYxODg1ZGRiMDM1YTA0OSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83NjNjOTk0OGZmZmU0Y2Q3YWRjYjU2MGM0M2VmMzZhZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3MzE4NTI5OTk5OTk5LC03OS40ODcyNjE5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JmMTk1ZmVkMmE2NDQ1NzE4NzgyMGFmYTYyOGFkZjI2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA3NWIxNGQ2ZDkwZTQxMGVhZTg1Y2UzYmE1MmRjODVhID0gJCgnPGRpdiBpZD0iaHRtbF8wNzViMTRkNmQ5MGU0MTBlYWU4NWNlM2JhNTJkYzg1YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iZjE5NWZlZDJhNjQ0NTcxODc4MjBhZmE2MjhhZGYyNi5zZXRDb250ZW50KGh0bWxfMDc1YjE0ZDZkOTBlNDEwZWFlODVjZTNiYTUyZGM4NWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzYzYzk5NDhmZmZlNGNkN2FkY2I1NjBjNDNlZjM2YWUuYmluZFBvcHVwKHBvcHVwX2JmMTk1ZmVkMmE2NDQ1NzE4NzgyMGFmYTYyOGFkZjI2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM1NmVkNjBhZGE2ZTRmNmJhM2YyMTg0ZmJhMWVkZjJiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2ODc2LC03OS41MTgxODg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzQ4YzBjZTE2YjI2MDRlOGM5YjU1YjllZWZmMTQwZmRkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk5NTE5ZGY5M2Q4MDRkYjM4ZTQwM2RlYWI5OGE5ZmZhID0gJCgnPGRpdiBpZD0iaHRtbF85OTUxOWRmOTNkODA0ZGIzOGU0MDNkZWFiOThhOWZmYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdG9uLCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80OGMwY2UxNmIyNjA0ZThjOWI1NWI5ZWVmZjE0MGZkZC5zZXRDb250ZW50KGh0bWxfOTk1MTlkZjkzZDgwNGRiMzhlNDAzZGVhYjk4YTlmZmEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzU2ZWQ2MGFkYTZlNGY2YmEzZjIxODRmYmExZWRmMmIuYmluZFBvcHVwKHBvcHVwXzQ4YzBjZTE2YjI2MDRlOGM5YjU1YjllZWZmMTQwZmRkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQzMDJkOGI1MTMwNzQ5Njg5ZjYyYWIzOWFlZGJhMjNkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzU3NDA5NiwtNzkuMjczMzA0MDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81OGIzODM0NmQ2Yzk0OWE2OWZmY2JmN2E2Mzg5YTkwMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mNGRmNzAwNDBjYzg0ODRhOWQ4NjkzNTBhMDAxN2Y5MyA9ICQoJzxkaXYgaWQ9Imh0bWxfZjRkZjcwMDQwY2M4NDg0YTlkODY5MzUwYTAwMTdmOTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRvcnNldCBQYXJrLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNThiMzgzNDZkNmM5NDlhNjlmZmNiZjdhNjM4OWE5MDEuc2V0Q29udGVudChodG1sX2Y0ZGY3MDA0MGNjODQ4NGE5ZDg2OTM1MGEwMDE3ZjkzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQzMDJkOGI1MTMwNzQ5Njg5ZjYyYWIzOWFlZGJhMjNkLmJpbmRQb3B1cChwb3B1cF81OGIzODM0NmQ2Yzk0OWE2OWZmY2JmN2E2Mzg5YTkwMSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85OWQ5MTFkOGQ3YTE0ZGY1ODYxY2JmOGIwMDY0M2RhZCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1NzQwOTYsLTc5LjI3MzMwNDAwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDg3NzViNTEwOTY0NDI5ZGFiYTNlMTJjODE5MWIwMDQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzk4OTM2YWRhZDNiNDk4Mzk5ZjdiMTQ3MDZmNDZiZjIgPSAkKCc8ZGl2IGlkPSJodG1sXzc5ODkzNmFkYWQzYjQ5ODM5OWY3YjE0NzA2ZjQ2YmYyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TY2FyYm9yb3VnaCBUb3duIENlbnRyZSwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ4Nzc1YjUxMDk2NDQyOWRhYmEzZTEyYzgxOTFiMDA0LnNldENvbnRlbnQoaHRtbF83OTg5MzZhZGFkM2I0OTgzOTlmN2IxNDcwNmY0NmJmMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85OWQ5MTFkOGQ3YTE0ZGY1ODYxY2JmOGIwMDY0M2RhZC5iaW5kUG9wdXAocG9wdXBfNDg3NzViNTEwOTY0NDI5ZGFiYTNlMTJjODE5MWIwMDQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzkwYmQzZTc2OTk5NDFjNTg3NGViNTZhZmJkY2IyMmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTc0MDk2LC03OS4yNzMzMDQwMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U1OTkwOTU5ODVmNjQzZWZhYzI4NTRkOGNhOTcwMGEzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzRhY2ZmYjA1MDM3ZDQwM2E4M2Q0OWYyYTRlMGY3OWE0ID0gJCgnPGRpdiBpZD0iaHRtbF80YWNmZmIwNTAzN2Q0MDNhODNkNDlmMmE0ZTBmNzlhNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2V4Zm9yZCBIZWlnaHRzLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTU5OTA5NTk4NWY2NDNlZmFjMjg1NGQ4Y2E5NzAwYTMuc2V0Q29udGVudChodG1sXzRhY2ZmYjA1MDM3ZDQwM2E4M2Q0OWYyYTRlMGY3OWE0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M5MGJkM2U3Njk5OTQxYzU4NzRlYjU2YWZiZGNiMjJjLmJpbmRQb3B1cChwb3B1cF9lNTk5MDk1OTg1ZjY0M2VmYWMyODU0ZDhjYTk3MDBhMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83M2U3NjhhZmMxZjE0MjEyYjVjNTNhZDk1MTJlNDNiNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1Mjc1ODI5OTk5OTk5NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUzMDU2Y2U2NWVmOTQ5OGQ5NmYwMmY0MWQ1Zjc2Yzg3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2YxNzY3NDk1YmU0NzQ4YTViNTIzOGJlMWVjMjIwNDEwID0gJCgnPGRpdiBpZD0iaHRtbF9mMTc2NzQ5NWJlNDc0OGE1YjUyMzhiZTFlYzIyMDQxMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+WW9yayBNaWxscyBXZXN0LCBOb3J0aCBZb3JrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MzA1NmNlNjVlZjk0OThkOTZmMDJmNDFkNWY3NmM4Ny5zZXRDb250ZW50KGh0bWxfZjE3Njc0OTViZTQ3NDhhNWI1MjM4YmUxZWMyMjA0MTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzNlNzY4YWZjMWYxNDIxMmI1YzUzYWQ5NTEyZTQzYjUuYmluZFBvcHVwKHBvcHVwXzUzMDU2Y2U2NWVmOTQ5OGQ5NmYwMmY0MWQ1Zjc2Yzg3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJmNmY5MjJhZTMyMjQ4N2M5OTAyNGFiY2U4ZjZjOGE4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzEyNzUxMSwtNzkuMzkwMTk3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YwNTk1ZmYzNDBiYjQ3Yjk4OWQ1NGY0OTJiODRjMjkxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y0M2NmNzU3ZGE1MzQ1YWRhNDg1N2FjOWYxZTEzYjdhID0gJCgnPGRpdiBpZD0iaHRtbF9mNDNjZjc1N2RhNTM0NWFkYTQ4NTdhYzlmMWUxM2I3YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGF2aXN2aWxsZSBOb3J0aCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMDU5NWZmMzQwYmI0N2I5ODlkNTRmNDkyYjg0YzI5MS5zZXRDb250ZW50KGh0bWxfZjQzY2Y3NTdkYTUzNDVhZGE0ODU3YWM5ZjFlMTNiN2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmY2ZjkyMmFlMzIyNDg3Yzk5MDI0YWJjZThmNmM4YTguYmluZFBvcHVwKHBvcHVwX2YwNTk1ZmYzNDBiYjQ3Yjk4OWQ1NGY0OTJiODRjMjkxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjZmQ0OWY2OTMxZDRkMWJhNzZkZWQ0ZmU2NmUxYjg1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2OTQ3NiwtNzkuNDExMzA3MjAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MDM5MTdhYmU3ZDk0YTI3YTkwZDQ5NjRhMDc1ODdhNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iZDZmM2Q0ZDc0MDE0YmQ1OThiNThhYjdlYWNhMTZmMiA9ICQoJzxkaXYgaWQ9Imh0bWxfYmQ2ZjNkNGQ3NDAxNGJkNTk4YjU4YWI3ZWFjYTE2ZjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZvcmVzdCBIaWxsIE5vcnRoLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQwMzkxN2FiZTdkOTRhMjdhOTBkNDk2NGEwNzU4N2E2LnNldENvbnRlbnQoaHRtbF9iZDZmM2Q0ZDc0MDE0YmQ1OThiNThhYjdlYWNhMTZmMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80Y2ZkNDlmNjkzMWQ0ZDFiYTc2ZGVkNGZlNjZlMWI4NS5iaW5kUG9wdXAocG9wdXBfNDAzOTE3YWJlN2Q5NGEyN2E5MGQ0OTY0YTA3NTg3YTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTg3NDU4YTNjZTdkNDhiNzgyOGZjN2YzNWJkOWY4ZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42OTY5NDc2LC03OS40MTEzMDcyMDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2YwYTg4OWFjY2EyZDQ1N2I4NWZhMjE1Y2I1ZWYzZmE2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FjYTRkNTg4MTZhMDRlZWZiZWVmNmExZWY0YzcwYWFhID0gJCgnPGRpdiBpZD0iaHRtbF9hY2E0ZDU4ODE2YTA0ZWVmYmVlZjZhMWVmNGM3MGFhYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rm9yZXN0IEhpbGwgV2VzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mMGE4ODlhY2NhMmQ0NTdiODVmYTIxNWNiNWVmM2ZhNi5zZXRDb250ZW50KGh0bWxfYWNhNGQ1ODgxNmEwNGVlZmJlZWY2YTFlZjRjNzBhYWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTg3NDU4YTNjZTdkNDhiNzgyOGZjN2YzNWJkOWY4ZDIuYmluZFBvcHVwKHBvcHVwX2YwYTg4OWFjY2EyZDQ1N2I4NWZhMjE1Y2I1ZWYzZmE2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzRjMWEyNDIwNzI0ZTRkYTI5MjA1NDcxOTgxYjkyOWUyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yZjU4NjgzODYwZjk0M2JlOWFjY2IzMmM3NzZkNDYwZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iNjczOTNmNjY2OWQ0ZGJkYTdhZjVjM2Q1YzgwYjEyNiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjY3MzkzZjY2NjlkNGRiZGE3YWY1YzNkNWM4MGIxMjYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhpZ2ggUGFyaywgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZjU4NjgzODYwZjk0M2JlOWFjY2IzMmM3NzZkNDYwZS5zZXRDb250ZW50KGh0bWxfYjY3MzkzZjY2NjlkNGRiZGE3YWY1YzNkNWM4MGIxMjYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNGMxYTI0MjA3MjRlNGRhMjkyMDU0NzE5ODFiOTI5ZTIuYmluZFBvcHVwKHBvcHVwXzJmNTg2ODM4NjBmOTQzYmU5YWNjYjMyYzc3NmQ0NjBlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NiNzk2Y2M0MzI1ZTRkOTY5Njk4MGFkOGZmYTZmN2I4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYxNjA4MywtNzkuNDY0NzYzMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iOTI4ZDAwMmJmZWQ0M2U2YjQwM2RjMjE2YzVjZWZlZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZGZiYWEwYjExMDY0ZmI4ODdlZDU5YWQ1ZWVhNmJhZCA9ICQoJzxkaXYgaWQ9Imh0bWxfM2RmYmFhMGIxMTA2NGZiODg3ZWQ1OWFkNWVlYTZiYWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBKdW5jdGlvbiBTb3V0aCwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iOTI4ZDAwMmJmZWQ0M2U2YjQwM2RjMjE2YzVjZWZlZS5zZXRDb250ZW50KGh0bWxfM2RmYmFhMGIxMTA2NGZiODg3ZWQ1OWFkNWVlYTZiYWQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2I3OTZjYzQzMjVlNGQ5Njk2OTgwYWQ4ZmZhNmY3YjguYmluZFBvcHVwKHBvcHVwX2I5MjhkMDAyYmZlZDQzZTZiNDAzZGMyMTZjNWNlZmVlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU0YzU1Y2NiMjQ0YzRiN2RiZjc3ODhjODA2OGI1NGM5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjk2MzE5LC03OS41MzIyNDI0MDAwMDAwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzE4ZWEwZWU4NThiOTQ4YWJiNWQ3NjUwYjBlMGFkMjBmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2QwOTk3MzI0MTIxOTRlMzM4ZDFjN2I0NDFhMzlhNjVhID0gJCgnPGRpdiBpZD0iaHRtbF9kMDk5NzMyNDEyMTk0ZTMzOGQxYzdiNDQxYTM5YTY1YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2VzdG1vdW50LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzE4ZWEwZWU4NThiOTQ4YWJiNWQ3NjUwYjBlMGFkMjBmLnNldENvbnRlbnQoaHRtbF9kMDk5NzMyNDEyMTk0ZTMzOGQxYzdiNDQxYTM5YTY1YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NGM1NWNjYjI0NGM0YjdkYmY3Nzg4YzgwNjhiNTRjOS5iaW5kUG9wdXAocG9wdXBfMThlYTBlZTg1OGI5NDhhYmI1ZDc2NTBiMGUwYWQyMGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDI4MGFmYjk0MzJkNGI3Mzk0Y2EwNGZmNjJkNjI5NjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43NTAwNzE1MDAwMDAwMDQsLTc5LjI5NTg0OTFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83N2EyN2I4NTFjMzE0YjU1OTM3NjEyZmZhMmE3Y2VlNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMTYzNzFhMzY1ODQ0YTFlYTM0NmJjNmQ1YmM2YTQ1ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfYjE2MzcxYTM2NTg0NGExZWEzNDZiYzZkNWJjNmE0NWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hcnl2YWxlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzdhMjdiODUxYzMxNGI1NTkzNzYxMmZmYTJhN2NlZTYuc2V0Q29udGVudChodG1sX2IxNjM3MWEzNjU4NDRhMWVhMzQ2YmM2ZDViYzZhNDVkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzAyODBhZmI5NDMyZDRiNzM5NGNhMDRmZjYyZDYyOTYzLmJpbmRQb3B1cChwb3B1cF83N2EyN2I4NTFjMzE0YjU1OTM3NjEyZmZhMmE3Y2VlNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MWQwNjk4ZWQxMWY0Yzk2ODBkMTM3YzA3NTY1ZGQzYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjc1MDA3MTUwMDAwMDAwNCwtNzkuMjk1ODQ5MV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzg3ZDVjMDA1NzJhMTRlYjI4NWU5NTQ4YzQ2Mzc4OGZhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzYyYWM1ZGEzODlkODRmZjlhZjRiNTZjMTVmMWU0ODI2ID0gJCgnPGRpdiBpZD0iaHRtbF82MmFjNWRhMzg5ZDg0ZmY5YWY0YjU2YzE1ZjFlNDgyNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+V2V4Zm9yZCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg3ZDVjMDA1NzJhMTRlYjI4NWU5NTQ4YzQ2Mzc4OGZhLnNldENvbnRlbnQoaHRtbF82MmFjNWRhMzg5ZDg0ZmY5YWY0YjU2YzE1ZjFlNDgyNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MWQwNjk4ZWQxMWY0Yzk2ODBkMTM3YzA3NTY1ZGQzYi5iaW5kUG9wdXAocG9wdXBfODdkNWMwMDU3MmExNGViMjg1ZTk1NDhjNDYzNzg4ZmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2Q4ZDZlNzI1ZGI5NGViM2FiZDkxNjA2NDBhZjZiZWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODI3MzY0LC03OS40NDIyNTkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjk4YWQ0MjllZDQ4NDE0ZTljYWQyZTY4YzkwZmZhMzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGU3YmNjMjZlNzU4NDgxYmJkNjM3YTkyN2VkMzI4YTcgPSAkKCc8ZGl2IGlkPSJodG1sX2RlN2JjYzI2ZTc1ODQ4MWJiZDYzN2E5MjdlZDMyOGE3IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5XaWxsb3dkYWxlIFdlc3QsIE5vcnRoIFlvcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI5OGFkNDI5ZWQ0ODQxNGU5Y2FkMmU2OGM5MGZmYTM4LnNldENvbnRlbnQoaHRtbF9kZTdiY2MyNmU3NTg0ODFiYmQ2MzdhOTI3ZWQzMjhhNyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jZDhkNmU3MjVkYjk0ZWIzYWJkOTE2MDY0MGFmNmJlZS5iaW5kUG9wdXAocG9wdXBfMjk4YWQ0MjllZDQ4NDE0ZTljYWQyZTY4YzkwZmZhMzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjJjYzg5M2MzOTZiNGQzZWIyZmJjMTI1MGI1NDczNDQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MTUzODM0LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk0OGI5NGVmYmUxNjQxOTA4YWYzOWYxYTU4YWVhNjc1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NhZjAzODhlOTI1NzQ4NDU5N2JhN2NjYTA0MTMxOTViID0gJCgnPGRpdiBpZD0iaHRtbF9jYWYwMzg4ZTkyNTc0ODQ1OTdiYTdjY2EwNDEzMTk1YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggVG9yb250byBXZXN0LCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk0OGI5NGVmYmUxNjQxOTA4YWYzOWYxYTU4YWVhNjc1LnNldENvbnRlbnQoaHRtbF9jYWYwMzg4ZTkyNTc0ODQ1OTdiYTdjY2EwNDEzMTk1Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMmNjODkzYzM5NmI0ZDNlYjJmYmMxMjUwYjU0NzM0NC5iaW5kUG9wdXAocG9wdXBfOTQ4Yjk0ZWZiZTE2NDE5MDhhZjM5ZjFhNThhZWE2NzUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjc0MGJkNTcwMGMzNDA3OGJmYzhkMTU3NWNiNWU5MTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRmNDIxZGZjODBiYTQ4Mjc4NDhmMDVlYTYwMjRiMTU0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzVlOTgxMGJkMDBjMTQ0OThhOWVlMDQ0YjZkMGQxYzQyID0gJCgnPGRpdiBpZD0iaHRtbF81ZTk4MTBiZDAwYzE0NDk4YTllZTA0NGI2ZDBkMWM0MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIEFubmV4LCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRmNDIxZGZjODBiYTQ4Mjc4NDhmMDVlYTYwMjRiMTU0LnNldENvbnRlbnQoaHRtbF81ZTk4MTBiZDAwYzE0NDk4YTllZTA0NGI2ZDBkMWM0Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NzQwYmQ1NzAwYzM0MDc4YmZjOGQxNTc1Y2I1ZTkxNS5iaW5kUG9wdXAocG9wdXBfNGY0MjFkZmM4MGJhNDgyNzg0OGYwNWVhNjAyNGIxNTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNWEyM2IwYWFhN2Q3NGE1MDg4ZTExMTQ4Mzk4YjQzYmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NzI3MDk3LC03OS40MDU2Nzg0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I0NjI2YjI1ZDNiMjQ3OTQ4Mzk1Y2UwMWRmNDViNmQ4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZlZTgxZGUzYWI4YjQ2ZGM5NmE1N2Q2NjE5ZjlhYTgyID0gJCgnPGRpdiBpZD0iaHRtbF82ZWU4MWRlM2FiOGI0NmRjOTZhNTdkNjYxOWY5YWE4MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Tm9ydGggTWlkdG93biwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNDYyNmIyNWQzYjI0Nzk0ODM5NWNlMDFkZjQ1YjZkOC5zZXRDb250ZW50KGh0bWxfNmVlODFkZTNhYjhiNDZkYzk2YTU3ZDY2MTlmOWFhODIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNWEyM2IwYWFhN2Q3NGE1MDg4ZTExMTQ4Mzk4YjQzYmUuYmluZFBvcHVwKHBvcHVwX2I0NjI2YjI1ZDNiMjQ3OTQ4Mzk1Y2UwMWRmNDViNmQ4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNhMjA3NGRhNTczMjQ1ODE4OTkxOTZiMTg0YWRjOTk0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjcyNzA5NywtNzkuNDA1Njc4NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YmI3YWY3MzNiNzA0MGVlOTJmYzY5MjYyNTc2NGEwNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zN2MyZmIxN2E2NDQ0ZjczYmViY2VjNmE5ZWMyMzZmZSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzdjMmZiMTdhNjQ0NGY3M2JlYmNlYzZhOWVjMjM2ZmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPllvcmt2aWxsZSwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YmI3YWY3MzNiNzA0MGVlOTJmYzY5MjYyNTc2NGEwNC5zZXRDb250ZW50KGh0bWxfMzdjMmZiMTdhNjQ0NGY3M2JlYmNlYzZhOWVjMjM2ZmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2EyMDc0ZGE1NzMyNDU4MTg5OTE5NmIxODRhZGM5OTQuYmluZFBvcHVwKHBvcHVwXzZiYjdhZjczM2I3MDQwZWU5MmZjNjkyNjI1NzY0YTA0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk4M2Q5NGQ2MDdhNDQyNGM4NDYyNmRmNmYyOGNhOWU0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmEwYTcxYzUwYjc1NDExYmFmZTFhODE5MWE2MDBiMGUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzRlYTAxODEzMWRjNDc3YmJlMjYyN2VhMGZkM2FhMzYgPSAkKCc8ZGl2IGlkPSJodG1sXzM0ZWEwMTgxMzFkYzQ3N2JiZTI2MjdlYTBmZDNhYTM2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5QYXJrZGFsZSwgV2VzdCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iYTBhNzFjNTBiNzU0MTFiYWZlMWE4MTkxYTYwMGIwZS5zZXRDb250ZW50KGh0bWxfMzRlYTAxODEzMWRjNDc3YmJlMjYyN2VhMGZkM2FhMzYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOTgzZDk0ZDYwN2E0NDI0Yzg0NjI2ZGY2ZjI4Y2E5ZTQuYmluZFBvcHVwKHBvcHVwX2JhMGE3MWM1MGI3NTQxMWJhZmUxYTgxOTFhNjAwYjBlKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzVmZTM1YjJjMzlkNTRkYzFhMzIxZjkyYWZjZDM4NTk2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4OTU5NywtNzkuNDU2MzI1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2JkMTFhNTQxOTRlNDhmZGFlZTMyNjgzZmZmMzU5YjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTYwZTUwN2FhNTVkNDIyMmFkYTliMTFmZDM1OGViYjMgPSAkKCc8ZGl2IGlkPSJodG1sX2U2MGU1MDdhYTU1ZDQyMjJhZGE5YjExZmQzNThlYmIzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Sb25jZXN2YWxsZXMsIFdlc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2JkMTFhNTQxOTRlNDhmZGFlZTMyNjgzZmZmMzU5YjQuc2V0Q29udGVudChodG1sX2U2MGU1MDdhYTU1ZDQyMjJhZGE5YjExZmQzNThlYmIzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzVmZTM1YjJjMzlkNTRkYzFhMzIxZjkyYWZjZDM4NTk2LmJpbmRQb3B1cChwb3B1cF9jYmQxMWE1NDE5NGU0OGZkYWVlMzI2ODNmZmYzNTliNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lMDdhOWJmYjU1YTI0OTk4ODk1MWZhZmI3MDY4MjRmYyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjk2NTYsLTc5LjYxNTgxODk5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjk4OTM2ZDlhMDJhNGFiYWE5OWIwNDMyZDA1MzI2MTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWY3MjNiMDlmZGNiNGJhNzhmMjZjZDgzZWQxNWFhZjAgPSAkKCc8ZGl2IGlkPSJodG1sXzVmNzIzYjA5ZmRjYjRiYTc4ZjI2Y2Q4M2VkMTVhYWYwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYW5hZGEgUG9zdCBHYXRld2F5IFByb2Nlc3NpbmcgQ2VudHJlLCBNaXNzaXNzYXVnYTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjk4OTM2ZDlhMDJhNGFiYWE5OWIwNDMyZDA1MzI2MTUuc2V0Q29udGVudChodG1sXzVmNzIzYjA5ZmRjYjRiYTc4ZjI2Y2Q4M2VkMTVhYWYwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UwN2E5YmZiNTVhMjQ5OTg4OTUxZmFmYjcwNjgyNGZjLmJpbmRQb3B1cChwb3B1cF9iOTg5MzZkOWEwMmE0YWJhYTk5YjA0MzJkMDUzMjYxNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84NTk4YzM3N2QzNzU0Yjg5OGVlMzg1NmU3NWU4MzE4MSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4ODkwNTQsLTc5LjU1NDcyNDQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDE0ZDRmOWQ0ODJhNGIzYjhiZmRkZDFmODdmYThmNmYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMDZlNTNiOTc3N2ZhNGYwMWEyYTJkYWIzZTY4ZmUxMGUgPSAkKCc8ZGl2IGlkPSJodG1sXzA2ZTUzYjk3NzdmYTRmMDFhMmEyZGFiM2U2OGZlMTBlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3ZpZXcgVmlsbGFnZSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMTRkNGY5ZDQ4MmE0YjNiOGJmZGRkMWY4N2ZhOGY2Zi5zZXRDb250ZW50KGh0bWxfMDZlNTNiOTc3N2ZhNGYwMWEyYTJkYWIzZTY4ZmUxMGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODU5OGMzNzdkMzc1NGI4OThlZTM4NTZlNzVlODMxODEuYmluZFBvcHVwKHBvcHVwX2QxNGQ0ZjlkNDgyYTRiM2I4YmZkZGQxZjg3ZmE4ZjZmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U2MTBjNzMzMGJjZjQyZmE4OGVjYjcwMTE4NzI3YWIwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg4OTA1NCwtNzkuNTU0NzI0NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMGFlNDFhMWQxYjU0OWRmOThlYjAzMjI2YzEyZDcwMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mYWI4YTAxZTgwMWE0MjczODExNmJmOTY2MTU2MDk3ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfZmFiOGEwMWU4MDFhNDI3MzgxMTZiZjk2NjE1NjA5N2YiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1hcnRpbiBHcm92ZSBHYXJkZW5zLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzEwYWU0MWExZDFiNTQ5ZGY5OGViMDMyMjZjMTJkNzAzLnNldENvbnRlbnQoaHRtbF9mYWI4YTAxZTgwMWE0MjczODExNmJmOTY2MTU2MDk3Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNjEwYzczMzBiY2Y0MmZhODhlY2I3MDExODcyN2FiMC5iaW5kUG9wdXAocG9wdXBfMTBhZTQxYTFkMWI1NDlkZjk4ZWIwMzIyNmMxMmQ3MDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDRkYTQ0NTFkMjU2NGExYThiMjA2MWRiMDk5NmZmMTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODg5MDU0LC03OS41NTQ3MjQ0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzA5NDY1YjAyYzYyNTRhY2ZiYWFhZTk3MzUyNTc5YTJmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQxZGRhNzRkY2VlNzQ1NDdiMDc5MGU0OWJjNWY2ODAxID0gJCgnPGRpdiBpZD0iaHRtbF80MWRkYTc0ZGNlZTc0NTQ3YjA3OTBlNDliYzVmNjgwMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmljaHZpZXcgR2FyZGVucywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wOTQ2NWIwMmM2MjU0YWNmYmFhYWU5NzM1MjU3OWEyZi5zZXRDb250ZW50KGh0bWxfNDFkZGE3NGRjZWU3NDU0N2IwNzkwZTQ5YmM1ZjY4MDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDRkYTQ0NTFkMjU2NGExYThiMjA2MWRiMDk5NmZmMTkuYmluZFBvcHVwKHBvcHVwXzA5NDY1YjAyYzYyNTRhY2ZiYWFhZTk3MzUyNTc5YTJmKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FiMTFjZWIyNGVhYjQ2ZmRhNDBjZTcyZGNhYjM4YmU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg4OTA1NCwtNzkuNTU0NzI0NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8wY2IxNTljZDU5MDA0NmRmODAyNjZlYjc3MGQ5NDNlYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yM2NmNzk3YWVhYWI0ZGZiODhmNmFkMmFhNDc3M2Q1YSA9ICQoJzxkaXYgaWQ9Imh0bWxfMjNjZjc5N2FlYWFiNGRmYjg4ZjZhZDJhYTQ3NzNkNWEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBQaGlsbGlwcywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wY2IxNTljZDU5MDA0NmRmODAyNjZlYjc3MGQ5NDNlYS5zZXRDb250ZW50KGh0bWxfMjNjZjc5N2FlYWFiNGRmYjg4ZjZhZDJhYTQ3NzNkNWEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWIxMWNlYjI0ZWFiNDZmZGE0MGNlNzJkY2FiMzhiZTkuYmluZFBvcHVwKHBvcHVwXzBjYjE1OWNkNTkwMDQ2ZGY4MDI2NmViNzcwZDk0M2VhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM5ZjRhMTJlZDI4ZDQ4YThhNTFjMmJhZGU3MTNjZjQxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk0MjAwMywtNzkuMjYyMDI5NDAwMDAwMDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yODlmMzkzNzk3ZjI0NDc2ODFlYmQ3ZGI1ZTM2YjFiYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85MjFiNGEwZmQ2MGU0OWZjYWViODE0ZGU3ODJlYzgyNSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTIxYjRhMGZkNjBlNDlmY2FlYjgxNGRlNzgyZWM4MjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFnaW5jb3VydCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI4OWYzOTM3OTdmMjQ0NzY4MWViZDdkYjVlMzZiMWJjLnNldENvbnRlbnQoaHRtbF85MjFiNGEwZmQ2MGU0OWZjYWViODE0ZGU3ODJlYzgyNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zOWY0YTEyZWQyOGQ0OGE4YTUxYzJiYWRlNzEzY2Y0MS5iaW5kUG9wdXAocG9wdXBfMjg5ZjM5Mzc5N2YyNDQ3NjgxZWJkN2RiNWUzNmIxYmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMzBiZjVkMTBiNTE4NDUyYzkwMzgyYzU2ODhiZGQyNjMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43MDQzMjQ0LC03OS4zODg3OTAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2NhZDRkMTE2YTg3NDJjNzhhOTQxMjQ0NTM5MDZiMTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzUwNjY2ZDQ2MWMyNDk0MzhjODkxM2JmNTI2MDcxMmEgPSAkKCc8ZGl2IGlkPSJodG1sXzM1MDY2NmQ0NjFjMjQ5NDM4Yzg5MTNiZjUyNjA3MTJhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5EYXZpc3ZpbGxlLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NjYWQ0ZDExNmE4NzQyYzc4YTk0MTI0NDUzOTA2YjE3LnNldENvbnRlbnQoaHRtbF8zNTA2NjZkNDYxYzI0OTQzOGM4OTEzYmY1MjYwNzEyYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMGJmNWQxMGI1MTg0NTJjOTAzODJjNTY4OGJkZDI2My5iaW5kUG9wdXAocG9wdXBfY2NhZDRkMTE2YTg3NDJjNzhhOTQxMjQ0NTM5MDZiMTcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOGZiOGQwMWQ3ZGIwNDhmNWFiYmIwMTFmNzFjZTU2MTggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI2OTU2LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjg4ZGQ2N2I4YzIwNGYyMmE3M2RkOTViNzc1OGRiNWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmNhMTkwMzBmMmNmNGFjNDg0MGM4MWU4NmZkMjYwZDYgPSAkKCc8ZGl2IGlkPSJodG1sXzZjYTE5MDMwZjJjZjRhYzQ4NDBjODFlODZmZDI2MGQ2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3JkLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iODhkZDY3YjhjMjA0ZjIyYTczZGQ5NWI3NzU4ZGI1Yi5zZXRDb250ZW50KGh0bWxfNmNhMTkwMzBmMmNmNGFjNDg0MGM4MWU4NmZkMjYwZDYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfOGZiOGQwMWQ3ZGIwNDhmNWFiYmIwMTFmNzFjZTU2MTguYmluZFBvcHVwKHBvcHVwX2I4OGRkNjdiOGMyMDRmMjJhNzNkZDk1Yjc3NThkYjViKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzFjMTMyYmYyMzZlYjQyOTY4MmIyMDAxODY4OTc4OTQ2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY4ZmI4M2RjYjAwODQwZmRhM2Y5ODBiYjJiZjY4NjA1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EyOGEyNDU3YmE4MjQ0YzRhOTljNDAxMDY5Nzk1YThlID0gJCgnPGRpdiBpZD0iaHRtbF9hMjhhMjQ1N2JhODI0NGM0YTk5YzQwMTA2OTc5NWE4ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5pdmVyc2l0eSBvZiBUb3JvbnRvLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82OGZiODNkY2IwMDg0MGZkYTNmOTgwYmIyYmY2ODYwNS5zZXRDb250ZW50KGh0bWxfYTI4YTI0NTdiYTgyNDRjNGE5OWM0MDEwNjk3OTVhOGUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWMxMzJiZjIzNmViNDI5NjgyYjIwMDE4Njg5Nzg5NDYuYmluZFBvcHVwKHBvcHVwXzY4ZmI4M2RjYjAwODQwZmRhM2Y5ODBiYjJiZjY4NjA1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzlkNjgxNzdmYjE5YzRiYjI4ZjZiYTUyYWMwNGQ4ZDU2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUxNTcwNiwtNzkuNDg0NDQ5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY1ODIyYTkyMmIzZjQ2NjM5Y2U5ZTQwOTIyMGM4YjY1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzllMDNmMGMwMGJhMDQ0OGI5ZjVkZjZhMjk0MDc1MzVlID0gJCgnPGRpdiBpZD0iaHRtbF85ZTAzZjBjMDBiYTA0NDhiOWY1ZGY2YTI5NDA3NTM1ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnVubnltZWRlLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY1ODIyYTkyMmIzZjQ2NjM5Y2U5ZTQwOTIyMGM4YjY1LnNldENvbnRlbnQoaHRtbF85ZTAzZjBjMDBiYTA0NDhiOWY1ZGY2YTI5NDA3NTM1ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZDY4MTc3ZmIxOWM0YmIyOGY2YmE1MmFjMDRkOGQ1Ni5iaW5kUG9wdXAocG9wdXBfNjU4MjJhOTIyYjNmNDY2MzljZTllNDA5MjIwYzhiNjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjhjYmQwMzU5MDkzNGNlNTg2OGRkZTRjM2JkNTMzMWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE1NzA2LC03OS40ODQ0NDk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNDgzNDllZmJkZmU5NDQ3ZWE4NDI4ZTM3NmM5MWY3NTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjFkMTI3ODEyNjA3NDBhZDg0ZDUxNGUwZmRmYjY3MDUgPSAkKCc8ZGl2IGlkPSJodG1sXzIxZDEyNzgxMjYwNzQwYWQ4NGQ1MTRlMGZkZmI2NzA1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Td2Fuc2VhLCBXZXN0IFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQ4MzQ5ZWZiZGZlOTQ0N2VhODQyOGUzNzZjOTFmNzU0LnNldENvbnRlbnQoaHRtbF8yMWQxMjc4MTI2MDc0MGFkODRkNTE0ZTBmZGZiNjcwNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82OGNiZDAzNTkwOTM0Y2U1ODY4ZGRlNGMzYmQ1MzMxYS5iaW5kUG9wdXAocG9wdXBfNDgzNDllZmJkZmU5NDQ3ZWE4NDI4ZTM3NmM5MWY3NTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTIxZDhiODcyZGEzNDRhYmExOGE3MTYwNWJmYmJmMDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODE2Mzc1LC03OS4zMDQzMDIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTEyNGNlZjAzOTM3NDYyODg5NTg0NDQyYWYzZTQzNzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjNkZTRkMDZlMDkxNGZkZjhmNjg1MmNmZWNjY2I4ZWEgPSAkKCc8ZGl2IGlkPSJodG1sX2YzZGU0ZDA2ZTA5MTRmZGY4ZjY4NTJjZmVjY2NiOGVhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DbGFya3MgQ29ybmVycywgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ExMjRjZWYwMzkzNzQ2Mjg4OTU4NDQ0MmFmM2U0MzcyLnNldENvbnRlbnQoaHRtbF9mM2RlNGQwNmUwOTE0ZmRmOGY2ODUyY2ZlY2NjYjhlYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85MjFkOGI4NzJkYTM0NGFiYTE4YTcxNjA1YmZiYmYwOS5iaW5kUG9wdXAocG9wdXBfYTEyNGNlZjAzOTM3NDYyODg5NTg0NDQyYWYzZTQzNzIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTlkYTEzZWJjOTVlNDdjMTk0YzA1YjgwZTc5MWQ5ZTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODE2Mzc1LC03OS4zMDQzMDIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfY2FmNTY2ODA0YTA2NGI2ZTk0NmUwMDFmODg2ZTdkNTQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjU4YzY2ZGUzMzViNGI1OWEwNzYzOWZkYjg1OTM0YzggPSAkKCc8ZGl2IGlkPSJodG1sX2Y1OGM2NmRlMzM1YjRiNTlhMDc2MzlmZGI4NTkzNGM4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdWxsaXZhbiwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2NhZjU2NjgwNGEwNjRiNmU5NDZlMDAxZjg4NmU3ZDU0LnNldENvbnRlbnQoaHRtbF9mNThjNjZkZTMzNWI0YjU5YTA3NjM5ZmRiODU5MzRjOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xOWRhMTNlYmM5NWU0N2MxOTRjMDViODBlNzkxZDllOS5iaW5kUG9wdXAocG9wdXBfY2FmNTY2ODA0YTA2NGI2ZTk0NmUwMDFmODg2ZTdkNTQpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTdiZjhiZTExNDQ0NGM0Njk0MjdkMmQ2OTNkNmZlMWQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43ODE2Mzc1LC03OS4zMDQzMDIxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTczNGZkODhhZDM1NDY1NDg2OGIyYzJhZTcyNTBmODIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZThmNzdhNzczOWYzNDNmNzk2OWMzNGZmMDJmMGZiNzIgPSAkKCc8ZGl2IGlkPSJodG1sX2U4Zjc3YTc3MzlmMzQzZjc5NjljMzRmZjAyZjBmYjcyIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UYW0gTyYjMzk7U2hhbnRlciwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk3MzRmZDg4YWQzNTQ2NTQ4NjhiMmMyYWU3MjUwZjgyLnNldENvbnRlbnQoaHRtbF9lOGY3N2E3NzM5ZjM0M2Y3OTY5YzM0ZmYwMmYwZmI3Mik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85N2JmOGJlMTE0NDQ0YzQ2OTQyN2QyZDY5M2Q2ZmUxZC5iaW5kUG9wdXAocG9wdXBfOTczNGZkODhhZDM1NDY1NDg2OGIyYzJhZTcyNTBmODIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMmFkYjI5OTE0ZDBiNGM4OTgyZjZmZTk1ZTJlMzBkN2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODk1NzQzLC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RiNGMyYTkzNzYxYjRkYWM4MmQ4YTNlMzZlZjhmZDI2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2NmZGEwZjY3YTk4ZjQ5MjFhYTE0ZWM4YmM5Y2ZjY2JmID0gJCgnPGRpdiBpZD0iaHRtbF9jZmRhMGY2N2E5OGY0OTIxYWExNGVjOGJjOWNmY2NiZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TW9vcmUgUGFyaywgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kYjRjMmE5Mzc2MWI0ZGFjODJkOGEzZTM2ZWY4ZmQyNi5zZXRDb250ZW50KGh0bWxfY2ZkYTBmNjdhOThmNDkyMWFhMTRlYzhiYzljZmNjYmYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmFkYjI5OTE0ZDBiNGM4OTgyZjZmZTk1ZTJlMzBkN2UuYmluZFBvcHVwKHBvcHVwX2RiNGMyYTkzNzYxYjRkYWM4MmQ4YTNlMzZlZjhmZDI2KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZlYzQ3MzVlZmQzYzQ4NTZhMTQ0NTFlMDdkMDVhYjNmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjg5NTc0MywtNzkuMzgzMTU5OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lYjdmZjFkOWFjMjU0NWRkODdhNzAxM2Q5N2M5ZDZmYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yNTc3OWYwNjQ5NDE0MTRkOGNiNzU4NmNjOTRjZjMwZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMjU3NzlmMDY0OTQxNDE0ZDhjYjc1ODZjYzk0Y2YzMGYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bW1lcmhpbGwgRWFzdCwgQ2VudHJhbCBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lYjdmZjFkOWFjMjU0NWRkODdhNzAxM2Q5N2M5ZDZmYy5zZXRDb250ZW50KGh0bWxfMjU3NzlmMDY0OTQxNDE0ZDhjYjc1ODZjYzk0Y2YzMGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNmVjNDczNWVmZDNjNDg1NmExNDQ1MWUwN2QwNWFiM2YuYmluZFBvcHVwKHBvcHVwX2ViN2ZmMWQ5YWMyNTQ1ZGQ4N2E3MDEzZDk3YzlkNmZjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY0NjUyM2MyMzMzYjRmMDlhYjNkZmE5NzRiY2Q5NjQyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2UwNTU0ZWU1MThjOTQyOTY5ODlkZTNmMzA1YjNmODIxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2Y1MzMzNTFiODg0MzRlNTM4ZGQ1OThhZDNhZmFkNzEzID0gJCgnPGRpdiBpZD0iaHRtbF9mNTMzMzUxYjg4NDM0ZTUzOGRkNTk4YWQzYWZhZDcxMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2hpbmF0b3duLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMDU1NGVlNTE4Yzk0Mjk2OTg5ZGUzZjMwNWIzZjgyMS5zZXRDb250ZW50KGh0bWxfZjUzMzM1MWI4ODQzNGU1MzhkZDU5OGFkM2FmYWQ3MTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjQ2NTIzYzIzMzNiNGYwOWFiM2RmYTk3NGJjZDk2NDIuYmluZFBvcHVwKHBvcHVwX2UwNTU0ZWU1MThjOTQyOTY5ODlkZTNmMzA1YjNmODIxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzk5NTkyOGVhZTRhZjQ0MmE4NGFiMmUyM2U3NTVhMGM4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYxODY3NTdhYTdkMDQwNTY4YTdkMjY4Y2NiNjNjNzhmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzcxODhlNDliMGU0MjQzNTZhM2RlNWI1NDg1NWQ2NmRjID0gJCgnPGRpdiBpZD0iaHRtbF83MTg4ZTQ5YjBlNDI0MzU2YTNkZTViNTQ4NTVkNjZkYyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R3JhbmdlIFBhcmssIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYxODY3NTdhYTdkMDQwNTY4YTdkMjY4Y2NiNjNjNzhmLnNldENvbnRlbnQoaHRtbF83MTg4ZTQ5YjBlNDI0MzU2YTNkZTViNTQ4NTVkNjZkYyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85OTU5MjhlYWU0YWY0NDJhODRhYjJlMjNlNzU1YTBjOC5iaW5kUG9wdXAocG9wdXBfNjE4Njc1N2FhN2QwNDA1NjhhN2QyNjhjY2I2M2M3OGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTcyNjE4NDZmMDZkNDc3ZGI1N2IzOTU3ZDQzMWNmNDUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjVmYmE3MDZkYmRlNGIxOTg5MzkwNWQzZWMyMDUyM2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODMyNzI4ZWQ4NzgyNDJmNDg3ZDM4OGFhMjhhN2QwODQgPSAkKCc8ZGl2IGlkPSJodG1sXzgzMjcyOGVkODc4MjQyZjQ4N2QzODhhYTI4YTdkMDg0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LZW5zaW5ndG9uIE1hcmtldCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMjVmYmE3MDZkYmRlNGIxOTg5MzkwNWQzZWMyMDUyM2Quc2V0Q29udGVudChodG1sXzgzMjcyOGVkODc4MjQyZjQ4N2QzODhhYTI4YTdkMDg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE3MjYxODQ2ZjA2ZDQ3N2RiNTdiMzk1N2Q0MzFjZjQ1LmJpbmRQb3B1cChwb3B1cF8yNWZiYTcwNmRiZGU0YjE5ODkzOTA1ZDNlYzIwNTIzZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wNjk0ZWY0MjMyMmU0MjE1OGU3YjY0NDg0YWFlMDQzOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgxNTI1MjIsLTc5LjI4NDU3NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80YjBlYjkxZTMzMGM0NjYwOGQ0NmQwODRhNzgwMjUyYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMTA0MzFiYzMxZmE0ZmE3OWUzMTFhZWRlMmEyNGNlMSA9ICQoJzxkaXYgaWQ9Imh0bWxfZjEwNDMxYmMzMWZhNGZhNzllMzExYWVkZTJhMjRjZTEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFnaW5jb3VydCBOb3J0aCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRiMGViOTFlMzMwYzQ2NjA4ZDQ2ZDA4NGE3ODAyNTJjLnNldENvbnRlbnQoaHRtbF9mMTA0MzFiYzMxZmE0ZmE3OWUzMTFhZWRlMmEyNGNlMSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wNjk0ZWY0MjMyMmU0MjE1OGU3YjY0NDg0YWFlMDQzOS5iaW5kUG9wdXAocG9wdXBfNGIwZWI5MWUzMzBjNDY2MDhkNDZkMDg0YTc4MDI1MmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDRkNjZlMDJjNTM4NGUyOWJhMjMwZTJmMjJlYmZkOTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My44MTUyNTIyLC03OS4yODQ1NzcyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWRkYTk1NDYxOGUzNDVhZWI1YmI1MTQxY2UxYWMwODAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYWM0MjNiN2NmODQ3NDkyZmE5Njg1MjhkMGYzMGNkMDQgPSAkKCc8ZGl2IGlkPSJodG1sX2FjNDIzYjdjZjg0NzQ5MmZhOTY4NTI4ZDBmMzBjZDA0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5MJiMzOTtBbW9yZWF1eCBFYXN0LCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWRkYTk1NDYxOGUzNDVhZWI1YmI1MTQxY2UxYWMwODAuc2V0Q29udGVudChodG1sX2FjNDIzYjdjZjg0NzQ5MmZhOTY4NTI4ZDBmMzBjZDA0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Q0ZDY2ZTAyYzUzODRlMjliYTIzMGUyZjIyZWJmZDk0LmJpbmRQb3B1cChwb3B1cF9hZGRhOTU0NjE4ZTM0NWFlYjViYjUxNDFjZTFhYzA4MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83YjkwYTM3ZWNkZjQ0ZjFjYmUwM2Y5YzQ0MTExNDQ4NCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgxNTI1MjIsLTc5LjI4NDU3NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NzY4ZTE4NmU1OTM0ZDg1YjU5Mjc2OTI3MjRiNmU3MyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wMmI5NTAxNGEyZTI0Y2YzOWQxZTRjMTE5MmZkODdjZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMDJiOTUwMTRhMmUyNGNmMzlkMWU0YzExOTJmZDg3Y2QiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk1pbGxpa2VuLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTc2OGUxODZlNTkzNGQ4NWI1OTI3NjkyNzI0YjZlNzMuc2V0Q29udGVudChodG1sXzAyYjk1MDE0YTJlMjRjZjM5ZDFlNGMxMTkyZmQ4N2NkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdiOTBhMzdlY2RmNDRmMWNiZTAzZjljNDQxMTE0NDg0LmJpbmRQb3B1cChwb3B1cF85NzY4ZTE4NmU1OTM0ZDg1YjU5Mjc2OTI3MjRiNmU3Myk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zY2RmNDhmZDg0NjQ0NGJiYWFhMTllMGQ3NGE2MDg3MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgxNTI1MjIsLTc5LjI4NDU3NzJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zNjQwOThhYmE0YTA0NGQ2YjVjMTg0ZWEzNjYyNDgyMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82NzRmZjRjZTRhNTc0ZDdkYmI5N2Q1ZjFmOGQ5NGY3ZSA9ICQoJzxkaXYgaWQ9Imh0bWxfNjc0ZmY0Y2U0YTU3NGQ3ZGJiOTdkNWYxZjhkOTRmN2UiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0ZWVsZXMgRWFzdCwgU2NhcmJvcm91Z2g8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM2NDA5OGFiYTRhMDQ0ZDZiNWMxODRlYTM2NjI0ODIwLnNldENvbnRlbnQoaHRtbF82NzRmZjRjZTRhNTc0ZDdkYmI5N2Q1ZjFmOGQ5NGY3ZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zY2RmNDhmZDg0NjQ0NGJiYWFhMTllMGQ3NGE2MDg3My5iaW5kUG9wdXAocG9wdXBfMzY0MDk4YWJhNGEwNDRkNmI1YzE4NGVhMzY2MjQ4MjApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZWFiM2RhZGI5MDJjNGY0NjgzZmE2NWEzOTQ0ZmMyMzcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdlYTEwYTliMjk4MzQyNTdhMGFmYjgyYjc1NzhhOWM2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzViMWUzOWQ2NTdmMDQ1ZTU4ZTQzYmE2YmMxODUzN2RkID0gJCgnPGRpdiBpZD0iaHRtbF81YjFlMzlkNjU3ZjA0NWU1OGU0M2JhNmJjMTg1MzdkZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVlciBQYXJrLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdlYTEwYTliMjk4MzQyNTdhMGFmYjgyYjc1NzhhOWM2LnNldENvbnRlbnQoaHRtbF81YjFlMzlkNjU3ZjA0NWU1OGU0M2JhNmJjMTg1MzdkZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYWIzZGFkYjkwMmM0ZjQ2ODNmYTY1YTM5NDRmYzIzNy5iaW5kUG9wdXAocG9wdXBfN2VhMTBhOWIyOTgzNDI1N2EwYWZiODJiNzU3OGE5YzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTUwMjQwN2U0YTY0NGIyY2JjNTNmZTBkMzlhZmRmOWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U4ZDMwZjExYjRiMTRlZjJhYmU0Y2ZiMDQ5NTU2NzkwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RkYTBlN2VhNzczNzRkMTFiMDM3NzIwOWY4NTYyZTI3ID0gJCgnPGRpdiBpZD0iaHRtbF9kZGEwZTdlYTc3Mzc0ZDExYjAzNzcyMDlmODU2MmUyNyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rm9yZXN0IEhpbGwgU0UsIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZThkMzBmMTFiNGIxNGVmMmFiZTRjZmIwNDk1NTY3OTAuc2V0Q29udGVudChodG1sX2RkYTBlN2VhNzczNzRkMTFiMDM3NzIwOWY4NTYyZTI3KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE1MDI0MDdlNGE2NDRiMmNiYzUzZmUwZDM5YWZkZjlmLmJpbmRQb3B1cChwb3B1cF9lOGQzMGYxMWI0YjE0ZWYyYWJlNGNmYjA0OTU1Njc5MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83YzZjOTgzMDUwZDc0OTEyYTRmNWQ4YmQ4NmE3YzdkOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NjQxMjI5OTk5OTk5LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDBhZWUwNDlhZDczNDA4MGEzZTk0MjYzZjU0MzFjMjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTEyYTA3YzFiM2Y0NDBiMTkwYWRkYjQ5MWMwNzZjODQgPSAkKCc8ZGl2IGlkPSJodG1sXzUxMmEwN2MxYjNmNDQwYjE5MGFkZGI0OTFjMDc2Yzg0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SYXRobmVsbHksIENlbnRyYWwgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDBhZWUwNDlhZDczNDA4MGEzZTk0MjYzZjU0MzFjMjAuc2V0Q29udGVudChodG1sXzUxMmEwN2MxYjNmNDQwYjE5MGFkZGI0OTFjMDc2Yzg0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzdjNmM5ODMwNTBkNzQ5MTJhNGY1ZDhiZDg2YTdjN2Q5LmJpbmRQb3B1cChwb3B1cF9kMGFlZTA0OWFkNzM0MDgwYTNlOTQyNjNmNTQzMWMyMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hNTM4ODBjM2U1OWM0MWYzODgyY2NmNWE5MzM5MmIwMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY4NjQxMjI5OTk5OTk5LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzg3ZGE4YzcxMzcyNDE2OWJkOWU2MWQzYjE0YjVmNGYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMTg2ZjEzOTBiMmYxNDBkZGEwNTk1MGQyNzkxN2RlYzkgPSAkKCc8ZGl2IGlkPSJodG1sXzE4NmYxMzkwYjJmMTQwZGRhMDU5NTBkMjc5MTdkZWM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Tb3V0aCBIaWxsLCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM4N2RhOGM3MTM3MjQxNjliZDllNjFkM2IxNGI1ZjRmLnNldENvbnRlbnQoaHRtbF8xODZmMTM5MGIyZjE0MGRkYTA1OTUwZDI3OTE3ZGVjOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hNTM4ODBjM2U1OWM0MWYzODgyY2NmNWE5MzM5MmIwMi5iaW5kUG9wdXAocG9wdXBfMzg3ZGE4YzcxMzcyNDE2OWJkOWU2MWQzYjE0YjVmNGYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjJiMTE3ODkxOWZiNGViN2IwYjdhODQwMGJjM2NjYzcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42ODY0MTIyOTk5OTk5OSwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJiMGM2YjgyYjUwMjQ5NWViNTNiMjEyNWZkZmIyNzY3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2M3NDYzYzhmYzkwMDQzOTBiYjVjNjkzNTgxODMzMGJlID0gJCgnPGRpdiBpZD0iaHRtbF9jNzQ2M2M4ZmM5MDA0MzkwYmI1YzY5MzU4MTgzMzBiZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3VtbWVyaGlsbCBXZXN0LCBDZW50cmFsIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJiMGM2YjgyYjUwMjQ5NWViNTNiMjEyNWZkZmIyNzY3LnNldENvbnRlbnQoaHRtbF9jNzQ2M2M4ZmM5MDA0MzkwYmI1YzY5MzU4MTgzMzBiZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMmIxMTc4OTE5ZmI0ZWI3YjBiN2E4NDAwYmMzY2NjNy5iaW5kUG9wdXAocG9wdXBfMmIwYzZiODJiNTAyNDk1ZWI1M2IyMTI1ZmRmYjI3NjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzg5ZTA1ZGM0YmY0NDAxYWJlZjhjNjU3ZGU1NjAwODMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTc0Nzc0ODFmMTgzNDdjNjgyOGI2YjE1Y2NiYjJhNjAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2Y4YTU5YTI2NTBhNDJiOTkzZjEwYzIyY2NjYjQxYzAgPSAkKCc8ZGl2IGlkPSJodG1sX2NmOGE1OWEyNjUwYTQyYjk5M2YxMGMyMmNjY2I0MWMwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DTiBUb3dlciwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTc0Nzc0ODFmMTgzNDdjNjgyOGI2YjE1Y2NiYjJhNjAuc2V0Q29udGVudChodG1sX2NmOGE1OWEyNjUwYTQyYjk5M2YxMGMyMmNjY2I0MWMwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzc4OWUwNWRjNGJmNDQwMWFiZWY4YzY1N2RlNTYwMDgzLmJpbmRQb3B1cChwb3B1cF8xNzQ3NzQ4MWYxODM0N2M2ODI4YjZiMTVjY2JiMmE2MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYmE4YWRhZTNkMjI0ZGQ1OTljMTEwZGRhYTdmMjRmMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zYjYyZTY5NzM3NTU0MGRkYWRhZGQ3YTA0NjRjN2QyMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zOWVkMDg4Mjk5OTQ0MjAwYjZmNDU3MGM2ZGIzMDFjYSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzllZDA4ODI5OTk0NDIwMGI2ZjQ1NzBjNmRiMzAxY2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJhdGh1cnN0IFF1YXksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNiNjJlNjk3Mzc1NTQwZGRhZGFkZDdhMDQ2NGM3ZDIwLnNldENvbnRlbnQoaHRtbF8zOWVkMDg4Mjk5OTQ0MjAwYjZmNDU3MGM2ZGIzMDFjYSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYmE4YWRhZTNkMjI0ZGQ1OTljMTEwZGRhYTdmMjRmMS5iaW5kUG9wdXAocG9wdXBfM2I2MmU2OTczNzU1NDBkZGFkYWRkN2EwNDY0YzdkMjApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTc0NjE1YzI0NDA1NDgyMjg3MjI2MWIzZTM1YWE4MDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzlkZGQ1MGU5ZTYzNGRmY2I0MDRjZmU3OWI4YjE0NWIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjViMWNiYzdjMDQ0NDg1ZDg3Y2U0ZDllYTI0YmEzMDggPSAkKCc8ZGl2IGlkPSJodG1sX2I1YjFjYmM3YzA0NDQ4NWQ4N2NlNGQ5ZWEyNGJhMzA4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Jc2xhbmQgYWlycG9ydCwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYzlkZGQ1MGU5ZTYzNGRmY2I0MDRjZmU3OWI4YjE0NWIuc2V0Q29udGVudChodG1sX2I1YjFjYmM3YzA0NDQ4NWQ4N2NlNGQ5ZWEyNGJhMzA4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE3NDYxNWMyNDQwNTQ4MjI4NzIyNjFiM2UzNWFhODA5LmJpbmRQb3B1cChwb3B1cF9jOWRkZDUwZTllNjM0ZGZjYjQwNGNmZTc5YjhiMTQ1Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84OGNiZWVmMGE1YTg0ZDU3YmUwMzk0YWExODdjNDg0ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hNjgwYTk4OTgxODQ0MjhlYTI0NDAzMGQ5Nzg0YTM2YSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85OWIzZDA2M2U4NTg0YjA3YmIyYWM5ZDZlMWQ3ZjVhMCA9ICQoJzxkaXYgaWQ9Imh0bWxfOTliM2QwNjNlODU4NGIwN2JiMmFjOWQ2ZTFkN2Y1YTAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCBXZXN0LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hNjgwYTk4OTgxODQ0MjhlYTI0NDAzMGQ5Nzg0YTM2YS5zZXRDb250ZW50KGh0bWxfOTliM2QwNjNlODU4NGIwN2JiMmFjOWQ2ZTFkN2Y1YTApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODhjYmVlZjBhNWE4NGQ1N2JlMDM5NGFhMTg3YzQ4NGUuYmluZFBvcHVwKHBvcHVwX2E2ODBhOTg5ODE4NDQyOGVhMjQ0MDMwZDk3ODRhMzZhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MwZDhlNTlmNzI4ZTQ0NjFiNDVlYTkzZWUwYzY5ZGVlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2U1NThlMjJlYzI3MTQ3YTliMWQ2Y2ZjODU5OGRiYjliID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2UzOWZlYjAyYzYyNzRmY2Y5MzY2NzJiNGRlZjM1OWNiID0gJCgnPGRpdiBpZD0iaHRtbF9lMzlmZWIwMmM2Mjc0ZmNmOTM2NjcyYjRkZWYzNTljYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZyBhbmQgU3BhZGluYSwgRG93bnRvd24gVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZTU1OGUyMmVjMjcxNDdhOWIxZDZjZmM4NTk4ZGJiOWIuc2V0Q29udGVudChodG1sX2UzOWZlYjAyYzYyNzRmY2Y5MzY2NzJiNGRlZjM1OWNiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MwZDhlNTlmNzI4ZTQ0NjFiNDVlYTkzZWUwYzY5ZGVlLmJpbmRQb3B1cChwb3B1cF9lNTU4ZTIyZWMyNzE0N2E5YjFkNmNmYzg1OThkYmI5Yik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83MDY5NmIyMjM1NjA0NjhkOWJlYmJlNjNmMzA4YWQwZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MTdmODQ0NWY0YTU0ZjIzYTFlN2NjYTU4NDhmMGRiYyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZTA4M2RkZGQ1MzE0NDEzYjlmMDU5NTZmZWNlNGFjNiA9ICQoJzxkaXYgaWQ9Imh0bWxfZWUwODNkZGRkNTMxNDQxM2I5ZjA1OTU2ZmVjZTRhYzYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJhaWx3YXkgTGFuZHMsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkxN2Y4NDQ1ZjRhNTRmMjNhMWU3Y2NhNTg0OGYwZGJjLnNldENvbnRlbnQoaHRtbF9lZTA4M2RkZGQ1MzE0NDEzYjlmMDU5NTZmZWNlNGFjNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MDY5NmIyMjM1NjA0NjhkOWJlYmJlNjNmMzA4YWQwZS5iaW5kUG9wdXAocG9wdXBfOTE3Zjg0NDVmNGE1NGYyM2ExZTdjY2E1ODQ4ZjBkYmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDZlNmVkMjEyNjk2NGJjY2EwNjExMGQ2MTM2NGRiODggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzkzZjMyNTY4YzIyNDBhZWJiZWNmYjUwMjY2NTJjM2EgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDMwZDlmMWIwNjA3NGUyODk5NzE5ODUxOTYwM2YwM2YgPSAkKCc8ZGl2IGlkPSJodG1sX2QzMGQ5ZjFiMDYwNzRlMjg5OTcxOTg1MTk2MDNmMDNmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Tb3V0aCBOaWFnYXJhLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zOTNmMzI1NjhjMjI0MGFlYmJlY2ZiNTAyNjY1MmMzYS5zZXRDb250ZW50KGh0bWxfZDMwZDlmMWIwNjA3NGUyODk5NzE5ODUxOTYwM2YwM2YpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDZlNmVkMjEyNjk2NGJjY2EwNjExMGQ2MTM2NGRiODguYmluZFBvcHVwKHBvcHVwXzM5M2YzMjU2OGMyMjQwYWViYmVjZmI1MDI2NjUyYzNhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUxZGU2Zjc1NjM3YzRhMmFiZGFmNGRiMzk0MWVkN2Q2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjA1NjQ2NiwtNzkuNTAxMzIwNzAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80MmE5YTI0NWFiY2U0ODQ1ODA5MDEwNDUxYjUyZmMzMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yY2M3ODYyNjExMTE0MmI5YmE1YmI2OWRlOWM0MmJiNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMmNjNzg2MjYxMTExNDJiOWJhNWJiNjlkZTljNDJiYjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkh1bWJlciBCYXkgU2hvcmVzLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzQyYTlhMjQ1YWJjZTQ4NDU4MDkwMTA0NTFiNTJmYzMzLnNldENvbnRlbnQoaHRtbF8yY2M3ODYyNjExMTE0MmI5YmE1YmI2OWRlOWM0MmJiNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81MWRlNmY3NTYzN2M0YTJhYmRhZjRkYjM5NDFlZDdkNi5iaW5kUG9wdXAocG9wdXBfNDJhOWEyNDVhYmNlNDg0NTgwOTAxMDQ1MWI1MmZjMzMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzA3MTRkOTNhZDc3NDE2ZGE2OTkyZWY5MzE3YTNiMWYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MDU2NDY2LC03OS41MDEzMjA3MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZlOTI0OGYzMGM0NTRhZjY5NWE3Y2NmNzc2Y2NhYjhjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZiZWY1MmZjNmFkMjQ2NzhiMmExYjIzY2U1ZGUyODYwID0gJCgnPGRpdiBpZD0iaHRtbF82YmVmNTJmYzZhZDI0Njc4YjJhMWIyM2NlNWRlMjg2MCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWltaWNvIFNvdXRoLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2ZlOTI0OGYzMGM0NTRhZjY5NWE3Y2NmNzc2Y2NhYjhjLnNldENvbnRlbnQoaHRtbF82YmVmNTJmYzZhZDI0Njc4YjJhMWIyM2NlNWRlMjg2MCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jMDcxNGQ5M2FkNzc0MTZkYTY5OTJlZjkzMTdhM2IxZi5iaW5kUG9wdXAocG9wdXBfZmU5MjQ4ZjMwYzQ1NGFmNjk1YTdjY2Y3NzZjY2FiOGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDYyNzczYmE4MTAwNDgwYzhhZGM0YWZhNzY1ZTViZDMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MDU2NDY2LC03OS41MDEzMjA3MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2E1MjVkM2M2NGQ5NDRkZmNiMzI4MDkxNDJmZjQ5NjU1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc0OTFhMzViYjNiOTQxMTM4NDY0Nzc0NjczN2U2ZGJkID0gJCgnPGRpdiBpZD0iaHRtbF83NDkxYTM1YmIzYjk0MTEzODQ2NDc3NDY3MzdlNmRiZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TmV3IFRvcm9udG8sIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYTUyNWQzYzY0ZDk0NGRmY2IzMjgwOTE0MmZmNDk2NTUuc2V0Q29udGVudChodG1sXzc0OTFhMzViYjNiOTQxMTM4NDY0Nzc0NjczN2U2ZGJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA2Mjc3M2JhODEwMDQ4MGM4YWRjNGFmYTc2NWU1YmQzLmJpbmRQb3B1cChwb3B1cF9hNTI1ZDNjNjRkOTQ0ZGZjYjMyODA5MTQyZmY0OTY1NSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xNTA2YTQ3NWM3MTY0ZDJiYmQwYmMyNWZlYmE1NWUwOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwtNzkuNTg4NDM2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2I5NjQxNjIxOTY1ZDQ2NGI4YzYxNmQ0ODI1ZTJkOWFjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzZlY2YyMmIzNzQzMjRlNDc4MjBmZjA0MWI2M2M5N2NhID0gJCgnPGRpdiBpZD0iaHRtbF82ZWNmMjJiMzc0MzI0ZTQ3ODIwZmYwNDFiNjNjOTdjYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QWxiaW9uIEdhcmRlbnMsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjk2NDE2MjE5NjVkNDY0YjhjNjE2ZDQ4MjVlMmQ5YWMuc2V0Q29udGVudChodG1sXzZlY2YyMmIzNzQzMjRlNDc4MjBmZjA0MWI2M2M5N2NhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzE1MDZhNDc1YzcxNjRkMmJiZDBiYzI1ZmViYTU1ZTA5LmJpbmRQb3B1cChwb3B1cF9iOTY0MTYyMTk2NWQ0NjRiOGM2MTZkNDgyNWUyZDlhYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zMmYyYmFjZjMzZDE0ZDA5YWQxNjYyMTE2NTFjZDUyZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjczOTQxNjM5OTk5OTk5NiwtNzkuNTg4NDM2OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdjYjE0MjI5OGNkMTRmOWU4MjFjZDY2NWIwOWMwMTU4ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQzM2E5ZjJkYzlkMTQxYzk4NmU3MjE2OWI5YTBjOTE2ID0gJCgnPGRpdiBpZD0iaHRtbF80MzNhOWYyZGM5ZDE0MWM5ODZlNzIxNjliOWEwYzkxNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVhdW1vbmQgSGVpZ2h0cywgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83Y2IxNDIyOThjZDE0ZjllODIxY2Q2NjViMDljMDE1OC5zZXRDb250ZW50KGh0bWxfNDMzYTlmMmRjOWQxNDFjOTg2ZTcyMTY5YjlhMGM5MTYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMzJmMmJhY2YzM2QxNGQwOWFkMTY2MjExNjUxY2Q1MmUuYmluZFBvcHVwKHBvcHVwXzdjYjE0MjI5OGNkMTRmOWU4MjFjZDY2NWIwOWMwMTU4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzM4ODY5ZjAyNDMxODQwNzY4YjJiODczOTAxOGRhYTY0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5NDE2Mzk5OTk5OTk2LC03OS41ODg0MzY5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWQ4NTJiNzAzODA5NDVlNjlkY2M3MjgwODY2MDQ0OTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZTViYWJiMmM3YTBmNGMzMGIwYzc5M2Q0ZmNkYWU4YzkgPSAkKCc8ZGl2IGlkPSJodG1sX2U1YmFiYjJjN2EwZjRjMzBiMGM3OTNkNGZjZGFlOGM5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXJnYXRlLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVkODUyYjcwMzgwOTQ1ZTY5ZGNjNzI4MDg2NjA0NDk1LnNldENvbnRlbnQoaHRtbF9lNWJhYmIyYzdhMGY0YzMwYjBjNzkzZDRmY2RhZThjOSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zODg2OWYwMjQzMTg0MDc2OGIyYjg3MzkwMThkYWE2NC5iaW5kUG9wdXAocG9wdXBfNWQ4NTJiNzAzODA5NDVlNjlkY2M3MjgwODY2MDQ0OTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODFkZjRhZWMwZjQ3NDQ5YTg4MjkwZTM4MWIzOGQ5MjYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mzk0MTYzOTk5OTk5OTYsLTc5LjU4ODQzNjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iNTNlMGQ2OTcxYWU0MGVmOWU3MTM2ZjUyOGZiYjhmMSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84MWRjNmRiYmVmZDY0MGEzOWNlNWRhMzRlY2QyYjBiNSA9ICQoJzxkaXYgaWQ9Imh0bWxfODFkYzZkYmJlZmQ2NDBhMzljZTVkYTM0ZWNkMmIwYjUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkphbWVzdG93biwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iNTNlMGQ2OTcxYWU0MGVmOWU3MTM2ZjUyOGZiYjhmMS5zZXRDb250ZW50KGh0bWxfODFkYzZkYmJlZmQ2NDBhMzljZTVkYTM0ZWNkMmIwYjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfODFkZjRhZWMwZjQ3NDQ5YTg4MjkwZTM4MWIzOGQ5MjYuYmluZFBvcHVwKHBvcHVwX2I1M2UwZDY5NzFhZTQwZWY5ZTcxMzZmNTI4ZmJiOGYxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJiZTY2MDVkNDUyZjRlNTk5ZmQzYzI0ZDlhYzZhMDgwID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5NDE2Mzk5OTk5OTk2LC03OS41ODg0MzY5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWQ5NjE0MjFmOTc1NGEyMDhiMjAzZDRhMWZjNzExMGMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZmFhNTNlOTQzY2JkNDc4MThjZjIzNTczNTBkMjE0YTUgPSAkKCc8ZGl2IGlkPSJodG1sX2ZhYTUzZTk0M2NiZDQ3ODE4Y2YyMzU3MzUwZDIxNGE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Nb3VudCBPbGl2ZSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZDk2MTQyMWY5NzU0YTIwOGIyMDNkNGExZmM3MTEwYy5zZXRDb250ZW50KGh0bWxfZmFhNTNlOTQzY2JkNDc4MThjZjIzNTczNTBkMjE0YTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMmJlNjYwNWQ0NTJmNGU1OTlmZDNjMjRkOWFjNmEwODAuYmluZFBvcHVwKHBvcHVwX2FkOTYxNDIxZjk3NTRhMjA4YjIwM2Q0YTFmYzcxMTBjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzdlMmE3ODI5M2IyMDQ2OTRhYzA1MTUxMWE1ZWM1Nzk3ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5NDE2Mzk5OTk5OTk2LC03OS41ODg0MzY5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODkyZDJlYTA5YjExNDBjYThjZWQzZTJlNzU1OGRlYzMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjM2ZGRlOThjM2NkNGE0OTg0NmVhMDUxYmExYTU1ZWUgPSAkKCc8ZGl2IGlkPSJodG1sX2YzNmRkZTk4YzNjZDRhNDk4NDZlYTA1MWJhMWE1NWVlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TaWx2ZXJzdG9uZSwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF84OTJkMmVhMDliMTE0MGNhOGNlZDNlMmU3NTU4ZGVjMy5zZXRDb250ZW50KGh0bWxfZjM2ZGRlOThjM2NkNGE0OTg0NmVhMDUxYmExYTU1ZWUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfN2UyYTc4MjkzYjIwNDY5NGFjMDUxNTExYTVlYzU3OTcuYmluZFBvcHVwKHBvcHVwXzg5MmQyZWEwOWIxMTQwY2E4Y2VkM2UyZTc1NThkZWMzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIxNGI4MjlmZmFlZjQxNGFhZDE0ZGRhMjExZDAzZDU5ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzM5NDE2Mzk5OTk5OTk2LC03OS41ODg0MzY5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmI2ZGM4NzQwMzA1NDdiYThhODdmOTExN2Q4OWVjODUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWMxMmFlNTMyY2JiNDZlMDhhODkzMTUzMDUwZWM2NmYgPSAkKCc8ZGl2IGlkPSJodG1sXzVjMTJhZTUzMmNiYjQ2ZTA4YTg5MzE1MzA1MGVjNjZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Tb3V0aCBTdGVlbGVzLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2JiNmRjODc0MDMwNTQ3YmE4YTg3ZjkxMTdkODllYzg1LnNldENvbnRlbnQoaHRtbF81YzEyYWU1MzJjYmI0NmUwOGE4OTMxNTMwNTBlYzY2Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMTRiODI5ZmZhZWY0MTRhYWQxNGRkYTIxMWQwM2Q1OS5iaW5kUG9wdXAocG9wdXBfYmI2ZGM4NzQwMzA1NDdiYThhODdmOTExN2Q4OWVjODUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWNkZTQyYjI0ZTIxNDFlZTliMDRlZmQ3NmRlOGYxY2UgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43Mzk0MTYzOTk5OTk5OTYsLTc5LjU4ODQzNjldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yN2ExMTc0ODk4MWI0Yjk4YTg2NTM4N2UyNGI3ZWMzMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85M2ZhZDM3MDc5MWE0ZjdmOTliNjZkNzhhOGUxZWQ0YiA9ICQoJzxkaXYgaWQ9Imh0bWxfOTNmYWQzNzA3OTFhNGY3Zjk5YjY2ZDc4YThlMWVkNGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoaXN0bGV0b3duLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI3YTExNzQ4OTgxYjRiOThhODY1Mzg3ZTI0YjdlYzMwLnNldENvbnRlbnQoaHRtbF85M2ZhZDM3MDc5MWE0ZjdmOTliNjZkNzhhOGUxZWQ0Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85Y2RlNDJiMjRlMjE0MWVlOWIwNGVmZDc2ZGU4ZjFjZS5iaW5kUG9wdXAocG9wdXBfMjdhMTE3NDg5ODFiNGI5OGE4NjUzODdlMjRiN2VjMzApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTFlZGM1MmQ3NWFhNGU5NzhjOWI4ODk4MjhjNTEyNTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My43OTk1MjUyMDAwMDAwMDUsLTc5LjMxODM4ODddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YzBjOGVkNjJhNDM0Y2E0OWNlMjM4ZGE3OTg4ODM2YiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zNjYyN2M4ZjEwYTI0ZDE3YTA5NDUzMTg0ZGMxMjEzYSA9ICQoJzxkaXYgaWQ9Imh0bWxfMzY2MjdjOGYxMGEyNGQxN2EwOTQ1MzE4NGRjMTIxM2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkwmIzM5O0Ftb3JlYXV4IFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82YzBjOGVkNjJhNDM0Y2E0OWNlMjM4ZGE3OTg4ODM2Yi5zZXRDb250ZW50KGh0bWxfMzY2MjdjOGYxMGEyNGQxN2EwOTQ1MzE4NGRjMTIxM2EpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTFlZGM1MmQ3NWFhNGU5NzhjOWI4ODk4MjhjNTEyNTUuYmluZFBvcHVwKHBvcHVwXzZjMGM4ZWQ2MmE0MzRjYTQ5Y2UyMzhkYTc5ODg4MzZiKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ5NzQ0YmJlYjA4NTQ3YjhiZTg4ZWY0YWE2NWRhMGU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzk5NTI1MjAwMDAwMDA1LC03OS4zMTgzODg3XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTI2ZjY3ODJkYzljNGMxODhhMGQxYTkwMGFjMjAyNjkgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2RjNTBmNTk2OGFmNGRhNjljZGJmN2IyMzdlOTZmZDAgPSAkKCc8ZGl2IGlkPSJodG1sXzdkYzUwZjU5NjhhZjRkYTY5Y2RiZjdiMjM3ZTk2ZmQwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdGVlbGVzIFdlc3QsIFNjYXJib3JvdWdoPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hMjZmNjc4MmRjOWM0YzE4OGEwZDFhOTAwYWMyMDI2OS5zZXRDb250ZW50KGh0bWxfN2RjNTBmNTk2OGFmNGRhNjljZGJmN2IyMzdlOTZmZDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDk3NDRiYmViMDg1NDdiOGJlODhlZjRhYTY1ZGEwZTguYmluZFBvcHVwKHBvcHVwX2EyNmY2NzgyZGM5YzRjMTg4YTBkMWE5MDBhYzIwMjY5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNkYmEwZTE3Njk4YzQ3YjhiNGYxMGJmOGU4YjM0MGI0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjc5NTYyNiwtNzkuMzc3NTI5NDAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82ZGIzYTZkZTY3ZTY0NjNiYTE2ZWQ5OTRhYjdmZmQwNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85Mjc0ZmJlOGI5YTM0MDUzYWNiZTA5NzE1MjcyYTEwMSA9ICQoJzxkaXYgaWQ9Imh0bWxfOTI3NGZiZThiOWEzNDA1M2FjYmUwOTcxNTI3MmExMDEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJvc2VkYWxlLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82ZGIzYTZkZTY3ZTY0NjNiYTE2ZWQ5OTRhYjdmZmQwNy5zZXRDb250ZW50KGh0bWxfOTI3NGZiZThiOWEzNDA1M2FjYmUwOTcxNTI3MmExMDEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2RiYTBlMTc2OThjNDdiOGI0ZjEwYmY4ZThiMzQwYjQuYmluZFBvcHVwKHBvcHVwXzZkYjNhNmRlNjdlNjQ2M2JhMTZlZDk5NGFiN2ZmZDA3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzU1MzZhZmIyZGE5YjQzMWJhOTcwNjZkOTk2YmUxNzIzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YjBmNTAxM2MyZjA0NzcxYjBiODA1YWRjMDVhOWY2OSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83Zjc4ZGZhOTkwMzg0NGZhOTllMTE3MGRmMDk2MzQyMiA9ICQoJzxkaXYgaWQ9Imh0bWxfN2Y3OGRmYTk5MDM4NDRmYTk5ZTExNzBkZjA5NjM0MjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzIDI1IFRoZSBFc3BsYW5hZGUsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZiMGY1MDEzYzJmMDQ3NzFiMGI4MDVhZGMwNWE5ZjY5LnNldENvbnRlbnQoaHRtbF83Zjc4ZGZhOTkwMzg0NGZhOTllMTE3MGRmMDk2MzQyMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NTM2YWZiMmRhOWI0MzFiYTk3MDY2ZDk5NmJlMTcyMy5iaW5kUG9wdXAocG9wdXBfNmIwZjUwMTNjMmYwNDc3MWIwYjgwNWFkYzA1YTlmNjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTk5YzA0YjMyNTI5NDUzYjk1MDMyNGFlYjM4ZTM5ZTYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MDI0MTM3MDAwMDAwMSwtNzkuNTQzNDg0MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80ZGUyNDQ2MTFkYWE0MTMzYjQ0OTIzOTM3ODJhOWNjNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NWJlNmNmMWNkNzc0YzFlYWU1ZDlhOGFkNmFhZTNmZSA9ICQoJzxkaXYgaWQ9Imh0bWxfODViZTZjZjFjZDc3NGMxZWFlNWQ5YThhZDZhYWUzZmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFsZGVyd29vZCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF80ZGUyNDQ2MTFkYWE0MTMzYjQ0OTIzOTM3ODJhOWNjNC5zZXRDb250ZW50KGh0bWxfODViZTZjZjFjZDc3NGMxZWFlNWQ5YThhZDZhYWUzZmUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNTk5YzA0YjMyNTI5NDUzYjk1MDMyNGFlYjM4ZTM5ZTYuYmluZFBvcHVwKHBvcHVwXzRkZTI0NDYxMWRhYTQxMzNiNDQ5MjM5Mzc4MmE5Y2M0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQwMDYxZDRiYzBjMTRhYWNiYTBkMDg2MmNlYWMxN2RlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjAyNDEzNzAwMDAwMDEsLTc5LjU0MzQ4NDA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMmU5OTQ3Y2FhY2I5NDY5NDlkYzBlNjFmMTVmYTBmYzQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNTQzOWFhYWFjZDQ0NDc1MWJhMjEwY2ZiYjgzMTBhMDAgPSAkKCc8ZGl2IGlkPSJodG1sXzU0MzlhYWFhY2Q0NDQ3NTFiYTIxMGNmYmI4MzEwYTAwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Mb25nIEJyYW5jaCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZTk5NDdjYWFjYjk0Njk0OWRjMGU2MWYxNWZhMGZjNC5zZXRDb250ZW50KGh0bWxfNTQzOWFhYWFjZDQ0NDc1MWJhMjEwY2ZiYjgzMTBhMDApOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNDAwNjFkNGJjMGMxNGFhY2JhMGQwODYyY2VhYzE3ZGUuYmluZFBvcHVwKHBvcHVwXzJlOTk0N2NhYWNiOTQ2OTQ5ZGMwZTYxZjE1ZmEwZmM0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2M0OTZlNzllNzcwYzQzNDhiZjg5NGFkNWNkODhmM2RlID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNzA2NzQ4Mjk5OTk5OTk0LC03OS41OTQwNTQ0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjIyMDUyZjJiYjNmNDE4MmFkZDA4ZDNiY2YzZjM2NjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzZlOTNlYmFhYTEyNDgxMmIyYmY2YmE3MzhmY2UyZmMgPSAkKCc8ZGl2IGlkPSJodG1sXzc2ZTkzZWJhYWExMjQ4MTJiMmJmNmJhNzM4ZmNlMmZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ob3J0aHdlc3QsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjIyMDUyZjJiYjNmNDE4MmFkZDA4ZDNiY2YzZjM2NjQuc2V0Q29udGVudChodG1sXzc2ZTkzZWJhYWExMjQ4MTJiMmJmNmJhNzM4ZmNlMmZjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M0OTZlNzllNzcwYzQzNDhiZjg5NGFkNWNkODhmM2RlLmJpbmRQb3B1cChwb3B1cF9mMjIwNTJmMmJiM2Y0MTgyYWRkMDhkM2JjZjNmMzY2NCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kYzJjYzU2NTQyNzQ0ZjgwYWE1N2Q5YTZiM2FiNzEzMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjgzNjEyNDcwMDAwMDAwNiwtNzkuMjA1NjM2MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNDZkYTI2MmIwNTU0YTI3YjllMWYxM2IxZDY3YzYyNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82YzVlMzg5ZWYzMjM0YzNkYmQyYzkxNzA3NjcyMDlmMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNmM1ZTM4OWVmMzIzNGMzZGJkMmM5MTcwNzY3MjA5ZjEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVwcGVyIFJvdWdlLCBTY2FyYm9yb3VnaDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjQ2ZGEyNjJiMDU1NGEyN2I5ZTFmMTNiMWQ2N2M2MjYuc2V0Q29udGVudChodG1sXzZjNWUzODllZjMyMzRjM2RiZDJjOTE3MDc2NzIwOWYxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2RjMmNjNTY1NDI3NDRmODBhYTU3ZDlhNmIzYWI3MTMzLmJpbmRQb3B1cChwb3B1cF9mNDZkYTI2MmIwNTU0YTI3YjllMWYxM2IxZDY3YzYyNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYjhiY2U0ZjEwYzA0OTY4Yjc2NTUxZTc3ZDNmN2RmNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2QwN2MwMzdjNGRlZTRjZDY5ZTQ2NGY4Mjg3ZmQ0NzE5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzkwZGNjMjMyNmY1ZDQzNTViYzZhZDc2MGVlNTZhY2JlID0gJCgnPGRpdiBpZD0iaHRtbF85MGRjYzIzMjZmNWQ0MzU1YmM2YWQ3NjBlZTU2YWNiZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2FiYmFnZXRvd24sIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2QwN2MwMzdjNGRlZTRjZDY5ZTQ2NGY4Mjg3ZmQ0NzE5LnNldENvbnRlbnQoaHRtbF85MGRjYzIzMjZmNWQ0MzU1YmM2YWQ3NjBlZTU2YWNiZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYjhiY2U0ZjEwYzA0OTY4Yjc2NTUxZTc3ZDNmN2RmNS5iaW5kUG9wdXAocG9wdXBfZDA3YzAzN2M0ZGVlNGNkNjllNDY0ZjgyODdmZDQ3MTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDc4ZDZjNWE4N2ExNGYyNTlmNWNmNGNkNDdiYjIxYzcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMGY1ZjJlNzhlOGQ0ZDlmOThiYjE3YWIzMGE5ZjAzMyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NGIwYmNkMWY0NDk0ODQ2YTE2YTNhZDM5NWE0MjE4NyA9ICQoJzxkaXYgaWQ9Imh0bWxfODRiMGJjZDFmNDQ5NDg0NmExNmEzYWQzOTVhNDIxODciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duLCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMGY1ZjJlNzhlOGQ0ZDlmOThiYjE3YWIzMGE5ZjAzMy5zZXRDb250ZW50KGh0bWxfODRiMGJjZDFmNDQ5NDg0NmExNmEzYWQzOTVhNDIxODcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDc4ZDZjNWE4N2ExNGYyNTlmNWNmNGNkNDdiYjIxYzcuYmluZFBvcHVwKHBvcHVwXzMwZjVmMmU3OGU4ZDRkOWY5OGJiMTdhYjMwYTlmMDMzKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJhZTdkM2ExMzViZjQ3ODBhMzJjNTI4MjI4YTI5ZGI1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJlNmJkYTVlY2RmZjQxMzk4OTY4NzA5NWM3M2JkYjhiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FhZWIyMmNjMjc5NTRjNDZhNzBlYWQxNDkzMTE1YWJmID0gJCgnPGRpdiBpZD0iaHRtbF9hYWViMjJjYzI3OTU0YzQ2YTcwZWFkMTQ5MzExNWFiZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Rmlyc3QgQ2FuYWRpYW4gUGxhY2UsIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJlNmJkYTVlY2RmZjQxMzk4OTY4NzA5NWM3M2JkYjhiLnNldENvbnRlbnQoaHRtbF9hYWViMjJjYzI3OTU0YzQ2YTcwZWFkMTQ5MzExNWFiZik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYWU3ZDNhMTM1YmY0NzgwYTMyYzUyODIyOGEyOWRiNS5iaW5kUG9wdXAocG9wdXBfMmU2YmRhNWVjZGZmNDEzOTg5Njg3MDk1YzczYmRiOGIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfY2JjOTZiNzEzZTRmNDUwM2IxZjA3MTY4NjY2YjE4OWMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDFlMTY2Y2M2ZjUxNGQ2NzlmMGU5MDE0ZGI4NjgzYWMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjBmODA2ZTIxOGRhNGUzZTg1M2Y0N2FlMjQ0ZDIyZGYgPSAkKCc8ZGl2IGlkPSJodG1sX2YwZjgwNmUyMThkYTRlM2U4NTNmNDdhZTI0NGQyMmRmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5VbmRlcmdyb3VuZCBjaXR5LCBEb3dudG93biBUb3JvbnRvPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9kMWUxNjZjYzZmNTE0ZDY3OWYwZTkwMTRkYjg2ODNhYy5zZXRDb250ZW50KGh0bWxfZjBmODA2ZTIxOGRhNGUzZTg1M2Y0N2FlMjQ0ZDIyZGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfY2JjOTZiNzEzZTRmNDUwM2IxZjA3MTY4NjY2YjE4OWMuYmluZFBvcHVwKHBvcHVwX2QxZTE2NmNjNmY1MTRkNjc5ZjBlOTAxNGRiODY4M2FjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzgwMTk2Y2M1NmU0ODQ0NTU5YzY5NWFjMmFjMDU1ZGQ1ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzNjUzNjAwMDAwMDA1LC03OS41MDY5NDM2XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMTAzZmQ4ZGJjZTUxNGY0Y2JmOWQzYTVhN2NjNGE3M2IgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfODU5YzI4ZmJhODE3NDdhZWEzN2Q4M2FiYWU1ZmZlZTAgPSAkKCc8ZGl2IGlkPSJodG1sXzg1OWMyOGZiYTgxNzQ3YWVhMzdkODNhYmFlNWZmZWUwIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5UaGUgS2luZ3N3YXksIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTAzZmQ4ZGJjZTUxNGY0Y2JmOWQzYTVhN2NjNGE3M2Iuc2V0Q29udGVudChodG1sXzg1OWMyOGZiYTgxNzQ3YWVhMzdkODNhYmFlNWZmZWUwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzgwMTk2Y2M1NmU0ODQ0NTU5YzY5NWFjMmFjMDU1ZGQ1LmJpbmRQb3B1cChwb3B1cF8xMDNmZDhkYmNlNTE0ZjRjYmY5ZDNhNWE3Y2M0YTczYik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yMmI4N2U2NDVmYjM0NTFmYjk3ZGI4ZjQ0YjBjZGU3YyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzY1MzYwMDAwMDAwNSwtNzkuNTA2OTQzNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzU1MjQxYjkzMzg5MTRkYmRiYmMwNDE1NmNmODUwYjI5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhjMDVkNTI0ZWEyNDQxNWZhZThlMzFkM2E0OGNmMjhhID0gJCgnPGRpdiBpZD0iaHRtbF84YzA1ZDUyNGVhMjQ0MTVmYWU4ZTMxZDNhNDhjZjI4YSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TW9udGdvbWVyeSBSb2FkLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU1MjQxYjkzMzg5MTRkYmRiYmMwNDE1NmNmODUwYjI5LnNldENvbnRlbnQoaHRtbF84YzA1ZDUyNGVhMjQ0MTVmYWU4ZTMxZDNhNDhjZjI4YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yMmI4N2U2NDVmYjM0NTFmYjk3ZGI4ZjQ0YjBjZGU3Yy5iaW5kUG9wdXAocG9wdXBfNTUyNDFiOTMzODkxNGRiZGJiYzA0MTU2Y2Y4NTBiMjkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTIzZmJlNDJhZjAzNGY5NzgxYzAzNDU0MDIzNTc3ZGUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTM2NTM2MDAwMDAwMDUsLTc5LjUwNjk0MzZdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lYTQ2OTQzMDdmMzk0YTE4OTk4MGU4YTIxMjliMzRhNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jMzMyYWM4Nzg3ZGU0ZThkYWExNmI4Yzk0ZWQ0NTRmNCA9ICQoJzxkaXYgaWQ9Imh0bWxfYzMzMmFjODc4N2RlNGU4ZGFhMTZiOGM5NGVkNDU0ZjQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPk9sZCBNaWxsIE5vcnRoLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VhNDY5NDMwN2YzOTRhMTg5OTgwZThhMjEyOWIzNGE2LnNldENvbnRlbnQoaHRtbF9jMzMyYWM4Nzg3ZGU0ZThkYWExNmI4Yzk0ZWQ0NTRmNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xMjNmYmU0MmFmMDM0Zjk3ODFjMDM0NTQwMjM1NzdkZS5iaW5kUG9wdXAocG9wdXBfZWE0Njk0MzA3ZjM5NGExODk5ODBlOGEyMTI5YjM0YTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzA4N2NiYTZiZGNhNGI1OTgwYmQ0M2E3ZWMzZjI3YTcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY5OThiZTQ3NzgwNjQyZWI4MzI5OTliNmJlNjM2OTQ3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA4MzBhYTJkYWU5ZjQ0ZDZiMDhiMmY0MzE3ZmUyZmM4ID0gJCgnPGRpdiBpZD0iaHRtbF8wODMwYWEyZGFlOWY0NGQ2YjA4YjJmNDMxN2ZlMmZjOCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q2h1cmNoIGFuZCBXZWxsZXNsZXksIERvd250b3duIFRvcm9udG88L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzY5OThiZTQ3NzgwNjQyZWI4MzI5OTliNmJlNjM2OTQ3LnNldENvbnRlbnQoaHRtbF8wODMwYWEyZGFlOWY0NGQ2YjA4YjJmNDMxN2ZlMmZjOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MDg3Y2JhNmJkY2E0YjU5ODBiZDQzYTdlYzNmMjdhNy5iaW5kUG9wdXAocG9wdXBfNjk5OGJlNDc3ODA2NDJlYjgzMjk5OWI2YmU2MzY5NDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDZhYjA5NmQ3ODllNDhjOWFiNDI2ZWJlZmZlMzUzZDggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjI3NDM5LC03OS4zMjE1NThdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF82YzAyMWNhMzE2ZTQ0YTA5YjU5ZTQyMmYyMjc2MTA0ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNTViODBiMjRhNmU0YjUxOWNkYjdkZTA1MjNiZDFjMSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzU1YjgwYjI0YTZlNGI1MTljZGI3ZGUwNTIzYmQxYzEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJ1c2luZXNzIFJlcGx5IE1haWwgUHJvY2Vzc2luZyBDZW50cmUgOTY5IEVhc3Rlcm4sIEVhc3QgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNmMwMjFjYTMxNmU0NGEwOWI1OWU0MjJmMjI3NjEwNGQuc2V0Q29udGVudChodG1sX2M1NWI4MGIyNGE2ZTRiNTE5Y2RiN2RlMDUyM2JkMWMxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ2YWIwOTZkNzg5ZTQ4YzlhYjQyNmViZWZmZTM1M2Q4LmJpbmRQb3B1cChwb3B1cF82YzAyMWNhMzE2ZTQ0YTA5YjU5ZTQyMmYyMjc2MTA0ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lYjc4ODg3Nzg0NzM0ZGI4OTdmM2M2NjY4NTEzZjAxMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjI1NzksLTc5LjQ5ODUwOTA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTg2YTk1YmJlYWRkNDNlYTg5OTY5YmY3OGZlYTVmN2YgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjIxNjZjNjYwOTQ3NDcwZGFlNGIyNzA1NWY1YzAxOTYgPSAkKCc8ZGl2IGlkPSJodG1sX2IyMTY2YzY2MDk0NzQ3MGRhZTRiMjcwNTVmNWMwMTk2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IdW1iZXIgQmF5LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU4NmE5NWJiZWFkZDQzZWE4OTk2OWJmNzhmZWE1ZjdmLnNldENvbnRlbnQoaHRtbF9iMjE2NmM2NjA5NDc0NzBkYWU0YjI3MDU1ZjVjMDE5Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lYjc4ODg3Nzg0NzM0ZGI4OTdmM2M2NjY4NTEzZjAxMS5iaW5kUG9wdXAocG9wdXBfNTg2YTk1YmJlYWRkNDNlYTg5OTY5YmY3OGZlYTVmN2YpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTM1ZjUyMGY3NmFlNDc3ZTk1MWE3ODQwMzEyYzJlMGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42MzYyNTc5LC03OS40OTg1MDkwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBlZTE1ZGRhZDQyZDRlM2VhYjRmZmRlNDY3NDFmNjVmID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzU3MDFiOWUwMTcwYjQ5OWJiNTg1YTE5ZmIwMmU1MTEwID0gJCgnPGRpdiBpZD0iaHRtbF81NzAxYjllMDE3MGI0OTliYjU4NWExOWZiMDJlNTExMCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZyYjMzk7cyBNaWxsIFBhcmssIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMGVlMTVkZGFkNDJkNGUzZWFiNGZmZGU0Njc0MWY2NWYuc2V0Q29udGVudChodG1sXzU3MDFiOWUwMTcwYjQ5OWJiNTg1YTE5ZmIwMmU1MTEwKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2UzNWY1MjBmNzZhZTQ3N2U5NTFhNzg0MDMxMmMyZTBkLmJpbmRQb3B1cChwb3B1cF8wZWUxNWRkYWQ0MmQ0ZTNlYWI0ZmZkZTQ2NzQxZjY1Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wNzhkZGQ5NzU0ZWE0ZDEzYWNhMjUxZGMzZGU4MTkyMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjI1NzksLTc5LjQ5ODUwOTA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGVhMGVjNTA3OWM1NGFkMWJhOTg1M2ZkZmZlZTdjMTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjEwMWFlOTRkMzJiNGQwNmE5NmM1ZTNmZmNiMjliMzQgPSAkKCc8ZGl2IGlkPSJodG1sXzIxMDFhZTk0ZDMyYjRkMDZhOTZjNWUzZmZjYjI5YjM0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nc3dheSBQYXJrIFNvdXRoIEVhc3QsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGVhMGVjNTA3OWM1NGFkMWJhOTg1M2ZkZmZlZTdjMTYuc2V0Q29udGVudChodG1sXzIxMDFhZTk0ZDMyYjRkMDZhOTZjNWUzZmZjYjI5YjM0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA3OGRkZDk3NTRlYTRkMTNhY2EyNTFkYzNkZTgxOTIzLmJpbmRQb3B1cChwb3B1cF80ZWEwZWM1MDc5YzU0YWQxYmE5ODUzZmRmZmVlN2MxNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZmIyM2ZiZjEwMzA0MTU2YTRlOGQ4MDc3OGVjNjMwYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjI1NzksLTc5LjQ5ODUwOTA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYWY1ODY1NThkZDY1NDhiMWI4MjU2NjllZTA0Yzg5OWYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZGQwMmEyMmIxODI0NDE2ZWJjOWJjYTM2ODMwZWY0MmQgPSAkKCc8ZGl2IGlkPSJodG1sX2RkMDJhMjJiMTgyNDQxNmViYzliY2EzNjgzMGVmNDJkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5NaW1pY28gTkUsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYWY1ODY1NThkZDY1NDhiMWI4MjU2NjllZTA0Yzg5OWYuc2V0Q29udGVudChodG1sX2RkMDJhMjJiMTgyNDQxNmViYzliY2EzNjgzMGVmNDJkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJmYjIzZmJmMTAzMDQxNTZhNGU4ZDgwNzc4ZWM2MzBhLmJpbmRQb3B1cChwb3B1cF9hZjU4NjU1OGRkNjU0OGIxYjgyNTY2OWVlMDRjODk5Zik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xMzk5NDNiNTIzMWY0ZDMwODhhMzUxY2Y2NWQyYjY5ZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYzNjI1NzksLTc5LjQ5ODUwOTA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogInllbGxvdyIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuOCwKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfNjViODYxMTM4MTEwNDY4YmIyYWYxMmVjYTZhNTQ3MzkpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMjNhMjM3YzlmYTBhNGJkNmE1ZTRkOTA2NzcyZTIxM2MgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfY2FkOWFjNWUyMzY0NDMxNDhiMTY4N2M2NGI4OTNlMjQgPSAkKCc8ZGl2IGlkPSJodG1sX2NhZDlhYzVlMjM2NDQzMTQ4YjE2ODdjNjRiODkzZTI0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5PbGQgTWlsbCBTb3V0aCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yM2EyMzdjOWZhMGE0YmQ2YTVlNGQ5MDY3NzJlMjEzYy5zZXRDb250ZW50KGh0bWxfY2FkOWFjNWUyMzY0NDMxNDhiMTY4N2M2NGI4OTNlMjQpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMTM5OTQzYjUyMzFmNGQzMDg4YTM1MWNmNjVkMmI2OWUuYmluZFBvcHVwKHBvcHVwXzIzYTIzN2M5ZmEwYTRiZDZhNWU0ZDkwNjc3MmUyMTNjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FmN2M0YzI3MDUyZDRiZjVhNmIxM2IxYzZlMDI4ODkzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2MjU3OSwtNzkuNDk4NTA5MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81NWE0ZTU2MTcwOTI0Y2U3YmI4ZjEzOTJlNDk3MmQzOCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mZDlkZDZjZjZhMmM0ZWI5YjQ4Zjk3OTM5MmJmMGY4MyA9ICQoJzxkaXYgaWQ9Imh0bWxfZmQ5ZGQ2Y2Y2YTJjNGViOWI0OGY5NzkzOTJiZjBmODMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlRoZSBRdWVlbnN3YXkgRWFzdCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81NWE0ZTU2MTcwOTI0Y2U3YmI4ZjEzOTJlNDk3MmQzOC5zZXRDb250ZW50KGh0bWxfZmQ5ZGQ2Y2Y2YTJjNGViOWI0OGY5NzkzOTJiZjBmODMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWY3YzRjMjcwNTJkNGJmNWE2YjEzYjFjNmUwMjg4OTMuYmluZFBvcHVwKHBvcHVwXzU1YTRlNTYxNzA5MjRjZTdiYjhmMTM5MmU0OTcyZDM4KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2FmYzcxNWMwZTY3NzQ4YjQ4OGIxOWJiZTdkNjcxYzUxID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2MjU3OSwtNzkuNDk4NTA5MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9hZDE4OTY5ODI3ZTM0MWVmYTU4ZDg4OGQ3MmZmZWFiNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85ZTYyNmM2MjMxYjc0MGMyYTc0Yzc3ZDFlN2VlN2RhNyA9ICQoJzxkaXYgaWQ9Imh0bWxfOWU2MjZjNjIzMWI3NDBjMmE3NGM3N2QxZTdlZTdkYTciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJveWFsIFlvcmsgU291dGggRWFzdCwgRXRvYmljb2tlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hZDE4OTY5ODI3ZTM0MWVmYTU4ZDg4OGQ3MmZmZWFiNy5zZXRDb250ZW50KGh0bWxfOWU2MjZjNjIzMWI3NDBjMmE3NGM3N2QxZTdlZTdkYTcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYWZjNzE1YzBlNjc3NDhiNDg4YjE5YmJlN2Q2NzFjNTEuYmluZFBvcHVwKHBvcHVwX2FkMTg5Njk4MjdlMzQxZWZhNThkODg4ZDcyZmZlYWI3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2Q2OWU1Yzk1MjczMDQ1ZDRiNzJiOWQ4YTJhZDBhMjY2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjM2MjU3OSwtNzkuNDk4NTA5MDk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAieWVsbG93IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC44LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF82NWI4NjExMzgxMTA0NjhiYjJhZjEyZWNhNmE1NDczOSk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NjVlMTE0MGU1ODA0N2MzYmIxY2MxNjliNWFlZjgyZiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8zZDNhZDJjMGRjN2M0ODEzOTljNGM0YzlkZmMxY2I0YSA9ICQoJzxkaXYgaWQ9Imh0bWxfM2QzYWQyYzBkYzdjNDgxMzk5YzRjNGM5ZGZjMWNiNGEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN1bm55bGVhLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk2NWUxMTQwZTU4MDQ3YzNiYjFjYzE2OWI1YWVmODJmLnNldENvbnRlbnQoaHRtbF8zZDNhZDJjMGRjN2M0ODEzOTljNGM0YzlkZmMxY2I0YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kNjllNWM5NTI3MzA0NWQ0YjcyYjlkOGEyYWQwYTI2Ni5iaW5kUG9wdXAocG9wdXBfOTY1ZTExNDBlNTgwNDdjM2JiMWNjMTY5YjVhZWY4MmYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNThjOGFjMmFhNzkxNDk3NmI4NWRmN2U4MjNkMWNkZTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg4NDA4LC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzIxZTA2ZmUwNmNjMzQ0YjViMmQwZDc2OTQyYTIyMDJjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA2ZWE0MDY1MDAyMDQwOThhZjA3MTgzNDgwZTVjNTljID0gJCgnPGRpdiBpZD0iaHRtbF8wNmVhNDA2NTAwMjA0MDk4YWYwNzE4MzQ4MGU1YzU5YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZ3N3YXkgUGFyayBTb3V0aCBXZXN0LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzIxZTA2ZmUwNmNjMzQ0YjViMmQwZDc2OTQyYTIyMDJjLnNldENvbnRlbnQoaHRtbF8wNmVhNDA2NTAwMjA0MDk4YWYwNzE4MzQ4MGU1YzU5Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81OGM4YWMyYWE3OTE0OTc2Yjg1ZGY3ZTgyM2QxY2RlNS5iaW5kUG9wdXAocG9wdXBfMjFlMDZmZTA2Y2MzNDRiNWIyZDBkNzY5NDJhMjIwMmMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYzgyNjZkYjIxZWJmNDIzMjgzM2Q5YjMzODA3MjFiZDQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg4NDA4LC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2UyMjJhYzg0YzdhYzQyZTJhOGY0ODJjOWM0ZTY2ZWM5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzcxMzgwYjJiYzg3NzRjZGRhY2IxNDQ4OGViNGJiZDUzID0gJCgnPGRpdiBpZD0iaHRtbF83MTM4MGIyYmM4Nzc0Y2RkYWNiMTQ0ODhlYjRiYmQ1MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+TWltaWNvIE5XLCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2UyMjJhYzg0YzdhYzQyZTJhOGY0ODJjOWM0ZTY2ZWM5LnNldENvbnRlbnQoaHRtbF83MTM4MGIyYmM4Nzc0Y2RkYWNiMTQ0ODhlYjRiYmQ1Myk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9jODI2NmRiMjFlYmY0MjMyODMzZDliMzM4MDcyMWJkNC5iaW5kUG9wdXAocG9wdXBfZTIyMmFjODRjN2FjNDJlMmE4ZjQ4MmM5YzRlNjZlYzkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTc5MzZiY2E5OWE0NDdiMjk0MGJkMDY3MWNlOGNmMTUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg4NDA4LC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzNiZjVkYzY0NWY2YzQwNmY5ODgyYjE0ZGZjNDUyM2ViID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I4OWQ5ZjJmMTRjYTRlYzI5MGMxYzVlYjQ3NWI5YjkxID0gJCgnPGRpdiBpZD0iaHRtbF9iODlkOWYyZjE0Y2E0ZWMyOTBjMWM1ZWI0NzViOWI5MSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VGhlIFF1ZWVuc3dheSBXZXN0LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzNiZjVkYzY0NWY2YzQwNmY5ODgyYjE0ZGZjNDUyM2ViLnNldENvbnRlbnQoaHRtbF9iODlkOWYyZjE0Y2E0ZWMyOTBjMWM1ZWI0NzViOWI5MSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9lNzkzNmJjYTk5YTQ0N2IyOTQwYmQwNjcxY2U4Y2YxNS5iaW5kUG9wdXAocG9wdXBfM2JmNWRjNjQ1ZjZjNDA2Zjk4ODJiMTRkZmM0NTIzZWIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjMzZjAyMTM3MjY0NGY4MmJlNjIyYWJiYzY5MGZlODcgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg4NDA4LC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzgxMmQ1MTcwNmU4NzQ1N2VhNmFhNzI1Y2Q5MTM2ZTU2ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EwNTA3NmZhMmM1ZjRlZjY5MzE5YTdjNTgzYjNkMjA2ID0gJCgnPGRpdiBpZD0iaHRtbF9hMDUwNzZmYTJjNWY0ZWY2OTMxOWE3YzU4M2IzZDIwNiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um95YWwgWW9yayBTb3V0aCBXZXN0LCBFdG9iaWNva2U8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzgxMmQ1MTcwNmU4NzQ1N2VhNmFhNzI1Y2Q5MTM2ZTU2LnNldENvbnRlbnQoaHRtbF9hMDUwNzZmYTJjNWY0ZWY2OTMxOWE3YzU4M2IzZDIwNik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iMzNmMDIxMzcyNjQ0ZjgyYmU2MjJhYmJjNjkwZmU4Ny5iaW5kUG9wdXAocG9wdXBfODEyZDUxNzA2ZTg3NDU3ZWE2YWE3MjVjZDkxMzZlNTYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMDk5Y2FkOWVkYTUzNGJhMmE4MTY5MThiNTFhOGRlNTIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg4NDA4LC03OS41MjA5OTk0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJ5ZWxsb3ciLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjgsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzY1Yjg2MTEzODExMDQ2OGJiMmFmMTJlY2E2YTU0NzM5KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2JjNmEyMjI2ZjlkMTRkZDE5Mzg3NmIyZDliNGI5YzBkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzJkNjVmODg1NmQxMjQwZWZiOTRiYjlkZjRkNDY0MDM5ID0gJCgnPGRpdiBpZD0iaHRtbF8yZDY1Zjg4NTZkMTI0MGVmYjk0YmI5ZGY0ZDQ2NDAzOSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U291dGggb2YgQmxvb3IsIEV0b2JpY29rZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmM2YTIyMjZmOWQxNGRkMTkzODc2YjJkOWI0YjljMGQuc2V0Q29udGVudChodG1sXzJkNjVmODg1NmQxMjQwZWZiOTRiYjlkZjRkNDY0MDM5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA5OWNhZDllZGE1MzRiYTJhODE2OTE4YjUxYThkZTUyLmJpbmRQb3B1cChwb3B1cF9iYzZhMjIyNmY5ZDE0ZGQxOTM4NzZiMmQ5YjRiOWMwZCk7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Simplify the above map and segment and cluster only the neighborhoods in  Downtown Toronto. So let's slice the original dataframe and create a new dataframe of the Downtown Toronto data


```python
df_toronto_new = neighborhoods[neighborhoods['Borough'].str.contains("Downtown Toronto")].reset_index(drop=True)
df_toronto_new.head()
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
      <th>Borough</th>
      <th>Neighborhood</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Downtown Toronto</td>
      <td>Harbourfront</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Downtown Toronto</td>
      <td>Regent Park</td>
      <td>43.654260</td>
      <td>-79.360636</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Downtown Toronto</td>
      <td>Ryerson</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Downtown Toronto</td>
      <td>Garden District</td>
      <td>43.657162</td>
      <td>-79.378937</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Downtown Toronto</td>
      <td>St. James Town</td>
      <td>43.651494</td>
      <td>-79.375418</td>
    </tr>
  </tbody>
</table>
</div>



### Lets Explore Downtown Toronto, Canada
 I will begin by getting the Geo Co-ordinates of Downtown Toronto


```python
address = 'Downtown Toronto, Canada'

geolocator = Nominatim(user_agent="dtCan_explorer")
location = geolocator.geocode(address)
latitude = location.latitude
longitude = location.longitude
print('The geograpical coordinate of Downtown Toronto, Canada are {}, {}.'.format(latitude, longitude))

```

    The geograpical coordinate of Downtown Toronto, Canada are 43.655115, -79.380219.


Let's visualize Downtown Toronto neighbourhoods in the Map


```python
# create map of Downtown Toronto using latitude and longitude values
map_dttoronto = folium.Map(location=[latitude, longitude], zoom_start=11)

# add markers to map
for lat, lng, label in zip(df_toronto_new['Latitude'], df_toronto_new['Longitude'], df_toronto_new['Neighborhood']):
    label = folium.Popup(label, parse_html=True)
    folium.CircleMarker(
        [lat, lng],
        radius=5,
        popup=label,
        color='blue',
        fill=True,
        fill_color='#3186cc',
        fill_opacity=0.7,
        ).add_to(map_dttoronto)  
    
map_dttoronto
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTAgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwIiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjU1MTE1LC03OS4zODAyMTldLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTEsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyXzcxMzExOGIxYjNhZjQzYWVhY2VlZGM5YTNlNjdiZDUxID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80MzRiODJlYTlhZWQ0OTQwYWRkYzk2ZmVkYjUwZGVhNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NDI1OTksLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZWU1NzM4MDY2ZDEwNDVlMzk0MWNhZDBhYjA1ZGI0MmEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjgxNmI1ZjNhMmViNDQwNzlkNGE1NGE3NGQ2OWI3NmYgPSAkKCc8ZGl2IGlkPSJodG1sX2Y4MTZiNWYzYTJlYjQ0MDc5ZDRhNTRhNzRkNjliNzZmIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2VlNTczODA2NmQxMDQ1ZTM5NDFjYWQwYWIwNWRiNDJhLnNldENvbnRlbnQoaHRtbF9mODE2YjVmM2EyZWI0NDA3OWQ0YTU0YTc0ZDY5Yjc2Zik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl80MzRiODJlYTlhZWQ0OTQwYWRkYzk2ZmVkYjUwZGVhNC5iaW5kUG9wdXAocG9wdXBfZWU1NzM4MDY2ZDEwNDVlMzk0MWNhZDBhYjA1ZGI0MmEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2UyNWJjMWFlMWE4NGY3ZGEwZjA2NjliNmQ2MWYxMjggPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTQyNTk5LC03OS4zNjA2MzU5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2MyOWU2NGExMWNhYjRjMTJiMmE2YTgwNmYxMjE4ZjY3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2FhNTE0YjQwODJmYjRlZmRhNjlmZGMzNGJiNjMzMjU5ID0gJCgnPGRpdiBpZD0iaHRtbF9hYTUxNGI0MDgyZmI0ZWZkYTY5ZmRjMzRiYjYzMzI1OSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmVnZW50IFBhcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MyOWU2NGExMWNhYjRjMTJiMmE2YTgwNmYxMjE4ZjY3LnNldENvbnRlbnQoaHRtbF9hYTUxNGI0MDgyZmI0ZWZkYTY5ZmRjMzRiYjYzMzI1OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83ZTI1YmMxYWUxYTg0ZjdkYTBmMDY2OWI2ZDYxZjEyOC5iaW5kUG9wdXAocG9wdXBfYzI5ZTY0YTExY2FiNGMxMmIyYTZhODA2ZjEyMThmNjcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTBiNTE3NTMzM2Y5NDk2ZGE1ODE3NjI2ZTc2Mjk4OGEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LC03OS4zNzg5MzcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jNGUyYWMzNzZhZTM0NjcxYTg2MTc0ZDQ5MzA4MWMxOSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84OTM2ODI2ODM2ZjI0NjA2ODkyNTkzNWU5Njk3ZjY1YiA9ICQoJzxkaXYgaWQ9Imh0bWxfODkzNjgyNjgzNmYyNDYwNjg5MjU5MzVlOTY5N2Y2NWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJ5ZXJzb248L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2M0ZTJhYzM3NmFlMzQ2NzFhODYxNzRkNDkzMDgxYzE5LnNldENvbnRlbnQoaHRtbF84OTM2ODI2ODM2ZjI0NjA2ODkyNTkzNWU5Njk3ZjY1Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hMGI1MTc1MzMzZjk0OTZkYTU4MTc2MjZlNzYyOTg4YS5iaW5kUG9wdXAocG9wdXBfYzRlMmFjMzc2YWUzNDY3MWE4NjE3NGQ0OTMwODFjMTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjFmYmEyOTUyNzMwNDcxMDliMjYzOTdiMGFkZWZiMGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LC03OS4zNzg5MzcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYjg3N2EyNDc5MDA0MTYwOTVlZDRmYzYwMjUwNTJlMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lNDljZjAxM2I2ZWQ0M2UwYTEyNTlhN2M4YTkxNWYwMyA9ICQoJzxkaXYgaWQ9Imh0bWxfZTQ5Y2YwMTNiNmVkNDNlMGExMjU5YTdjOGE5MTVmMDMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhcmRlbiBEaXN0cmljdDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2I4NzdhMjQ3OTAwNDE2MDk1ZWQ0ZmM2MDI1MDUyZTAuc2V0Q29udGVudChodG1sX2U0OWNmMDEzYjZlZDQzZTBhMTI1OWE3YzhhOTE1ZjAzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzYxZmJhMjk1MjczMDQ3MTA5YjI2Mzk3YjBhZGVmYjBkLmJpbmRQb3B1cChwb3B1cF9jYjg3N2EyNDc5MDA0MTYwOTVlZDRmYzYwMjUwNTJlMCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85NTU4OThlMzMzYjg0MjI2YjFjMzhkZTZjZDliMWYwZSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MTQ5MzksLTc5LjM3NTQxNzldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZGFjZWU0NWExMjZmNGY1YjgyYmExMjFlN2MwMmI2YTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYmJjNDIyYzdlYmYwNDk1ZWI1Y2IyYzIyNzE5NmE4MjEgPSAkKCc8ZGl2IGlkPSJodG1sX2JiYzQyMmM3ZWJmMDQ5NWViNWNiMmMyMjcxOTZhODIxIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdC4gSmFtZXMgVG93bjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZGFjZWU0NWExMjZmNGY1YjgyYmExMjFlN2MwMmI2YTMuc2V0Q29udGVudChodG1sX2JiYzQyMmM3ZWJmMDQ5NWViNWNiMmMyMjcxOTZhODIxKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzk1NTg5OGUzMzNiODQyMjZiMWMzOGRlNmNkOWIxZjBlLmJpbmRQb3B1cChwb3B1cF9kYWNlZTQ1YTEyNmY0ZjViODJiYTEyMWU3YzAyYjZhMyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mMGI1YmQ5NDUzNDU0NTJjODI0ZWFmYzljMzFmOWVkMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NDc3MDc5OTk5OTk5NiwtNzkuMzczMzA2NF0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yZmE4ZTQ5N2MyYWU0M2YwYWI2MWMzNGMwYWQ4ZGVkMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jOTEzYzdlMjg3MzE0OTg3OGVmNzUxNzY0OGRjZTEzNyA9ICQoJzxkaXYgaWQ9Imh0bWxfYzkxM2M3ZTI4NzMxNDk4NzhlZjc1MTc2NDhkY2UxMzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkJlcmN6eSBQYXJrPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8yZmE4ZTQ5N2MyYWU0M2YwYWI2MWMzNGMwYWQ4ZGVkMC5zZXRDb250ZW50KGh0bWxfYzkxM2M3ZTI4NzMxNDk4NzhlZjc1MTc2NDhkY2UxMzcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZjBiNWJkOTQ1MzQ1NDUyYzgyNGVhZmM5YzMxZjllZDIuYmluZFBvcHVwKHBvcHVwXzJmYThlNDk3YzJhZTQzZjBhYjYxYzM0YzBhZDhkZWQwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzhmYTE4NzhjMTdjNzQxY2Y5N2MxYmY4MGYyZTgzN2QzID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xMmI4MTc5NDlmMjk0ZjY3YWM3NmFkMGVmMDliOGQ4ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83OWFiY2E0NzNiM2E0NmUzOGE2MzQ3YzVmNjBmNjkzOCA9ICQoJzxkaXYgaWQ9Imh0bWxfNzlhYmNhNDczYjNhNDZlMzhhNjM0N2M1ZjYwZjY5MzgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMTJiODE3OTQ5ZjI5NGY2N2FjNzZhZDBlZjA5YjhkOGQuc2V0Q29udGVudChodG1sXzc5YWJjYTQ3M2IzYTQ2ZTM4YTYzNDdjNWY2MGY2OTM4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzhmYTE4NzhjMTdjNzQxY2Y5N2MxYmY4MGYyZTgzN2QzLmJpbmRQb3B1cChwb3B1cF8xMmI4MTc5NDlmMjk0ZjY3YWM3NmFkMGVmMDliOGQ4ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83MTgwNGNhNzBlNDI0MzM5YTZkMzUzMzkwZWY5OGQ0NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2OTU0MiwtNzkuNDIyNTYzN10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85YTIwZjNiYThiZDY0ZjY2OWQ0NzA5MjQ0YTcxYzU4MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xMGM3ZTRiMmUwYjY0MzUzYTc4YjExODE5MTRiYmI0MiA9ICQoJzxkaXYgaWQ9Imh0bWxfMTBjN2U0YjJlMGI2NDM1M2E3OGIxMTgxOTE0YmJiNDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNocmlzdGllPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85YTIwZjNiYThiZDY0ZjY2OWQ0NzA5MjQ0YTcxYzU4MS5zZXRDb250ZW50KGh0bWxfMTBjN2U0YjJlMGI2NDM1M2E3OGIxMTgxOTE0YmJiNDIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzE4MDRjYTcwZTQyNDMzOWE2ZDM1MzM5MGVmOThkNDUuYmluZFBvcHVwKHBvcHVwXzlhMjBmM2JhOGJkNjRmNjY5ZDQ3MDkyNDRhNzFjNTgxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2NmY2M5NDM1M2I1ZTRkOGNhMTllODQ4NzViZWJiODRiID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUwNTcxMjAwMDAwMDEsLTc5LjM4NDU2NzVdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTAxNDZlOWJmYzRiNDhlYzgyY2ZmMmQ0OTg2MDk4ZTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTBhODcwZmZiMWJhNDFhNDkyZmQ5OTg3YWQxZmE3ZTMgPSAkKCc8ZGl2IGlkPSJodG1sX2EwYTg3MGZmYjFiYTQxYTQ5MmZkOTk4N2FkMWZhN2UzIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5BZGVsYWlkZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNTAxNDZlOWJmYzRiNDhlYzgyY2ZmMmQ0OTg2MDk4ZTYuc2V0Q29udGVudChodG1sX2EwYTg3MGZmYjFiYTQxYTQ5MmZkOTk4N2FkMWZhN2UzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2NmY2M5NDM1M2I1ZTRkOGNhMTllODQ4NzViZWJiODRiLmJpbmRQb3B1cChwb3B1cF81MDE0NmU5YmZjNGI0OGVjODJjZmYyZDQ5ODYwOThlNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zODgwNGU1NDBlODk0MGEzYTFmNjZjZTg1OGRjZWNjMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzkxY2Q0M2RiYzUzMDRkMTFiNmEzYWQ5ZTg5MmUyM2RhID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzNlZmJmNTMyYTdkZDQ1MWU5NmEzMmIxYjY4MjE5ZWQyID0gJCgnPGRpdiBpZD0iaHRtbF8zZWZiZjUzMmE3ZGQ0NTFlOTZhMzJiMWI2ODIxOWVkMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTFjZDQzZGJjNTMwNGQxMWI2YTNhZDllODkyZTIzZGEuc2V0Q29udGVudChodG1sXzNlZmJmNTMyYTdkZDQ1MWU5NmEzMmIxYjY4MjE5ZWQyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzM4ODA0ZTU0MGU4OTQwYTNhMWY2NmNlODU4ZGNlY2MxLmJpbmRQb3B1cChwb3B1cF85MWNkNDNkYmM1MzA0ZDExYjZhM2FkOWU4OTJlMjNkYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hYTI0NDVkMGQxZjk0YTIyOTA0NDM1OTQ4ZmM5YzFkNSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzYxY2RkMDE3OWJlNzQ0OGZhYWFhMjczODE3YjEyZGQxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I0N2NiNzFjNGMyYTQ2ZWM5OTI0MDFiZmY0NmMwMjAzID0gJCgnPGRpdiBpZD0iaHRtbF9iNDdjYjcxYzRjMmE0NmVjOTkyNDAxYmZmNDZjMDIwMyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UmljaG1vbmQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzYxY2RkMDE3OWJlNzQ0OGZhYWFhMjczODE3YjEyZGQxLnNldENvbnRlbnQoaHRtbF9iNDdjYjcxYzRjMmE0NmVjOTkyNDAxYmZmNDZjMDIwMyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9hYTI0NDVkMGQxZjk0YTIyOTA0NDM1OTQ4ZmM5YzFkNS5iaW5kUG9wdXAocG9wdXBfNjFjZGQwMTc5YmU3NDQ4ZmFhYWEyNzM4MTdiMTJkZDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGFiYjg4MmU1ZTQ1NDVjYjk2OGE4MTdkMzZiY2IwYmUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NGRkOGVkNzQ0ZjM0NzFjOTEzZTNiYjc3Yjg5YmZkZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF82ZWQ3OTNjMWNmZjE0ODMwOGE0NGZlZGUzY2U0N2IzMSA9ICQoJzxkaXYgaWQ9Imh0bWxfNmVkNzkzYzFjZmYxNDgzMDhhNDRmZWRlM2NlNDdiMzEiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCBFYXN0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85NGRkOGVkNzQ0ZjM0NzFjOTEzZTNiYjc3Yjg5YmZkZC5zZXRDb250ZW50KGh0bWxfNmVkNzkzYzFjZmYxNDgzMDhhNDRmZWRlM2NlNDdiMzEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGFiYjg4MmU1ZTQ1NDVjYjk2OGE4MTdkMzZiY2IwYmUuYmluZFBvcHVwKHBvcHVwXzk0ZGQ4ZWQ3NDRmMzQ3MWM5MTNlM2JiNzdiODliZmRkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzJhZDEwOTIxZDVkYTQ1ZDdhMDQzMTc5NGMzYWRhNWZkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTE4ODdhMThlYjFkNGFjMjgwODg1OGM1OTM1MjExNjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNzAzYzU4N2Q5YzljNGE5NmI3M2M2ZTAzNjQ4NDI5ZDQgPSAkKCc8ZGl2IGlkPSJodG1sXzcwM2M1ODdkOWM5YzRhOTZiNzNjNmUwMzY0ODQyOWQ0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ub3JvbnRvIElzbGFuZHM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzkxODg3YTE4ZWIxZDRhYzI4MDg4NThjNTkzNTIxMTY2LnNldENvbnRlbnQoaHRtbF83MDNjNTg3ZDljOWM0YTk2YjczYzZlMDM2NDg0MjlkNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8yYWQxMDkyMWQ1ZGE0NWQ3YTA0MzE3OTRjM2FkYTVmZC5iaW5kUG9wdXAocG9wdXBfOTE4ODdhMThlYjFkNGFjMjgwODg1OGM1OTM1MjExNjYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMWJjMGM0N2JiNjM3NGZjNGIxMjIyYTcxNjBiNjhhMmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDA4MTU3LC03OS4zODE3NTIyOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83Y2M4MDE3NjY2Y2Q0YTk5YTdiMDk4MTYyZTE4Yzk4MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MmEyMTg0YTZiZmQ0YWMwYjM2ODRiYjk3NDI0YjQzNSA9ICQoJzxkaXYgaWQ9Imh0bWxfNTJhMjE4NGE2YmZkNGFjMGIzNjg0YmI5NzQyNGI0MzUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaW9uIFN0YXRpb248L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdjYzgwMTc2NjZjZDRhOTlhN2IwOTgxNjJlMThjOTgxLnNldENvbnRlbnQoaHRtbF81MmEyMTg0YTZiZmQ0YWMwYjM2ODRiYjk3NDI0YjQzNSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8xYmMwYzQ3YmI2Mzc0ZmM0YjEyMjJhNzE2MGI2OGEyZi5iaW5kUG9wdXAocG9wdXBfN2NjODAxNzY2NmNkNGE5OWE3YjA5ODE2MmUxOGM5ODEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMTAwZWQ2MmNiMTllNGRjZTk0NDY4YjdjYTNmZWRjOTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDcxNzY4LC03OS4zODE1NzY0MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83MmRjNmJhMmQwNGE0NjFjOTUyMDViNzZmY2Y3MmZjNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xNzM4NmUxYzNhN2E0ODdjOWMzMTJjN2I3ODkwMWQ5ZCA9ICQoJzxkaXYgaWQ9Imh0bWxfMTczODZlMWMzYTdhNDg3YzljMzEyYzdiNzg5MDFkOWQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkRlc2lnbiBFeGNoYW5nZTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzJkYzZiYTJkMDRhNDYxYzk1MjA1Yjc2ZmNmNzJmYzcuc2V0Q29udGVudChodG1sXzE3Mzg2ZTFjM2E3YTQ4N2M5YzMxMmM3Yjc4OTAxZDlkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzEwMGVkNjJjYjE5ZTRkY2U5NDQ2OGI3Y2EzZmVkYzk5LmJpbmRQb3B1cChwb3B1cF83MmRjNmJhMmQwNGE0NjFjOTUyMDViNzZmY2Y3MmZjNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl83MmUwNjkzNTI2NmI0Njg3YmY0OTZhN2FkODNjNDkwMCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzJhZTA5MWM3ZDFjNTRkYzk4NDdkOTEzZWEwN2IyOWUyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2RjZDg0NTU5MDEyNDQ5YjhiMDU0YmMyODE5NTlkN2RiID0gJCgnPGRpdiBpZD0iaHRtbF9kY2Q4NDU1OTAxMjQ0OWI4YjA1NGJjMjgxOTU5ZDdkYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBEb21pbmlvbiBDZW50cmU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzJhZTA5MWM3ZDFjNTRkYzk4NDdkOTEzZWEwN2IyOWUyLnNldENvbnRlbnQoaHRtbF9kY2Q4NDU1OTAxMjQ0OWI4YjA1NGJjMjgxOTU5ZDdkYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83MmUwNjkzNTI2NmI0Njg3YmY0OTZhN2FkODNjNDkwMC5iaW5kUG9wdXAocG9wdXBfMmFlMDkxYzdkMWM1NGRjOTg0N2Q5MTNlYTA3YjI5ZTIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTIzZDU5MGFlYzhmNDM2M2EyOWNiNGI2YjM4ODU5MGIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xYWI1YWI0OWVmNDI0Yjc3OWUzNzUyYmFmNDk5NmFjZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9hNzQxMjVlZGNmZGY0MjQ4YjQ4ODcyMDg2ZmNhZDZiYyA9ICQoJzxkaXYgaWQ9Imh0bWxfYTc0MTI1ZWRjZmRmNDI0OGI0ODg3MjA4NmZjYWQ2YmMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNvbW1lcmNlIENvdXJ0PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xYWI1YWI0OWVmNDI0Yjc3OWUzNzUyYmFmNDk5NmFjZC5zZXRDb250ZW50KGh0bWxfYTc0MTI1ZWRjZmRmNDI0OGI0ODg3MjA4NmZjYWQ2YmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYTIzZDU5MGFlYzhmNDM2M2EyOWNiNGI2YjM4ODU5MGIuYmluZFBvcHVwKHBvcHVwXzFhYjVhYjQ5ZWY0MjRiNzc5ZTM3NTJiYWY0OTk2YWNkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzUzYTRiOTRlMjk2MjRlZTc5MzA4MTgwNzNlYmI2MDFmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4MTk4NSwtNzkuMzc5ODE2OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWNhZDQ0OTk3ZWVlNDY2YTgyMTAxNTcxMDhjMTQwY2QgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOWUwN2I1MjQwZTEwNDIxNGIyOTM4MzA3MmE3YTQyNGMgPSAkKCc8ZGl2IGlkPSJodG1sXzllMDdiNTI0MGUxMDQyMTRiMjkzODMwNzJhN2E0MjRjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5WaWN0b3JpYSBIb3RlbDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWNhZDQ0OTk3ZWVlNDY2YTgyMTAxNTcxMDhjMTQwY2Quc2V0Q29udGVudChodG1sXzllMDdiNTI0MGUxMDQyMTRiMjkzODMwNzJhN2E0MjRjKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzUzYTRiOTRlMjk2MjRlZTc5MzA4MTgwNzNlYmI2MDFmLmJpbmRQb3B1cChwb3B1cF8xY2FkNDQ5OTdlZWU0NjZhODIxMDE1NzEwOGMxNDBjZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8xY2ExN2Y4ZGUzMDA0MTIzYjM4ZDlhZmI5YWFkNTBjNyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYTYxZmNmYTNjMTRjNDY2ZmFlZmIxNDVlYzY3MTNlYmMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDYyNzc4NDY1YjNkNGVhYWI5MzIyOGU0ZGFlZDkzNzUgPSAkKCc8ZGl2IGlkPSJodG1sX2Q2Mjc3ODQ2NWIzZDRlYWFiOTMyMjhlNGRhZWQ5Mzc1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3JkPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9hNjFmY2ZhM2MxNGM0NjZmYWVmYjE0NWVjNjcxM2ViYy5zZXRDb250ZW50KGh0bWxfZDYyNzc4NDY1YjNkNGVhYWI5MzIyOGU0ZGFlZDkzNzUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMWNhMTdmOGRlMzAwNDEyM2IzOGQ5YWZiOWFhZDUwYzcuYmluZFBvcHVwKHBvcHVwX2E2MWZjZmEzYzE0YzQ2NmZhZWZiMTQ1ZWM2NzEzZWJjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzNkZGFhZjRlM2ZlYTRjYWRhZThlMzcxMjdkN2I4ZWJkID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjYyNjk1NiwtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8xY2YxMjY4Nzg4YTE0MDM5OTU3ZWVlZTYyMjVhODMzNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NDg3MTQyMWE0NGE0MDc1OTczYTFhNDQwNjA4ZjA3YSA9ICQoJzxkaXYgaWQ9Imh0bWxfODQ4NzE0MjFhNDRhNDA3NTk3M2ExYTQ0MDYwOGYwN2EiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuaXZlcnNpdHkgb2YgVG9yb250bzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMWNmMTI2ODc4OGExNDAzOTk1N2VlZWU2MjI1YTgzMzcuc2V0Q29udGVudChodG1sXzg0ODcxNDIxYTQ0YTQwNzU5NzNhMWE0NDA2MDhmMDdhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNkZGFhZjRlM2ZlYTRjYWRhZThlMzcxMjdkN2I4ZWJkLmJpbmRQb3B1cChwb3B1cF8xY2YxMjY4Nzg4YTE0MDM5OTU3ZWVlZTYyMjVhODMzNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wYWY1ZTFjMTE3NDY0NjI1YWIxZTdkN2ZlYjNhMjgxZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MzIwNTcsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZDY4MTQ0ZTkxNmM2NDI2N2JkNjBkMmIwNWQ0ZTZlOTMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYjcxMjcwMmZkZDE4NDRkYWIzZWRkNTllODI3ZTgwNjYgPSAkKCc8ZGl2IGlkPSJodG1sX2I3MTI3MDJmZGQxODQ0ZGFiM2VkZDU5ZTgyN2U4MDY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaGluYXRvd248L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2Q2ODE0NGU5MTZjNjQyNjdiZDYwZDJiMDVkNGU2ZTkzLnNldENvbnRlbnQoaHRtbF9iNzEyNzAyZmRkMTg0NGRhYjNlZGQ1OWU4MjdlODA2Nik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wYWY1ZTFjMTE3NDY0NjI1YWIxZTdkN2ZlYjNhMjgxZi5iaW5kUG9wdXAocG9wdXBfZDY4MTQ0ZTkxNmM2NDI2N2JkNjBkMmIwNWQ0ZTZlOTMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDJhODhmMzExNjkyNDk1ZWI4M2ExMTgwZDhhNmI2OTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzllNjQ4MjMwYzdjMzRkZTliOGI5NTE2NzFmYTQzMDAxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzA5NzMxMmNkNDYxZTQyYzk5NDE0YzZhMzAwNmUxOTlkID0gJCgnPGRpdiBpZD0iaHRtbF8wOTczMTJjZDQ2MWU0MmM5OTQxNGM2YTMwMDZlMTk5ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R3JhbmdlIFBhcms8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzllNjQ4MjMwYzdjMzRkZTliOGI5NTE2NzFmYTQzMDAxLnNldENvbnRlbnQoaHRtbF8wOTczMTJjZDQ2MWU0MmM5OTQxNGM2YTMwMDZlMTk5ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMmE4OGYzMTE2OTI0OTVlYjgzYTExODBkOGE2YjY5OS5iaW5kUG9wdXAocG9wdXBfOWU2NDgyMzBjN2MzNGRlOWI4Yjk1MTY3MWZhNDMwMDEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfN2E1MTJiZDE5ZTE3NDg3N2E4ZGFkYWUwMDBmMWEyYmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RmNDI0ZmUxODI1OTRiMjQ4OTVjMTMxZTVjNWM3NzA3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2ZhOTJiZjRkMzdmNzQwYjE5OWFiNGFmODAzZTg5NTBkID0gJCgnPGRpdiBpZD0iaHRtbF9mYTkyYmY0ZDM3Zjc0MGIxOTlhYjRhZjgwM2U4OTUwZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2Vuc2luZ3RvbiBNYXJrZXQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RmNDI0ZmUxODI1OTRiMjQ4OTVjMTMxZTVjNWM3NzA3LnNldENvbnRlbnQoaHRtbF9mYTkyYmY0ZDM3Zjc0MGIxOTlhYjRhZjgwM2U4OTUwZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl83YTUxMmJkMTllMTc0ODc3YThkYWRhZTAwMGYxYTJiYy5iaW5kUG9wdXAocG9wdXBfZGY0MjRmZTE4MjU5NGIyNDg5NWMxMzFlNWM1Yzc3MDcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNTU1MGE3MTIyNjcwNDViNzlmMDNjNjk4YjkzMDRhYWIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzlmMGQ1MmZhOTU4ZDQ3YTNiZjdhZTM4Y2ZhODM3ZmU5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzhlODE5MWVhNGFlZTQ3NGNhOGUxMzI3YTE2YzZkNTMyID0gJCgnPGRpdiBpZD0iaHRtbF84ZTgxOTFlYTRhZWU0NzRjYThlMTMyN2ExNmM2ZDUzMiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Q04gVG93ZXI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzlmMGQ1MmZhOTU4ZDQ3YTNiZjdhZTM4Y2ZhODM3ZmU5LnNldENvbnRlbnQoaHRtbF84ZTgxOTFlYTRhZWU0NzRjYThlMTMyN2ExNmM2ZDUzMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl81NTUwYTcxMjI2NzA0NWI3OWYwM2M2OThiOTMwNGFhYi5iaW5kUG9wdXAocG9wdXBfOWYwZDUyZmE5NThkNDdhM2JmN2FlMzhjZmE4MzdmZTkpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTM5NzdmNzdkYWYwNDYwZmJjNGQyNjg4MDJjZjNhNzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NjNjNlMDhkNWJmODQ4NWRiZmE4ODIwNDdjMTcxZGNkID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2EzNmIzN2E5NGNkYzRlYjdhNjNiMGY0MzRmYjUyM2RhID0gJCgnPGRpdiBpZD0iaHRtbF9hMzZiMzdhOTRjZGM0ZWI3YTYzYjBmNDM0ZmI1MjNkYSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgUXVheTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2M2M2UwOGQ1YmY4NDg1ZGJmYTg4MjA0N2MxNzFkY2Quc2V0Q29udGVudChodG1sX2EzNmIzN2E5NGNkYzRlYjdhNjNiMGY0MzRmYjUyM2RhKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzkzOTc3Zjc3ZGFmMDQ2MGZiYzRkMjY4ODAyY2YzYTc1LmJpbmRQb3B1cChwb3B1cF9jYzYzZTA4ZDViZjg0ODVkYmZhODgyMDQ3YzE3MWRjZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wOTIxNTZlOTNmZmI0NDllODJiOGM1NDcxM2I1NGRlNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMDJlYjRlMDZkNzEwNDI1MzljMGQ0ZmQ3YmM1MzAzMTYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzI0MTg1MmFmZGFjNGRhNzgyNGFiYTVjYjhmODA0YTUgPSAkKCc8ZGl2IGlkPSJodG1sXzMyNDE4NTJhZmRhYzRkYTc4MjRhYmE1Y2I4ZjgwNGE1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Jc2xhbmQgYWlycG9ydDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMDJlYjRlMDZkNzEwNDI1MzljMGQ0ZmQ3YmM1MzAzMTYuc2V0Q29udGVudChodG1sXzMyNDE4NTJhZmRhYzRkYTc4MjRhYmE1Y2I4ZjgwNGE1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzA5MjE1NmU5M2ZmYjQ0OWU4MmI4YzU0NzEzYjU0ZGU0LmJpbmRQb3B1cChwb3B1cF8wMmViNGUwNmQ3MTA0MjUzOWMwZDRmZDdiYzUzMDMxNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNWEyZWQxMzJmMmU0N2M4OTliNTg0YzY1NDNjNmMxZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTMyMmMyYjk4MGI5NGUyZDk5MWMzZWZhNDVhYjdlOTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzUyZTkxNDEyNGE4NGQ0MDk2ZGU2ZjA0NmYxZWRiZWQgPSAkKCc8ZGl2IGlkPSJodG1sXzM1MmU5MTQxMjRhODRkNDA5NmRlNmYwNDZmMWVkYmVkIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgV2VzdDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTMyMmMyYjk4MGI5NGUyZDk5MWMzZWZhNDVhYjdlOTAuc2V0Q29udGVudChodG1sXzM1MmU5MTQxMjRhODRkNDA5NmRlNmYwNDZmMWVkYmVkKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2M1YTJlZDEzMmYyZTQ3Yzg5OWI1ODRjNjU0M2M2YzFmLmJpbmRQb3B1cChwb3B1cF85MzIyYzJiOTgwYjk0ZTJkOTkxYzNlZmE0NWFiN2U5MCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zY2ExYjcwMGEzMzE0ZGI3YmU3ZWE1ZDRhNTYzZmE5ZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNTljNzQ4NTk3ZWRjNDBmYWFmOGVlNTYyOTVkMWRjZDEgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNmM2ZGZlNDk3ODU5NGI5Mjk5ZmU3NGZhZmRjYWQ5YjUgPSAkKCc8ZGl2IGlkPSJodG1sXzZjNmRmZTQ5Nzg1OTRiOTI5OWZlNzRmYWZkY2FkOWI1IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nIGFuZCBTcGFkaW5hPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81OWM3NDg1OTdlZGM0MGZhYWY4ZWU1NjI5NWQxZGNkMS5zZXRDb250ZW50KGh0bWxfNmM2ZGZlNDk3ODU5NGI5Mjk5ZmU3NGZhZmRjYWQ5YjUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2NhMWI3MDBhMzMxNGRiN2JlN2VhNWQ0YTU2M2ZhOWYuYmluZFBvcHVwKHBvcHVwXzU5Yzc0ODU5N2VkYzQwZmFhZjhlZTU2Mjk1ZDFkY2QxKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZhZDhkZGE3NDNhNjQ5OWJiOWZlMmNkZDcwMmFjNjM4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81ODQ5ZTQ5ZjYxYTk0ZmE3ODE4YWVmZTUyMWRhMmIzNiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84ODI5OGExMDJjMjI0NTQzODc1ODQ3NjU3ODM0Y2JmMCA9ICQoJzxkaXYgaWQ9Imh0bWxfODgyOThhMTAyYzIyNDU0Mzg3NTg0NzY1NzgzNGNiZjAiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJhaWx3YXkgTGFuZHM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzU4NDllNDlmNjFhOTRmYTc4MThhZWZlNTIxZGEyYjM2LnNldENvbnRlbnQoaHRtbF84ODI5OGExMDJjMjI0NTQzODc1ODQ3NjU3ODM0Y2JmMCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82YWQ4ZGRhNzQzYTY0OTliYjlmZTJjZGQ3MDJhYzYzOC5iaW5kUG9wdXAocG9wdXBfNTg0OWU0OWY2MWE5NGZhNzgxOGFlZmU1MjFkYTJiMzYpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfODkxZjU5MmEzOWE4NGZhNzkxODlkZGE5ZTZkOTM1NzUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY2MzU0ZjdiNDdmZTRlZDZhYjgzNzgxNDkxNDRlODc3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2JiYzU5YTIwYmMxMTQ2YzdhMmM4ZDU1OWFhMjQwMjQzID0gJCgnPGRpdiBpZD0iaHRtbF9iYmM1OWEyMGJjMTE0NmM3YTJjOGQ1NTlhYTI0MDI0MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U291dGggTmlhZ2FyYTwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNjYzNTRmN2I0N2ZlNGVkNmFiODM3ODE0OTE0NGU4Nzcuc2V0Q29udGVudChodG1sX2JiYzU5YTIwYmMxMTQ2YzdhMmM4ZDU1OWFhMjQwMjQzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzg5MWY1OTJhMzlhODRmYTc5MTg5ZGRhOWU2ZDkzNTc1LmJpbmRQb3B1cChwb3B1cF82NjM1NGY3YjQ3ZmU0ZWQ2YWI4Mzc4MTQ5MTQ0ZTg3Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zMzc0MmE2NGMzMDc0MDA3YjAwZjYyMDBjMjQxNDc2MyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsLTc5LjM3NzUyOTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogImJsdWUiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMzE4NmNjIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwX2VhZTNhYzEyMGJiMTRkM2JiMGNiZTNlMTFmNDUxM2UwKTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzk1ODE0MTZkOWI1NzRhNjc4ZWJmZWIwM2YzMTZlZmQyID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc2ZTM4MTU2ZmNiOTRkZmE5MTFjZDY5Y2E2NjJmNGE0ID0gJCgnPGRpdiBpZD0iaHRtbF83NmUzODE1NmZjYjk0ZGZhOTExY2Q2OWNhNjYyZjRhNCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzk1ODE0MTZkOWI1NzRhNjc4ZWJmZWIwM2YzMTZlZmQyLnNldENvbnRlbnQoaHRtbF83NmUzODE1NmZjYjk0ZGZhOTExY2Q2OWNhNjYyZjRhNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zMzc0MmE2NGMzMDc0MDA3YjAwZjYyMDBjMjQxNDc2My5iaW5kUG9wdXAocG9wdXBfOTU4MTQxNmQ5YjU3NGE2NzhlYmZlYjAzZjMxNmVmZDIpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOWU0NjhiYWYwYjY0NGY5MWFjZmQzNTkyNDkyOTJkMmEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDY0MzUyLC03OS4zNzQ4NDU5OTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yOTI5OWQ4ZDdlMWQ0ZTdjOTgyOWFiNmM3Njk4Y2Q4NyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9jNGM1ZWJkMTEzZTA0NjJiYjkxYjNjZDdiZjc3NjFiZSA9ICQoJzxkaXYgaWQ9Imh0bWxfYzRjNWViZDExM2UwNDYyYmI5MWIzY2Q3YmY3NzYxYmUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0biBBIFBPIEJveGVzIDI1IFRoZSBFc3BsYW5hZGU8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzI5Mjk5ZDhkN2UxZDRlN2M5ODI5YWI2Yzc2OThjZDg3LnNldENvbnRlbnQoaHRtbF9jNGM1ZWJkMTEzZTA0NjJiYjkxYjNjZDdiZjc3NjFiZSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85ZTQ2OGJhZjBiNjQ0ZjkxYWNmZDM1OTI0OTI5MmQyYS5iaW5kUG9wdXAocG9wdXBfMjkyOTlkOGQ3ZTFkNGU3Yzk4MjlhYjZjNzY5OGNkODcpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYTg3MDFmN2VmNGM2NGU0YmEwZDZhYThmYWExYTY1ZDkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njc5NjcsLTc5LjM2NzY3NTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiYmx1ZSIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMzMTg2Y2MiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfZWFlM2FjMTIwYmIxNGQzYmIwY2JlM2UxMWY0NTEzZTApOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYmRhMzA2ZDIzNWRlNDFiNGJjZTQ2ODM1NmNiNjUxNTcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTEzMGZhZDFjMTY3NDQxMzk4NDY2MDJlYjY4NDBiYmIgPSAkKCc8ZGl2IGlkPSJodG1sX2ExMzBmYWQxYzE2NzQ0MTM5ODQ2NjAyZWI2ODQwYmJiIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DYWJiYWdldG93bjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYmRhMzA2ZDIzNWRlNDFiNGJjZTQ2ODM1NmNiNjUxNTcuc2V0Q29udGVudChodG1sX2ExMzBmYWQxYzE2NzQ0MTM5ODQ2NjAyZWI2ODQwYmJiKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2E4NzAxZjdlZjRjNjRlNGJhMGQ2YWE4ZmFhMWE2NWQ5LmJpbmRQb3B1cChwb3B1cF9iZGEzMDZkMjM1ZGU0MWI0YmNlNDY4MzU2Y2I2NTE1Nyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9jNGE0ZGI4NDBhNmI0MGE5YmViYTUwNWYxMTQ2ZmNhMiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85NTFiYmU1OTg4ZjU0OGU1ODlhYTU3YTkwNWU1NTcxYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8wZjA4ZTc3NDYxMDk0OGZlYTVlZjE2ODRiYzA0YjlhNSA9ICQoJzxkaXYgaWQ9Imh0bWxfMGYwOGU3NzQ2MTA5NDhmZWE1ZWYxNjg0YmMwNGI5YTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85NTFiYmU1OTg4ZjU0OGU1ODlhYTU3YTkwNWU1NTcxYS5zZXRDb250ZW50KGh0bWxfMGYwOGU3NzQ2MTA5NDhmZWE1ZWYxNjg0YmMwNGI5YTUpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYzRhNGRiODQwYTZiNDBhOWJlYmE1MDVmMTE0NmZjYTIuYmluZFBvcHVwKHBvcHVwXzk1MWJiZTU5ODhmNTQ4ZTU4OWFhNTdhOTA1ZTU1NzFhKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzcwN2Y1NTA1MTc4NjQ4NTNhMmQxNWZmNmRiNmU2OTBmID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9lMWY0Y2RjMWU0Mjc0OTdkYTA4NjYwYmM0NjkxNmQzMCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yMmMzZjc3M2ZjMzc0YTczOGU4NzJjNGYzYTExZjMzNyA9ICQoJzxkaXYgaWQ9Imh0bWxfMjJjM2Y3NzNmYzM3NGE3MzhlODcyYzRmM2ExMWYzMzciIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkZpcnN0IENhbmFkaWFuIFBsYWNlPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9lMWY0Y2RjMWU0Mjc0OTdkYTA4NjYwYmM0NjkxNmQzMC5zZXRDb250ZW50KGh0bWxfMjJjM2Y3NzNmYzM3NGE3MzhlODcyYzRmM2ExMWYzMzcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNzA3ZjU1MDUxNzg2NDg1M2EyZDE1ZmY2ZGI2ZTY5MGYuYmluZFBvcHVwKHBvcHVwX2UxZjRjZGMxZTQyNzQ5N2RhMDg2NjBiYzQ2OTE2ZDMwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzg2ZDUzNTRiYWI4YjQ4YWE5ZDk3ZWRkMzRhN2Q0MmZhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4NDI5MiwtNzkuMzgyMjgwMl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8zMjZmMjcwYjEzYWU0MmIzYjcwM2EwNzVhNDg0Njg3MSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZjc4NWI5NjJlM2Y0MGRlOWYyYjBmZmQ3MTUxOTgxOCA9ICQoJzxkaXYgaWQ9Imh0bWxfMmY3ODViOTYyZTNmNDBkZTlmMmIwZmZkNzE1MTk4MTgiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlVuZGVyZ3JvdW5kIGNpdHk8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzMyNmYyNzBiMTNhZTQyYjNiNzAzYTA3NWE0ODQ2ODcxLnNldENvbnRlbnQoaHRtbF8yZjc4NWI5NjJlM2Y0MGRlOWYyYjBmZmQ3MTUxOTgxOCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84NmQ1MzU0YmFiOGI0OGFhOWQ5N2VkZDM0YTdkNDJmYS5iaW5kUG9wdXAocG9wdXBfMzI2ZjI3MGIxM2FlNDJiM2I3MDNhMDc1YTQ4NDY4NzEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZTBmYzczYWNmMjg0NDMwMjlhZDY2ODY4ZDZhZTA1NWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICJibHVlIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzMxODZjYyIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF9lYWUzYWMxMjBiYjE0ZDNiYjBjYmUzZTExZjQ1MTNlMCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMTA3NjZhYjgxNzY0YmQ1YWM0NGQ5OWEwMzMwYzhlMiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF85YzE0Yzk5NTViNmU0MGUzOTkzOTJmMGYxYTk5YTM0YiA9ICQoJzxkaXYgaWQ9Imh0bWxfOWMxNGM5OTU1YjZlNDBlMzk5MzkyZjBmMWE5OWEzNGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5PC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9iMTA3NjZhYjgxNzY0YmQ1YWM0NGQ5OWEwMzMwYzhlMi5zZXRDb250ZW50KGh0bWxfOWMxNGM5OTU1YjZlNDBlMzk5MzkyZjBmMWE5OWEzNGIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTBmYzczYWNmMjg0NDMwMjlhZDY2ODY4ZDZhZTA1NWUuYmluZFBvcHVwKHBvcHVwX2IxMDc2NmFiODE3NjRiZDVhYzQ0ZDk5YTAzMzBjOGUyKTsKCiAgICAgICAgICAgIAogICAgICAgIAo8L3NjcmlwdD4=" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



### Define Foursquare Credentials and Version


```python
CLIENT_ID = 'NLXCKQ23MN2GCTBG4TRVXSFXYTVUPRPS120IIZIGNQ4CIYTP' # your Foursquare ID
CLIENT_SECRET = 'LPDQZSSGUP3DSYQ4X3VKLKFZMYTIMCZ1MJFM1K43XMLS4TJK' # your Foursquare Secret
VERSION = '20180605' # Foursquare API version

print('Your credentails:')
print('CLIENT_ID: ' + CLIENT_ID)
print('CLIENT_SECRET:' + CLIENT_SECRET)
```

    Your credentails:
    CLIENT_ID: NLXCKQ23MN2GCTBG4TRVXSFXYTVUPRPS120IIZIGNQ4CIYTP
    CLIENT_SECRET:LPDQZSSGUP3DSYQ4X3VKLKFZMYTIMCZ1MJFM1K43XMLS4TJK


#### Get the neighborhood's name


```python
df_toronto_new.loc[0, 'Neighborhood']
```




    'Harbourfront'



#### Get the neighborhood's latitude and longitude values


```python
neighborhood_latitude = df_toronto_new.loc[0, 'Latitude'] # neighborhood latitude value
neighborhood_longitude = df_toronto_new.loc[0, 'Longitude'] # neighborhood longitude value

neighborhood_name = df_toronto_new.loc[0, 'Neighborhood'] # neighborhood name

print('Latitude and longitude values of {} are {}, {}.'.format(neighborhood_name, 
                                                               neighborhood_latitude, 
                                                               neighborhood_longitude))
```

    Latitude and longitude values of Harbourfront are 43.6542599, -79.3606359.


### Now, let's get the top 100 venues that are in Harbourfront within a radius of 500 meters

Create the GET request URL


```python
LIMIT = 100 # limit of number of venues returned by Foursquare API
radius = 500 # define radius

# create URL
url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
    CLIENT_ID, 
    CLIENT_SECRET, 
    VERSION, 
    neighborhood_latitude, 
    neighborhood_longitude, 
    radius, 
    LIMIT)
url # display URL

```




    'https://api.foursquare.com/v2/venues/explore?&client_id=NLXCKQ23MN2GCTBG4TRVXSFXYTVUPRPS120IIZIGNQ4CIYTP&client_secret=LPDQZSSGUP3DSYQ4X3VKLKFZMYTIMCZ1MJFM1K43XMLS4TJK&v=20180605&ll=43.6542599,-79.3606359&radius=500&limit=100'



### Sending the GET request and examine the Results


```python
results = requests.get(url).json()
results
```




    {'meta': {'code': 200, 'requestId': '5c4c8aec351e3d1df9988979'},
     'response': {'suggestedFilters': {'header': 'Tap to show:',
       'filters': [{'name': 'Open now', 'key': 'openNow'}]},
      'headerLocation': 'Corktown',
      'headerFullLocation': 'Corktown, Toronto',
      'headerLocationGranularity': 'neighborhood',
      'totalResults': 48,
      'suggestedBounds': {'ne': {'lat': 43.6587599045, 'lng': -79.3544279001486},
       'sw': {'lat': 43.6497598955, 'lng': -79.36684389985142}},
      'groups': [{'type': 'Recommended Places',
        'name': 'recommended',
        'items': [{'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '54ea41ad498e9a11e9e13308',
           'name': 'Roselle Desserts',
           'location': {'address': '362 King St E',
            'crossStreet': 'Trinity St',
            'lat': 43.653446723052674,
            'lng': -79.3620167174383,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.653446723052674,
              'lng': -79.3620167174383}],
            'distance': 143,
            'postalCode': 'M5A 1K9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['362 King St E (Trinity St)',
             'Toronto ON M5A 1K9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16a941735',
             'name': 'Bakery',
             'pluralName': 'Bakeries',
             'shortName': 'Bakery',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/bakery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-54ea41ad498e9a11e9e13308-0'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '53b8466a498e83df908c3f21',
           'name': 'Tandem Coffee',
           'location': {'address': '368 King St E',
            'crossStreet': 'at Trinity St',
            'lat': 43.65355870959944,
            'lng': -79.36180945913513,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65355870959944,
              'lng': -79.36180945913513}],
            'distance': 122,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['368 King St E (at Trinity St)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-53b8466a498e83df908c3f21-1'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '574c229e498ebb5c6b257902',
           'name': 'Toronto Cooper Koo Family Cherry St YMCA Centre',
           'location': {'address': '461 Cherry Street',
            'lat': 43.65319052672638,
            'lng': -79.35794700053884,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65319052672638,
              'lng': -79.35794700053884}],
            'distance': 247,
            'postalCode': 'M5A 1H1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['461 Cherry Street',
             'Toronto ON M5A 1H1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d175941735',
             'name': 'Gym / Fitness Center',
             'pluralName': 'Gyms or Fitness Centers',
             'shortName': 'Gym / Fitness',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/building/gym_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-574c229e498ebb5c6b257902-2'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ae5b91ff964a520a6a121e3',
           'name': 'Morning Glory Cafe',
           'location': {'address': '457 King St. E',
            'crossStreet': 'Gilead Place',
            'lat': 43.653946942635294,
            'lng': -79.36114884214422,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.653946942635294,
              'lng': -79.36114884214422}],
            'distance': 54,
            'postalCode': 'M5A 1L6',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['457 King St. E (Gilead Place)',
             'Toronto ON M5A 1L6',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d143941735',
             'name': 'Breakfast Spot',
             'pluralName': 'Breakfast Spots',
             'shortName': 'Breakfast',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/breakfast_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '39686393'}},
          'referralId': 'e-0-4ae5b91ff964a520a6a121e3-3'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '50760559e4b0e8c7babe2497',
           'name': 'Body Blitz Spa East',
           'location': {'address': '497 King Street East',
            'crossStreet': 'btwn Sackville St and Sumach St',
            'lat': 43.65473505045365,
            'lng': -79.35987433132891,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65473505045365,
              'lng': -79.35987433132891}],
            'distance': 80,
            'postalCode': 'M5A 1L9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['497 King Street East (btwn Sackville St and Sumach St)',
             'Toronto ON M5A 1L9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1ed941735',
             'name': 'Spa',
             'pluralName': 'Spas',
             'shortName': 'Spa',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/spa_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-50760559e4b0e8c7babe2497-4'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5612b1cc498e3dd742af0dc8',
           'name': 'Impact Kitchen',
           'location': {'address': '573 King St E',
            'crossStreet': 'at St Lawrence St',
            'lat': 43.65636850543279,
            'lng': -79.35697968750694,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65636850543279,
              'lng': -79.35697968750694}],
            'distance': 376,
            'postalCode': 'M5A 4L3',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['573 King St E (at St Lawrence St)',
             'Toronto ON M5A 4L3',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c4941735',
             'name': 'Restaurant',
             'pluralName': 'Restaurants',
             'shortName': 'Restaurant',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/default_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5612b1cc498e3dd742af0dc8-5'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4af59046f964a520e0f921e3',
           'name': 'Figs Breakfast & Lunch',
           'location': {'address': '344 Queen St. E.',
            'crossStreet': 'at Parliament St.',
            'lat': 43.65567455427388,
            'lng': -79.3645032892494,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65567455427388,
              'lng': -79.3645032892494}],
            'distance': 349,
            'postalCode': 'M5A 1S8',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['344 Queen St. E. (at Parliament St.)',
             'Toronto ON M5A 1S8',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d143941735',
             'name': 'Breakfast Spot',
             'pluralName': 'Breakfast Spots',
             'shortName': 'Breakfast',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/breakfast_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4af59046f964a520e0f921e3-6'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '566e1294498e3f6629006bc3',
           'name': 'Dominion Pub and Kitchen',
           'location': {'address': '500 Queen Street East',
            'lat': 43.65691857501867,
            'lng': -79.35896684476664,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65691857501867,
              'lng': -79.35896684476664}],
            'distance': 325,
            'postalCode': 'M5A 1T9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['500 Queen Street East',
             'Toronto ON M5A 1T9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d11b941735',
             'name': 'Pub',
             'pluralName': 'Pubs',
             'shortName': 'Pub',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/nightlife/pub_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-566e1294498e3f6629006bc3-7'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '51ccc048498ec7792efc955e',
           'name': 'Corktown Common',
           'location': {'lat': 43.655617799749734,
            'lng': -79.3562113397429,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.655617799749734,
              'lng': -79.3562113397429}],
            'distance': 387,
            'cc': 'CA',
            'country': 'Canada',
            'formattedAddress': ['Canada']},
           'categories': [{'id': '4bf58dd8d48988d163941735',
             'name': 'Park',
             'pluralName': 'Parks',
             'shortName': 'Park',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/park_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-51ccc048498ec7792efc955e-8'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c05ef964a520bff620e3',
           'name': 'The Distillery Historic District',
           'location': {'address': 'btwn Front, Cherry, Gardiner & Parliament',
            'lat': 43.65024435658077,
            'lng': -79.35932278633118,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65024435658077,
              'lng': -79.35932278633118}],
            'distance': 459,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['btwn Front, Cherry, Gardiner & Parliament',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4deefb944765f83613cdba6e',
             'name': 'Historic Site',
             'pluralName': 'Historic Sites',
             'shortName': 'Historic Site',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/historicsite_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c05ef964a520bff620e3-9'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b0978e1f964a520cd1723e3',
           'name': 'SOMA chocolatemaker',
           'location': {'address': '55 Mill Street, Unit #48',
            'crossStreet': 'The Distillery District',
            'lat': 43.65062222570758,
            'lng': -79.35812684032683,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65062222570758,
              'lng': -79.35812684032683}],
            'distance': 452,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['55 Mill Street, Unit #48 (The Distillery District)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '52f2ab2ebcbc57f1066b8b31',
             'name': 'Chocolate Shop',
             'pluralName': 'Chocolate Shops',
             'shortName': 'Chocolate Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/candystore_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b0978e1f964a520cd1723e3-10'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5542ab36498e2f92a8c248f2',
           'name': 'Cocina Economica',
           'location': {'address': '114 Berkeley St',
            'crossStreet': 'btwn Queen & Richmond',
            'lat': 43.65495889022676,
            'lng': -79.3656572507398,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65495889022676,
              'lng': -79.3656572507398}],
            'distance': 411,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['114 Berkeley St (btwn Queen & Richmond)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c1941735',
             'name': 'Mexican Restaurant',
             'pluralName': 'Mexican Restaurants',
             'shortName': 'Mexican',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/mexican_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5542ab36498e2f92a8c248f2-11'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c3e1eaa6faac9b66dc60d76',
           'name': 'Distillery Sunday Market',
           'location': {'address': '1 Trinity St',
            'lat': 43.650074989330655,
            'lng': -79.36183171318665,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.650074989330655,
              'lng': -79.36183171318665}],
            'distance': 475,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['1 Trinity St', 'Toronto ON', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1fa941735',
             'name': 'Farmers Market',
             'pluralName': 'Farmers Markets',
             'shortName': "Farmer's Market",
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/food_farmersmarket_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c3e1eaa6faac9b66dc60d76-12'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5619551a498e9e35fce2256b',
           'name': 'Sumach Espresso',
           'location': {'address': '118 Sumach St',
            'lat': 43.65813540553308,
            'lng': -79.35951549011845,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65813540553308,
              'lng': -79.35951549011845}],
            'distance': 440,
            'postalCode': 'M5A 3J9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['118 Sumach St', 'Toronto ON M5A 3J9', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5619551a498e9e35fce2256b-13'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b156a02f964a5207fac23e3',
           'name': 'Brick Street Bakery',
           'location': {'address': '27 Trinity St',
            'crossStreet': 'in Distillery District',
            'lat': 43.650574039683974,
            'lng': -79.35953942981405,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.650574039683974,
              'lng': -79.35953942981405}],
            'distance': 419,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['27 Trinity St (in Distillery District)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16a941735',
             'name': 'Bakery',
             'pluralName': 'Bakeries',
             'shortName': 'Bakery',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/bakery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b156a02f964a5207fac23e3-14'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5653a96f498e99c91027730b',
           'name': 'Cacao 70',
           'location': {'address': '28 Gristmill Lane',
            'lat': 43.650066694561666,
            'lng': -79.36072263183006,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.650066694561666,
              'lng': -79.36072263183006}],
            'distance': 466,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['28 Gristmill Lane',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1d0941735',
             'name': 'Dessert Shop',
             'pluralName': 'Dessert Shops',
             'shortName': 'Desserts',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/dessert_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5653a96f498e99c91027730b-15'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '51853a73498e4d97a8b20831',
           'name': 'Rooster Coffee',
           'location': {'address': '343 King St E',
            'crossStreet': 'btwn Princess & Berkeley St',
            'lat': 43.65189965670432,
            'lng': -79.36560912104514,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65189965670432,
              'lng': -79.36560912104514}],
            'distance': 479,
            'postalCode': 'M5A 1L1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['343 King St E (btwn Princess & Berkeley St)',
             'Toronto ON M5A 1L1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-51853a73498e4d97a8b20831-16'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '57cd9d20498e6ab8342980e2',
           'name': 'Arvo',
           'location': {'address': '17 Gristmill Ln',
            'crossStreet': 'at Parliament St',
            'lat': 43.64996280366945,
            'lng': -79.36144178325522,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.64996280366945,
              'lng': -79.36144178325522}],
            'distance': 482,
            'postalCode': 'M5A 3R6',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['17 Gristmill Ln (at Parliament St)',
             'Toronto ON M5A 3R6',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-57cd9d20498e6ab8342980e2-17'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '58c7fbf7424f9373e6427e99',
           'name': 'Starbucks',
           'location': {'address': '351 King St E,60',
            'lat': 43.65132698417445,
            'lng': -79.36432857157803,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65132698417445,
              'lng': -79.36432857157803}],
            'distance': 441,
            'postalCode': 'M5A 1L1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['351 King St E,60',
             'Toronto ON M5A 1L1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-58c7fbf7424f9373e6427e99-18'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4bb68165f562ef3b88483097',
           'name': 'Young Centre for the Performing Arts',
           'location': {'address': '50 Tank House Ln.',
            'crossStreet': 'at Cherry St.',
            'lat': 43.65082466432163,
            'lng': -79.35759324240415,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65082466432163,
              'lng': -79.35759324240415}],
            'distance': 454,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['50 Tank House Ln. (at Cherry St.)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1f2931735',
             'name': 'Performing Arts Venue',
             'pluralName': 'Performing Arts Venues',
             'shortName': 'Performing Arts',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/performingarts_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4bb68165f562ef3b88483097-19'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '51ddecee498e1ffd34185d2f',
           'name': 'El Catrin',
           'location': {'address': '18 Tank House Lane',
            'crossStreet': 'Distillery District',
            'lat': 43.650600737116996,
            'lng': -79.35892024942333,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.650600737116996,
              'lng': -79.35892024942333}],
            'distance': 430,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['18 Tank House Lane (Distillery District)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c1941735',
             'name': 'Mexican Restaurant',
             'pluralName': 'Mexican Restaurants',
             'shortName': 'Mexican',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/mexican_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '61086042'}},
          'referralId': 'e-0-51ddecee498e1ffd34185d2f-20'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4c16a548955976b0cadea4f6',
           'name': 'Parliament Square Park',
           'location': {'address': '44 Parliament Street',
            'lat': 43.65026388338689,
            'lng': -79.36219509081177,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65026388338689,
              'lng': -79.36219509081177}],
            'distance': 462,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['44 Parliament Street', 'Toronto ON', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d163941735',
             'name': 'Park',
             'pluralName': 'Parks',
             'shortName': 'Park',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/park_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4c16a548955976b0cadea4f6-21'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ddfbaca185035f3a44e8df6',
           'name': 'Underpass Park',
           'location': {'address': 'Eastern Ave.',
            'crossStreet': 'Richmond St.',
            'lat': 43.65576361726024,
            'lng': -79.3548059463501,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65576361726024,
              'lng': -79.3548059463501}],
            'distance': 498,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Eastern Ave. (Richmond St.)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d163941735',
             'name': 'Park',
             'pluralName': 'Parks',
             'shortName': 'Park',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/parks_outdoors/park_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ddfbaca185035f3a44e8df6-22'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ac3e6cef964a520629d20e3',
           'name': 'Archeo',
           'location': {'address': '31 Trinity St.',
            'crossStreet': 'in The Distillery District',
            'lat': 43.65066723014277,
            'lng': -79.35943064816142,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65066723014277,
              'lng': -79.35943064816142}],
            'distance': 411,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['31 Trinity St. (in The Distillery District)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d110941735',
             'name': 'Italian Restaurant',
             'pluralName': 'Italian Restaurants',
             'shortName': 'Italian',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/italian_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '38103525'}},
          'referralId': 'e-0-4ac3e6cef964a520629d20e3-23'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4bc39c914cdfc9b6f29c9721',
           'name': 'Souvlaki Express',
           'location': {'address': '348 Queen street east',
            'crossStreet': 'at Parliament St',
            'lat': 43.65558391537734,
            'lng': -79.36443816909016,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65558391537734,
              'lng': -79.36443816909016}],
            'distance': 339,
            'postalCode': 'M5A 1T1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['348 Queen street east (at Parliament St)',
             'Toronto ON M5A 1T1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10e941735',
             'name': 'Greek Restaurant',
             'pluralName': 'Greek Restaurants',
             'shortName': 'Greek',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/greek_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4bc39c914cdfc9b6f29c9721-24'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '53a22c92498ec91fda7ce133',
           'name': 'Cluny Bistro & Boulangerie',
           'location': {'address': '35 Tank House Lane',
            'crossStreet': 'Trinity St',
            'lat': 43.650565116074695,
            'lng': -79.35784287026658,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.650565116074695,
              'lng': -79.35784287026658}],
            'distance': 468,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'neighborhood': 'Distillery District, Toronto, ON',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['35 Tank House Lane (Trinity St)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10c941735',
             'name': 'French Restaurant',
             'pluralName': 'French Restaurants',
             'shortName': 'French',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/french_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '199972479'}},
          'referralId': 'e-0-53a22c92498ec91fda7ce133-25'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '54d37e23498e5e29bcb35362',
           'name': 'ODIN Cafe + Bar',
           'location': {'address': '514 King St E',
            'crossStreet': 'at River St',
            'lat': 43.656738544928054,
            'lng': -79.35650305267754,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656738544928054,
              'lng': -79.35650305267754}],
            'distance': 432,
            'postalCode': 'M5A 1M1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['514 King St E (at River St)',
             'Toronto ON M5A 1M1',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16d941735',
             'name': 'Café',
             'pluralName': 'Cafés',
             'shortName': 'Café',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-54d37e23498e5e29bcb35362-26'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4af9fc95f964a520ca1522e3',
           'name': 'Mill St. Brew Pub',
           'location': {'address': '21 Tank House Ln',
            'crossStreet': 'at Pure Spirits Mews',
            'lat': 43.65035331843578,
            'lng': -79.35848936650571,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65035331843578,
              'lng': -79.35848936650571}],
            'distance': 467,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'neighborhood': 'Distillery District',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['21 Tank House Ln (at Pure Spirits Mews)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d11b941735',
             'name': 'Pub',
             'pluralName': 'Pubs',
             'shortName': 'Pub',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/nightlife/pub_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4af9fc95f964a520ca1522e3-27'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d84d98181fdb1f7d4a704c0',
           'name': 'Caffe Furbo',
           'location': {'address': '12 case goods lane',
            'lat': 43.649969882303814,
            'lng': -79.35884946388191,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.649969882303814,
              'lng': -79.35884946388191}],
            'distance': 498,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['12 case goods lane',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16d941735',
             'name': 'Café',
             'pluralName': 'Cafés',
             'shortName': 'Café',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []},
           'venuePage': {'id': '47611149'}},
          'referralId': 'e-0-4d84d98181fdb1f7d4a704c0-28'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad4c05df964a5204ef620e3',
           'name': 'The Sweet Escape Patisserie',
           'location': {'address': '55 Mill Street',
            'lat': 43.65063217302609,
            'lng': -79.35870913127346,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65063217302609,
              'lng': -79.35870913127346}],
            'distance': 432,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['55 Mill Street',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16a941735',
             'name': 'Bakery',
             'pluralName': 'Bakeries',
             'shortName': 'Bakery',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/bakery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad4c05df964a5204ef620e3-29'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ade8ea8f964a5205a7621e3',
           'name': 'Berkeley Church',
           'location': {'address': '315 Queen St E',
            'crossStreet': 'at Berkeley St',
            'lat': 43.65512324174501,
            'lng': -79.36587330410705,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65512324174501,
              'lng': -79.36587330410705}],
            'distance': 432,
            'postalCode': 'M5A 1S7',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['315 Queen St E (at Berkeley St)',
             'Toronto ON M5A 1S7',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d171941735',
             'name': 'Event Space',
             'pluralName': 'Event Spaces',
             'shortName': 'Event Space',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/building/eventspace_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ade8ea8f964a5205a7621e3-30'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '51828372e4b0ea1bac906b00',
           'name': 'John Fluevog Shoes',
           'location': {'address': '4 Trinity St.',
            'crossStreet': 'Distillery District',
            'lat': 43.64989585773889,
            'lng': -79.3594359503061,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.64989585773889,
              'lng': -79.3594359503061}],
            'distance': 495,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['4 Trinity St. (Distillery District)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d107951735',
             'name': 'Shoe Store',
             'pluralName': 'Shoe Stores',
             'shortName': 'Shoes',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/apparel_shoestore_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-51828372e4b0ea1bac906b00-31'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b8c46f3f964a5200cc832e3',
           'name': 'Alumnae Theatre',
           'location': {'address': '70 Berkley St.',
            'crossStreet': 'at Adelaide St.',
            'lat': 43.65275554626444,
            'lng': -79.36475283805089,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65275554626444,
              'lng': -79.36475283805089}],
            'distance': 371,
            'postalCode': 'M5A 2W6',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['70 Berkley St. (at Adelaide St.)',
             'Toronto ON M5A 2W6',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d137941735',
             'name': 'Theater',
             'pluralName': 'Theaters',
             'shortName': 'Theater',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/performingarts_theater_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b8c46f3f964a5200cc832e3-32'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4bb52a8bf562ef3b7d992e97',
           'name': 'Soulpepper Theatre',
           'location': {'address': '55 Mill St',
            'lat': 43.65078003430492,
            'lng': -79.35761534068922,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65078003430492,
              'lng': -79.35761534068922}],
            'distance': 457,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['55 Mill St', 'Toronto ON', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d137941735',
             'name': 'Theater',
             'pluralName': 'Theaters',
             'shortName': 'Theater',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/performingarts_theater_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4bb52a8bf562ef3b7d992e97-33'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4dbb2d7a6e815ab0de64d555',
           'name': 'Ontario Spring Water Sake Company',
           'location': {'address': '51 Gristmill Lane',
            'lat': 43.649921646081026,
            'lng': -79.36007302731126,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.649921646081026,
              'lng': -79.36007302731126}],
            'distance': 485,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['51 Gristmill Lane',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '50327c8591d4c4b30a586d5d',
             'name': 'Brewery',
             'pluralName': 'Breweries',
             'shortName': 'Brewery',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/brewery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4dbb2d7a6e815ab0de64d555-34'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '581258b738fa5bbefe4c0857',
           'name': 'Dark Horse Espresso Bar',
           'location': {'address': '416 Front St E',
            'crossStreet': 'Cooperage St',
            'lat': 43.653080578844275,
            'lng': -79.3570778621122,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.653080578844275,
              'lng': -79.3570778621122}],
            'distance': 315,
            'cc': 'CA',
            'neighborhood': 'Downtown Toronto',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['416 Front St E (Cooperage St)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-581258b738fa5bbefe4c0857-35'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4f776828e4b040208c312fc5',
           'name': "Greg's Ice Cream",
           'location': {'address': '55 Mill (Back)',
            'crossStreet': 'Distillery District',
            'lat': 43.65010164153305,
            'lng': -79.3604404498866,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65010164153305,
              'lng': -79.3604404498866}],
            'distance': 463,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['55 Mill (Back) (Distillery District)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c9941735',
             'name': 'Ice Cream Shop',
             'pluralName': 'Ice Cream Shops',
             'shortName': 'Ice Cream',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/icecream_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4f776828e4b040208c312fc5-36'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '583e2cde9435a913b34de355',
           'name': 'Wildly Delicious Cafe',
           'location': {'lat': 43.65043587832196,
            'lng': -79.3588615181307,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65043587832196,
              'lng': -79.3588615181307}],
            'distance': 449,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['Toronto ON', 'Canada']},
           'categories': [{'id': '4bf58dd8d48988d16d941735',
             'name': 'Café',
             'pluralName': 'Cafés',
             'shortName': 'Café',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/cafe_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-583e2cde9435a913b34de355-37'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4af21e78f964a520fae521e3',
           'name': 'Arta Gallery',
           'location': {'address': '14 Distillery Lane',
            'crossStreet': 'Distillery District',
            'lat': 43.650022181882065,
            'lng': -79.36122168476822,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.650022181882065,
              'lng': -79.36122168476822}],
            'distance': 474,
            'postalCode': 'M5A 3C4',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['14 Distillery Lane (Distillery District)',
             'Toronto ON M5A 3C4',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e2931735',
             'name': 'Art Gallery',
             'pluralName': 'Art Galleries',
             'shortName': 'Art Gallery',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/arts_entertainment/artgallery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4af21e78f964a520fae521e3-38'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4ad8d551f964a5201f1521e3',
           'name': 'Vistek',
           'location': {'address': '496 Queen St E',
            'crossStreet': 'at Sumach',
            'lat': 43.65704640025721,
            'lng': -79.35966695607046,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65704640025721,
              'lng': -79.35966695607046}],
            'distance': 319,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['496 Queen St E (at Sumach)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d122951735',
             'name': 'Electronics Store',
             'pluralName': 'Electronics Stores',
             'shortName': 'Electronics',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/technology_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4ad8d551f964a5201f1521e3-39'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b8acdfdf964a520378232e3',
           'name': 'The Beer Store',
           'location': {'address': '28 River Street',
            'crossStreet': 'Queen St E',
            'lat': 43.657773,
            'lng': -79.3574632,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.657773,
              'lng': -79.3574632}],
            'distance': 467,
            'postalCode': 'M5A 3N9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['28 River Street (Queen St E)',
             'Toronto ON M5A 3N9',
             'Canada']},
           'categories': [{'id': '5370f356bcbc57f1066c94c2',
             'name': 'Beer Store',
             'pluralName': 'Beer Stores',
             'shortName': 'Beer Store',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/nightlife/beergarden_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b8acdfdf964a520378232e3-40'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4d126fd3d1848cfad5cfbb71',
           'name': 'TD Canada Trust',
           'location': {'address': '457 Front Street East',
            'lat': 43.652779762702764,
            'lng': -79.3563362625656,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.652779762702764,
              'lng': -79.3563362625656}],
            'distance': 383,
            'postalCode': 'M5A 0J2',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['457 Front Street East',
             'Toronto ON M5A 0J2',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d10a951735',
             'name': 'Bank',
             'pluralName': 'Banks',
             'shortName': 'Bank',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/financial_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4d126fd3d1848cfad5cfbb71-41'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '4b2cf78cf964a52077cb24e3',
           'name': 'Savoury Grounds',
           'location': {'address': '481 Queen St. East',
            'crossStreet': 'Sumach St',
            'lat': 43.656820970004496,
            'lng': -79.3589698353747,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656820970004496,
              'lng': -79.3589698353747}],
            'distance': 315,
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['481 Queen St. East (Sumach St)',
             'Toronto ON',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-4b2cf78cf964a52077cb24e3-42'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '51c085d3498eadedb67ba6cd',
           'name': 'Flame Shack',
           'location': {'address': '506 Queen St E',
            'crossStreet': 'Sumach St',
            'lat': 43.656844075440546,
            'lng': -79.35891727496157,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656844075440546,
              'lng': -79.35891727496157}],
            'distance': 319,
            'postalCode': 'M5A 1V2',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['506 Queen St E (Sumach St)',
             'Toronto ON M5A 1V2',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1c4941735',
             'name': 'Restaurant',
             'pluralName': 'Restaurants',
             'shortName': 'Restaurant',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/default_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-51c085d3498eadedb67ba6cd-43'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '593b5fbea35dce738ebbb4ec',
           'name': 'Residence & Conference Centre',
           'location': {'address': '80 Cooperage',
            'crossStreet': 'Front Street',
            'lat': 43.65304,
            'lng': -79.35704,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.65304,
              'lng': -79.35704}],
            'distance': 319,
            'postalCode': 'M5A 1G9',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['80 Cooperage (Front Street)',
             'Toronto ON M5A 1G9',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1fa931735',
             'name': 'Hotel',
             'pluralName': 'Hotels',
             'shortName': 'Hotel',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/travel/hotel_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-593b5fbea35dce738ebbb4ec-44'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '5292543a498ec4d4c99c0c64',
           'name': 'The Healthy Road',
           'location': {'address': '518 King Street East',
            'crossStreet': 'Sumach',
            'lat': 43.656264585886454,
            'lng': -79.35711882680904,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.656264585886454,
              'lng': -79.35711882680904}],
            'distance': 360,
            'postalCode': 'M5A 1M1',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['518 King Street East (Sumach)',
             'Toronto ON M5A 1M1',
             'Canada']},
           'categories': [{'id': '50aa9e744b90af0d42d5de0e',
             'name': 'Health Food Store',
             'pluralName': 'Health Food Stores',
             'shortName': 'Health Food Store',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/food_grocery_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-5292543a498ec4d4c99c0c64-45'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '57dd8bcd498e8eec19b9c896',
           'name': 'FUEL+',
           'location': {'address': '469 Front St',
            'crossStreet': 'Rolling Mills Rd',
            'lat': 43.653193,
            'lng': -79.355867,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.653193,
              'lng': -79.355867}],
            'distance': 402,
            'postalCode': 'M5A',
            'cc': 'CA',
            'city': 'Toronto',
            'state': 'ON',
            'country': 'Canada',
            'formattedAddress': ['469 Front St (Rolling Mills Rd)',
             'Toronto ON M5A',
             'Canada']},
           'categories': [{'id': '4bf58dd8d48988d1e0931735',
             'name': 'Coffee Shop',
             'pluralName': 'Coffee Shops',
             'shortName': 'Coffee Shop',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/food/coffeeshop_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-57dd8bcd498e8eec19b9c896-46'},
         {'reasons': {'count': 0,
           'items': [{'summary': 'This spot is popular',
             'type': 'general',
             'reasonName': 'globalInteractionReason'}]},
          'venue': {'id': '579cc5f2498e0fe8397deebb',
           'name': 'GW General',
           'location': {'lat': 43.650495,
            'lng': -79.357538,
            'labeledLatLngs': [{'label': 'display',
              'lat': 43.650495,
              'lng': -79.357538}],
            'distance': 487,
            'cc': 'CA',
            'country': 'Canada',
            'formattedAddress': ['Canada']},
           'categories': [{'id': '4bf58dd8d48988d116951735',
             'name': 'Antique Shop',
             'pluralName': 'Antique Shops',
             'shortName': 'Antiques',
             'icon': {'prefix': 'https://ss3.4sqi.net/img/categories_v2/shops/antique_',
              'suffix': '.png'},
             'primary': True}],
           'photos': {'count': 0, 'groups': []}},
          'referralId': 'e-0-579cc5f2498e0fe8397deebb-47'}]}]}}



#### We know that all the information is in the "items" key. Before we proceed, we will borrow the "get_category_type" function from the Foursquare lab


```python
# function that extracts the category of the venue
def get_category_type(row):
    try:
        categories_list = row['categories']
    except:
        categories_list = row['venue.categories']
        
    if len(categories_list) == 0:
        return None
    else:
        return categories_list[0]['name']
```

### Clean the json and structure it into a pandas dataframe


```python
venues = results['response']['groups'][0]['items']
    
nearby_venues = json_normalize(venues) # flatten JSON

# filter columns
filtered_columns = ['venue.name', 'venue.categories', 'venue.location.lat', 'venue.location.lng']
nearby_venues =nearby_venues.loc[:, filtered_columns]

# filter the category for each row
nearby_venues['venue.categories'] = nearby_venues.apply(get_category_type, axis=1)

# clean columns
nearby_venues.columns = [col.split(".")[-1] for col in nearby_venues.columns]

nearby_venues.head()
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
      <th>name</th>
      <th>categories</th>
      <th>lat</th>
      <th>lng</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Roselle Desserts</td>
      <td>Bakery</td>
      <td>43.653447</td>
      <td>-79.362017</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Tandem Coffee</td>
      <td>Coffee Shop</td>
      <td>43.653559</td>
      <td>-79.361809</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Toronto Cooper Koo Family Cherry St YMCA Centre</td>
      <td>Gym / Fitness Center</td>
      <td>43.653191</td>
      <td>-79.357947</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Morning Glory Cafe</td>
      <td>Breakfast Spot</td>
      <td>43.653947</td>
      <td>-79.361149</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Body Blitz Spa East</td>
      <td>Spa</td>
      <td>43.654735</td>
      <td>-79.359874</td>
    </tr>
  </tbody>
</table>
</div>



### Check how many venues were returned by Foursquare


```python
print('{} venues were returned by Foursquare.'.format(nearby_venues.shape[0]))
```

    48 venues were returned by Foursquare.


## Explore Neighborhoods in Downtown Toronto

#### Create a function to repeat the same process to all the neighborhoods in Downtown Toronto


```python
def getNearbyVenues(names, latitudes, longitudes, radius=500):
    
    venues_list=[]
    for name, lat, lng in zip(names, latitudes, longitudes):
        print(name)
            
        # create the API request URL
        url = 'https://api.foursquare.com/v2/venues/explore?&client_id={}&client_secret={}&v={}&ll={},{}&radius={}&limit={}'.format(
            CLIENT_ID, 
            CLIENT_SECRET, 
            VERSION, 
            lat, 
            lng, 
            radius, 
            LIMIT)
            
        # make the GET request
        results = requests.get(url).json()["response"]['groups'][0]['items']
        
        # return only relevant information for each nearby venue
        venues_list.append([(
            name, 
            lat, 
            lng, 
            v['venue']['name'], 
            v['venue']['location']['lat'], 
            v['venue']['location']['lng'],  
            v['venue']['categories'][0]['name']) for v in results])

    nearby_venues = pd.DataFrame([item for venue_list in venues_list for item in venue_list])
    nearby_venues.columns = ['Neighborhood', 
                  'Neighborhood Latitude', 
                  'Neighborhood Longitude', 
                  'Venue', 
                  'Venue Latitude', 
                  'Venue Longitude', 
                  'Venue Category']
    
    return(nearby_venues)
```

### Running the above function on each neighborhood and create a new dataframe called *DowntownTorontoVenues*.


```python
DowntownTorontoVenues = getNearbyVenues(names=df_toronto_new['Neighborhood'],
                                   latitudes=df_toronto_new['Latitude'],
                                   longitudes=df_toronto_new['Longitude']
                                  )
```

    Harbourfront
    Regent Park
    Ryerson
    Garden District
    St. James Town
    Berczy Park
    Central Bay Street
    Christie
    Adelaide
    King
    Richmond
    Harbourfront East
    Toronto Islands
    Union Station
    Design Exchange
    Toronto Dominion Centre
    Commerce Court
    Victoria Hotel
    Harbord
    University of Toronto
    Chinatown
    Grange Park
    Kensington Market
    CN Tower
    Bathurst Quay
    Island airport
    Harbourfront West
    King and Spadina
    Railway Lands
    South Niagara
    Rosedale
    Stn A PO Boxes 25 The Esplanade
    Cabbagetown
    St. James Town
    First Canadian Place
    Underground city
    Church and Wellesley



```python
print(DowntownTorontoVenues.shape)
DowntownTorontoVenues.head()
```

    (2499, 7)





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
      <th>Neighborhood</th>
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Roselle Desserts</td>
      <td>43.653447</td>
      <td>-79.362017</td>
      <td>Bakery</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Tandem Coffee</td>
      <td>43.653559</td>
      <td>-79.361809</td>
      <td>Coffee Shop</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Toronto Cooper Koo Family Cherry St YMCA Centre</td>
      <td>43.653191</td>
      <td>-79.357947</td>
      <td>Gym / Fitness Center</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Morning Glory Cafe</td>
      <td>43.653947</td>
      <td>-79.361149</td>
      <td>Breakfast Spot</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Harbourfront</td>
      <td>43.65426</td>
      <td>-79.360636</td>
      <td>Body Blitz Spa East</td>
      <td>43.654735</td>
      <td>-79.359874</td>
      <td>Spa</td>
    </tr>
  </tbody>
</table>
</div>



### Venues returned for each neighborhood


```python
DowntownTorontoVenues.groupby('Neighborhood').count()
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
      <th>Neighborhood Latitude</th>
      <th>Neighborhood Longitude</th>
      <th>Venue</th>
      <th>Venue Latitude</th>
      <th>Venue Longitude</th>
      <th>Venue Category</th>
    </tr>
    <tr>
      <th>Neighborhood</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Adelaide</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Bathurst Quay</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Berczy Park</th>
      <td>56</td>
      <td>56</td>
      <td>56</td>
      <td>56</td>
      <td>56</td>
      <td>56</td>
    </tr>
    <tr>
      <th>CN Tower</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Cabbagetown</th>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Central Bay Street</th>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
      <td>82</td>
    </tr>
    <tr>
      <th>Chinatown</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Christie</th>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
      <td>15</td>
    </tr>
    <tr>
      <th>Church and Wellesley</th>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
      <td>87</td>
    </tr>
    <tr>
      <th>Commerce Court</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Design Exchange</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>First Canadian Place</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Garden District</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Grange Park</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Harbord</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Harbourfront</th>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Harbourfront East</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Harbourfront West</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Island airport</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Kensington Market</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>King</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>King and Spadina</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Railway Lands</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>Regent Park</th>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
      <td>48</td>
    </tr>
    <tr>
      <th>Richmond</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Rosedale</th>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Ryerson</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>South Niagara</th>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
      <td>14</td>
    </tr>
    <tr>
      <th>St. James Town</th>
      <td>148</td>
      <td>148</td>
      <td>148</td>
      <td>148</td>
      <td>148</td>
      <td>148</td>
    </tr>
    <tr>
      <th>Stn A PO Boxes 25 The Esplanade</th>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
      <td>95</td>
    </tr>
    <tr>
      <th>Toronto Dominion Centre</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Toronto Islands</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Underground city</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>Union Station</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
    <tr>
      <th>University of Toronto</th>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
      <td>35</td>
    </tr>
    <tr>
      <th>Victoria Hotel</th>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
      <td>100</td>
    </tr>
  </tbody>
</table>
</div>



#### Finding out how many unique categories can be curated from all the returned venues


```python
print('There are {} uniques categories.'.format(len(DowntownTorontoVenues['Venue Category'].unique())))
```

    There are 206 uniques categories.


## Analyze Each Neighborhood


```python
# one hot encoding
dtToronto_onehot = pd.get_dummies(DowntownTorontoVenues[['Venue Category']], prefix="", prefix_sep="")

# add neighborhood column back to dataframe
dtToronto_onehot['Neighborhood'] = DowntownTorontoVenues['Neighborhood'] 

# move neighborhood column to the first column
fixed_columns = [dtToronto_onehot.columns[-1]] + list(dtToronto_onehot.columns[:-1])
dtToronto_onehot = dtToronto_onehot[fixed_columns]

dtToronto_onehot.head()
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
      <th>Yoga Studio</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>American Restaurant</th>
      <th>...</th>
      <th>Thrift / Vintage Store</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 206 columns</p>
</div>




```python
dtToronto_onehot.shape
```




    (2499, 206)



#### Group rows by neighborhood and by taking the mean of the frequency of occurrence of each category


```python
dtToronto_grouped = dtToronto_onehot.groupby('Neighborhood').mean().reset_index()
dtToronto_grouped
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
      <th>Neighborhood</th>
      <th>Yoga Studio</th>
      <th>Adult Boutique</th>
      <th>Afghan Restaurant</th>
      <th>Airport</th>
      <th>Airport Food Court</th>
      <th>Airport Gate</th>
      <th>Airport Lounge</th>
      <th>Airport Service</th>
      <th>Airport Terminal</th>
      <th>...</th>
      <th>Thrift / Vintage Store</th>
      <th>Toy / Game Store</th>
      <th>Trail</th>
      <th>Train Station</th>
      <th>Vegetarian / Vegan Restaurant</th>
      <th>Video Game Store</th>
      <th>Vietnamese Restaurant</th>
      <th>Wine Bar</th>
      <th>Wings Joint</th>
      <th>Women's Store</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelaide</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bathurst Quay</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Berczy Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CN Tower</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cabbagetown</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Central Bay Street</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.012195</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Chinatown</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Christie</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Church and Wellesley</td>
      <td>0.011494</td>
      <td>0.011494</td>
      <td>0.011494</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.011494</td>
      <td>0.011494</td>
      <td>0.000000</td>
      <td>0.011494</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Commerce Court</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Design Exchange</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>First Canadian Place</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Garden District</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Grange Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Harbord</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Harbourfront</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Harbourfront East</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Harbourfront West</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Island airport</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Kensington Market</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.050000</td>
      <td>0.000000</td>
      <td>0.050000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>King</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>King and Spadina</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Railway Lands</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Regent Park</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Richmond</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.010000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Rosedale</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.25</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Ryerson</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>South Niagara</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.071429</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>28</th>
      <td>St. James Town</td>
      <td>0.006757</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.006757</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Stn A PO Boxes 25 The Esplanade</td>
      <td>0.010526</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Toronto Dominion Centre</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Toronto Islands</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Underground city</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Union Station</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>34</th>
      <td>University of Toronto</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.028571</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Victoria Hotel</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
<p>36 rows × 206 columns</p>
</div>




```python
dtToronto_grouped.shape
```




    (36, 206)



#### Print each neighborhood along with the top 5 most common venues


```python
num_top_venues = 5

for hood in dtToronto_grouped['Neighborhood']:
    print("----"+hood+"----")
    temp = dtToronto_grouped[dtToronto_grouped['Neighborhood'] == hood].T.reset_index()
    temp.columns = ['venue','freq']
    temp = temp.iloc[1:]
    temp['freq'] = temp['freq'].astype(float)
    temp = temp.round({'freq': 2})
    print(temp.sort_values('freq', ascending=False).reset_index(drop=True).head(num_top_venues))
    print('\n')
```

    ----Adelaide----
                     venue  freq
    0          Coffee Shop  0.06
    1                 Café  0.05
    2           Steakhouse  0.04
    3  American Restaurant  0.04
    4      Thai Restaurant  0.04
    
    
    ----Bathurst Quay----
                  venue  freq
    0   Airport Service  0.14
    1  Airport Terminal  0.14
    2    Airport Lounge  0.14
    3             Plane  0.07
    4     Boat or Ferry  0.07
    
    
    ----Berczy Park----
              venue  freq
    0   Coffee Shop  0.07
    1  Cocktail Bar  0.05
    2    Restaurant  0.05
    3          Café  0.04
    4           Pub  0.04
    
    
    ----CN Tower----
                  venue  freq
    0   Airport Service  0.14
    1  Airport Terminal  0.14
    2    Airport Lounge  0.14
    3             Plane  0.07
    4     Boat or Ferry  0.07
    
    
    ----Cabbagetown----
             venue  freq
    0   Restaurant  0.08
    1  Coffee Shop  0.08
    2          Pub  0.04
    3         Café  0.04
    4       Market  0.04
    
    
    ----Central Bay Street----
                    venue  freq
    0         Coffee Shop  0.16
    1                Café  0.07
    2  Italian Restaurant  0.06
    3        Burger Joint  0.04
    4                 Bar  0.04
    
    
    ----Chinatown----
                               venue  freq
    0                           Café  0.07
    1                            Bar  0.07
    2          Vietnamese Restaurant  0.05
    3  Vegetarian / Vegan Restaurant  0.05
    4                    Coffee Shop  0.04
    
    
    ----Christie----
                   venue  freq
    0      Grocery Store  0.20
    1               Café  0.20
    2               Park  0.13
    3  Convenience Store  0.07
    4          Nightclub  0.07
    
    
    ----Church and Wellesley----
                     venue  freq
    0  Japanese Restaurant  0.07
    1          Coffee Shop  0.06
    2     Sushi Restaurant  0.06
    3              Gay Bar  0.05
    4         Burger Joint  0.03
    
    
    ----Commerce Court----
                     venue  freq
    0          Coffee Shop  0.10
    1                 Café  0.07
    2           Restaurant  0.06
    3                Hotel  0.06
    4  American Restaurant  0.04
    
    
    ----Design Exchange----
                     venue  freq
    0          Coffee Shop  0.14
    1                Hotel  0.08
    2                 Café  0.08
    3  American Restaurant  0.04
    4           Restaurant  0.04
    
    
    ----First Canadian Place----
                     venue  freq
    0                 Café  0.08
    1          Coffee Shop  0.08
    2                Hotel  0.06
    3           Restaurant  0.05
    4  American Restaurant  0.04
    
    
    ----Garden District----
                           venue  freq
    0                Coffee Shop  0.09
    1             Clothing Store  0.09
    2                       Café  0.04
    3  Middle Eastern Restaurant  0.03
    4             Cosmetics Shop  0.03
    
    
    ----Grange Park----
                               venue  freq
    0                           Café  0.07
    1                            Bar  0.07
    2          Vietnamese Restaurant  0.05
    3  Vegetarian / Vegan Restaurant  0.05
    4                    Coffee Shop  0.04
    
    
    ----Harbord----
                     venue  freq
    0                 Café  0.11
    1               Bakery  0.06
    2  Japanese Restaurant  0.06
    3            Bookstore  0.06
    4           Restaurant  0.06
    
    
    ----Harbourfront----
             venue  freq
    0  Coffee Shop  0.17
    1         Park  0.06
    2         Café  0.06
    3       Bakery  0.06
    4      Theater  0.04
    
    
    ----Harbourfront East----
             venue  freq
    0  Coffee Shop  0.14
    1        Hotel  0.05
    2     Aquarium  0.05
    3  Pizza Place  0.04
    4         Café  0.04
    
    
    ----Harbourfront West----
                  venue  freq
    0   Airport Service  0.14
    1  Airport Terminal  0.14
    2    Airport Lounge  0.14
    3             Plane  0.07
    4     Boat or Ferry  0.07
    
    
    ----Island airport----
                  venue  freq
    0   Airport Service  0.14
    1  Airport Terminal  0.14
    2    Airport Lounge  0.14
    3             Plane  0.07
    4     Boat or Ferry  0.07
    
    
    ----Kensington Market----
                               venue  freq
    0                           Café  0.07
    1                            Bar  0.07
    2          Vietnamese Restaurant  0.05
    3  Vegetarian / Vegan Restaurant  0.05
    4                    Coffee Shop  0.04
    
    
    ----King----
                     venue  freq
    0          Coffee Shop  0.06
    1                 Café  0.05
    2           Steakhouse  0.04
    3  American Restaurant  0.04
    4      Thai Restaurant  0.04
    
    
    ----King and Spadina----
                  venue  freq
    0   Airport Service  0.14
    1  Airport Terminal  0.14
    2    Airport Lounge  0.14
    3             Plane  0.07
    4     Boat or Ferry  0.07
    
    
    ----Railway Lands----
                  venue  freq
    0   Airport Service  0.14
    1  Airport Terminal  0.14
    2    Airport Lounge  0.14
    3             Plane  0.07
    4     Boat or Ferry  0.07
    
    
    ----Regent Park----
             venue  freq
    0  Coffee Shop  0.17
    1         Park  0.06
    2         Café  0.06
    3       Bakery  0.06
    4      Theater  0.04
    
    
    ----Richmond----
                     venue  freq
    0          Coffee Shop  0.06
    1                 Café  0.05
    2           Steakhouse  0.04
    3  American Restaurant  0.04
    4      Thai Restaurant  0.04
    
    
    ----Rosedale----
             venue  freq
    0         Park  0.50
    1   Playground  0.25
    2        Trail  0.25
    3  Yoga Studio  0.00
    4  Music Venue  0.00
    
    
    ----Ryerson----
                           venue  freq
    0                Coffee Shop  0.09
    1             Clothing Store  0.09
    2                       Café  0.04
    3  Middle Eastern Restaurant  0.03
    4             Cosmetics Shop  0.03
    
    
    ----South Niagara----
                  venue  freq
    0   Airport Service  0.14
    1  Airport Terminal  0.14
    2    Airport Lounge  0.14
    3             Plane  0.07
    4     Boat or Ferry  0.07
    
    
    ----St. James Town----
                venue  freq
    0     Coffee Shop  0.07
    1      Restaurant  0.07
    2            Café  0.05
    3  Clothing Store  0.03
    4  Breakfast Spot  0.03
    
    
    ----Stn A PO Boxes 25 The Esplanade----
                    venue  freq
    0         Coffee Shop  0.09
    1          Restaurant  0.05
    2                Café  0.04
    3                 Pub  0.03
    4  Italian Restaurant  0.03
    
    
    ----Toronto Dominion Centre----
                     venue  freq
    0          Coffee Shop  0.14
    1                Hotel  0.08
    2                 Café  0.08
    3  American Restaurant  0.04
    4           Restaurant  0.04
    
    
    ----Toronto Islands----
             venue  freq
    0  Coffee Shop  0.14
    1        Hotel  0.05
    2     Aquarium  0.05
    3  Pizza Place  0.04
    4         Café  0.04
    
    
    ----Underground city----
                     venue  freq
    0                 Café  0.08
    1          Coffee Shop  0.08
    2                Hotel  0.06
    3           Restaurant  0.05
    4  American Restaurant  0.04
    
    
    ----Union Station----
             venue  freq
    0  Coffee Shop  0.14
    1        Hotel  0.05
    2     Aquarium  0.05
    3  Pizza Place  0.04
    4         Café  0.04
    
    
    ----University of Toronto----
                     venue  freq
    0                 Café  0.11
    1               Bakery  0.06
    2  Japanese Restaurant  0.06
    3            Bookstore  0.06
    4           Restaurant  0.06
    
    
    ----Victoria Hotel----
                     venue  freq
    0          Coffee Shop  0.10
    1                 Café  0.07
    2           Restaurant  0.06
    3                Hotel  0.06
    4  American Restaurant  0.04
    
    


#### Organising the data into a *pandas* dataframe


```python
## function to sort the venues in descending order
def return_most_common_venues(row, num_top_venues):
    row_categories = row.iloc[1:]
    row_categories_sorted = row_categories.sort_values(ascending=False)
    
    return row_categories_sorted.index.values[0:num_top_venues]
```


```python
## create the new dataframe and display the top 10 venues for each neighborhood
num_top_venues = 10

indicators = ['st', 'nd', 'rd']

# create columns according to number of top venues
columns = ['Neighborhood']
for ind in np.arange(num_top_venues):
    try:
        columns.append('{}{} Most Common Venue'.format(ind+1, indicators[ind]))
    except:
        columns.append('{}th Most Common Venue'.format(ind+1))

# create a new dataframe
neighborhoods_venues_sorted = pd.DataFrame(columns=columns)
neighborhoods_venues_sorted['Neighborhood'] = dtToronto_grouped['Neighborhood']

for ind in np.arange(dtToronto_grouped.shape[0]):
    neighborhoods_venues_sorted.iloc[ind, 1:] = return_most_common_venues(dtToronto_grouped.iloc[ind, :], num_top_venues)

neighborhoods_venues_sorted.head()
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
      <th>Neighborhood</th>
      <th>1st Most Common Venue</th>
      <th>2nd Most Common Venue</th>
      <th>3rd Most Common Venue</th>
      <th>4th Most Common Venue</th>
      <th>5th Most Common Venue</th>
      <th>6th Most Common Venue</th>
      <th>7th Most Common Venue</th>
      <th>8th Most Common Venue</th>
      <th>9th Most Common Venue</th>
      <th>10th Most Common Venue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Adelaide</td>
      <td>Coffee Shop</td>
      <td>Café</td>
      <td>Steakhouse</td>
      <td>American Restaurant</td>
      <td>Thai Restaurant</td>
      <td>Hotel</td>
      <td>Asian Restaurant</td>
      <td>Clothing Store</td>
      <td>Bakery</td>
      <td>Bar</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bathurst Quay</td>
      <td>Airport Lounge</td>
      <td>Airport Service</td>
      <td>Airport Terminal</td>
      <td>Boat or Ferry</td>
      <td>Plane</td>
      <td>Sculpture Garden</td>
      <td>Boutique</td>
      <td>Airport</td>
      <td>Airport Food Court</td>
      <td>Airport Gate</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Berczy Park</td>
      <td>Coffee Shop</td>
      <td>Cocktail Bar</td>
      <td>Restaurant</td>
      <td>Pub</td>
      <td>Café</td>
      <td>Steakhouse</td>
      <td>Cheese Shop</td>
      <td>Italian Restaurant</td>
      <td>Beer Bar</td>
      <td>Seafood Restaurant</td>
    </tr>
    <tr>
      <th>3</th>
      <td>CN Tower</td>
      <td>Airport Lounge</td>
      <td>Airport Service</td>
      <td>Airport Terminal</td>
      <td>Boat or Ferry</td>
      <td>Plane</td>
      <td>Sculpture Garden</td>
      <td>Boutique</td>
      <td>Airport</td>
      <td>Airport Food Court</td>
      <td>Airport Gate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Cabbagetown</td>
      <td>Restaurant</td>
      <td>Coffee Shop</td>
      <td>Bakery</td>
      <td>Convenience Store</td>
      <td>Pub</td>
      <td>Café</td>
      <td>Market</td>
      <td>Pizza Place</td>
      <td>Italian Restaurant</td>
      <td>Park</td>
    </tr>
  </tbody>
</table>
</div>



## Cluster the Neighborhoods

### Running k-means to Cluster the neighbourhoods into 5 Clusters


```python
# set number of clusters
kclusters = 5

dtToronto_grouped_clustering = dtToronto_grouped.drop('Neighborhood', 1)

# run k-means clustering
kmeans = KMeans(n_clusters=kclusters, random_state=0).fit(dtToronto_grouped_clustering)

# check cluster labels generated for each row in the dataframe
kmeans.labels_[0:10] 
```




    array([0, 2, 0, 2, 0, 0, 3, 4, 0, 0], dtype=int32)



### Create a new dataframe that includes the cluster as well as the top 10 venues for each neighborhood


```python
# add clustering labels
neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)

dtToronto_merged = df_toronto_new

# merge toronto_grouped with toronto_data to add latitude/longitude for each neighborhood
dtToronto_merged = dtToronto_merged1.join(neighborhoods_venues_sorted.set_index('Neighborhood'), on='Neighborhood')

dtToronto_merged.head() 
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-82-dee2dd6ec7c5> in <module>
          1 # add clustering labels
    ----> 2 neighborhoods_venues_sorted.insert(0, 'Cluster Labels', kmeans.labels_)
          3 
          4 dtToronto_merged1 = df_toronto_new
          5 


    ~/conda/lib/python3.6/site-packages/pandas/core/frame.py in insert(self, loc, column, value, allow_duplicates)
       3220         value = self._sanitize_column(column, value, broadcast=False)
       3221         self._data.insert(loc, column, value,
    -> 3222                           allow_duplicates=allow_duplicates)
       3223 
       3224     def assign(self, **kwargs):


    ~/conda/lib/python3.6/site-packages/pandas/core/internals.py in insert(self, loc, item, value, allow_duplicates)
       4336         if not allow_duplicates and item in self.items:
       4337             # Should this be a different kind of error??
    -> 4338             raise ValueError('cannot insert {}, already exists'.format(item))
       4339 
       4340         if not isinstance(loc, int):


    ValueError: cannot insert Cluster Labels, already exists


## Visualize the resulting clusters


```python
# create map
map_clusters = folium.Map(location=[latitude, longitude], zoom_start=11)

# set color scheme for the clusters
x = np.arange(kclusters)
ys = [i + x + (i*x)**2 for i in range(kclusters)]
colors_array = cm.rainbow(np.linspace(0, 1, len(ys)))
rainbow = [colors.rgb2hex(i) for i in colors_array]

# add markers to the map
markers_colors = []
for lat, lon, poi, cluster in zip(dtToronto_merged['Latitude'], dtToronto_merged['Longitude'], dtToronto_merged['Neighborhood'], dtToronto_merged['Cluster Labels']):
    label = folium.Popup(str(poi) + ' Cluster ' + str(cluster), parse_html=True)
    folium.CircleMarker(
        [lat, lon],
        radius=5,
        popup=label,
        color=rainbow[cluster-1],
        fill=True,
        fill_color=rainbow[cluster-1],
        fill_opacity=0.7).add_to(map_clusters)
       
map_clusters
```




<div style="width:100%;"><div style="position:relative;width:100%;height:0;padding-bottom:60%;"><iframe src="data:text/html;charset=utf-8;base64,PCFET0NUWVBFIGh0bWw+CjxoZWFkPiAgICAKICAgIDxtZXRhIGh0dHAtZXF1aXY9ImNvbnRlbnQtdHlwZSIgY29udGVudD0idGV4dC9odG1sOyBjaGFyc2V0PVVURi04IiAvPgogICAgPHNjcmlwdD5MX1BSRUZFUl9DQU5WQVMgPSBmYWxzZTsgTF9OT19UT1VDSCA9IGZhbHNlOyBMX0RJU0FCTEVfM0QgPSBmYWxzZTs8L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmpzIj48L3NjcmlwdD4KICAgIDxzY3JpcHQgc3JjPSJodHRwczovL2FqYXguZ29vZ2xlYXBpcy5jb20vYWpheC9saWJzL2pxdWVyeS8xLjExLjEvanF1ZXJ5Lm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvanMvYm9vdHN0cmFwLm1pbi5qcyI+PC9zY3JpcHQ+CiAgICA8c2NyaXB0IHNyYz0iaHR0cHM6Ly9jZG5qcy5jbG91ZGZsYXJlLmNvbS9hamF4L2xpYnMvTGVhZmxldC5hd2Vzb21lLW1hcmtlcnMvMi4wLjIvbGVhZmxldC5hd2Vzb21lLW1hcmtlcnMuanMiPjwvc2NyaXB0PgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2Nkbi5qc2RlbGl2ci5uZXQvbnBtL2xlYWZsZXRAMS4yLjAvZGlzdC9sZWFmbGV0LmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL21heGNkbi5ib290c3RyYXBjZG4uY29tL2Jvb3RzdHJhcC8zLjIuMC9jc3MvYm9vdHN0cmFwLm1pbi5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9tYXhjZG4uYm9vdHN0cmFwY2RuLmNvbS9ib290c3RyYXAvMy4yLjAvY3NzL2Jvb3RzdHJhcC10aGVtZS5taW4uY3NzIi8+CiAgICA8bGluayByZWw9InN0eWxlc2hlZXQiIGhyZWY9Imh0dHBzOi8vbWF4Y2RuLmJvb3RzdHJhcGNkbi5jb20vZm9udC1hd2Vzb21lLzQuNi4zL2Nzcy9mb250LWF3ZXNvbWUubWluLmNzcyIvPgogICAgPGxpbmsgcmVsPSJzdHlsZXNoZWV0IiBocmVmPSJodHRwczovL2NkbmpzLmNsb3VkZmxhcmUuY29tL2FqYXgvbGlicy9MZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy8yLjAuMi9sZWFmbGV0LmF3ZXNvbWUtbWFya2Vycy5jc3MiLz4KICAgIDxsaW5rIHJlbD0ic3R5bGVzaGVldCIgaHJlZj0iaHR0cHM6Ly9yYXdnaXQuY29tL3B5dGhvbi12aXN1YWxpemF0aW9uL2ZvbGl1bS9tYXN0ZXIvZm9saXVtL3RlbXBsYXRlcy9sZWFmbGV0LmF3ZXNvbWUucm90YXRlLmNzcyIvPgogICAgPHN0eWxlPmh0bWwsIGJvZHkge3dpZHRoOiAxMDAlO2hlaWdodDogMTAwJTttYXJnaW46IDA7cGFkZGluZzogMDt9PC9zdHlsZT4KICAgIDxzdHlsZT4jbWFwIHtwb3NpdGlvbjphYnNvbHV0ZTt0b3A6MDtib3R0b206MDtyaWdodDowO2xlZnQ6MDt9PC9zdHlsZT4KICAgIAogICAgICAgICAgICA8c3R5bGU+ICNtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQgewogICAgICAgICAgICAgICAgcG9zaXRpb24gOiByZWxhdGl2ZTsKICAgICAgICAgICAgICAgIHdpZHRoIDogMTAwLjAlOwogICAgICAgICAgICAgICAgaGVpZ2h0OiAxMDAuMCU7CiAgICAgICAgICAgICAgICBsZWZ0OiAwLjAlOwogICAgICAgICAgICAgICAgdG9wOiAwLjAlOwogICAgICAgICAgICAgICAgfQogICAgICAgICAgICA8L3N0eWxlPgogICAgICAgIAo8L2hlYWQ+Cjxib2R5PiAgICAKICAgIAogICAgICAgICAgICA8ZGl2IGNsYXNzPSJmb2xpdW0tbWFwIiBpZD0ibWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0IiA+PC9kaXY+CiAgICAgICAgCjwvYm9keT4KPHNjcmlwdD4gICAgCiAgICAKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGJvdW5kcyA9IG51bGw7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgdmFyIG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCA9IEwubWFwKAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgJ21hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCcsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB7Y2VudGVyOiBbNDMuNjU1MTE1LC03OS4zODAyMTldLAogICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgem9vbTogMTEsCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICBtYXhCb3VuZHM6IGJvdW5kcywKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGxheWVyczogW10sCiAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICB3b3JsZENvcHlKdW1wOiBmYWxzZSwKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgIGNyczogTC5DUlMuRVBTRzM4NTcKICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgfSk7CiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciB0aWxlX2xheWVyX2YzNTFiY2EyZGMwYzRhNGNiMTQxZWRlZTIxNjY4ZWJkID0gTC50aWxlTGF5ZXIoCiAgICAgICAgICAgICAgICAnaHR0cHM6Ly97c30udGlsZS5vcGVuc3RyZWV0bWFwLm9yZy97en0ve3h9L3t5fS5wbmcnLAogICAgICAgICAgICAgICAgewogICJhdHRyaWJ1dGlvbiI6IG51bGwsCiAgImRldGVjdFJldGluYSI6IGZhbHNlLAogICJtYXhab29tIjogMTgsCiAgIm1pblpvb20iOiAxLAogICJub1dyYXAiOiBmYWxzZSwKICAic3ViZG9tYWlucyI6ICJhYmMiCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82ODBhMGQzNzliMDA0NjgzOThlMWE5YWZkMzIzNjk1NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NDI1OTksLTc5LjM2MDYzNTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMWQwM2U2YjIwZDQ4NDFlOWJmMjYyMzEzM2JmMjMxNWQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjJjMjNlYzNiODc3NDJkOThkMTdjNWViZjU4YmJlNjYgPSAkKCc8ZGl2IGlkPSJodG1sXzIyYzIzZWMzYjg3NzQyZDk4ZDE3YzVlYmY1OGJiZTY2IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3VyZnJvbnQgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8xZDAzZTZiMjBkNDg0MWU5YmYyNjIzMTMzYmYyMzE1ZC5zZXRDb250ZW50KGh0bWxfMjJjMjNlYzNiODc3NDJkOThkMTdjNWViZjU4YmJlNjYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjgwYTBkMzc5YjAwNDY4Mzk4ZTFhOWFmZDMyMzY5NTUuYmluZFBvcHVwKHBvcHVwXzFkMDNlNmIyMGQ0ODQxZTliZjI2MjMxMzNiZjIzMTVkKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ2YmJjODliMmU2YjRhMmNhNTc0NWIwZjQ3ZDQ3NDVjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU0MjU5OSwtNzkuMzYwNjM1OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF81YTQxMWRmMGE4ZDY0ZTg5ODY1ZjdlMWE2N2Y2YzkwYSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yM2ViZjEwN2ZhMWM0MzIxYjhmNmRkMGViN2M4ZjgxZiA9ICQoJzxkaXYgaWQ9Imh0bWxfMjNlYmYxMDdmYTFjNDMyMWI4ZjZkZDBlYjdjOGY4MWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJlZ2VudCBQYXJrIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNWE0MTFkZjBhOGQ2NGU4OTg2NWY3ZTFhNjdmNmM5MGEuc2V0Q29udGVudChodG1sXzIzZWJmMTA3ZmExYzQzMjFiOGY2ZGQwZWI3YzhmODFmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ2YmJjODliMmU2YjRhMmNhNTc0NWIwZjQ3ZDQ3NDVjLmJpbmRQb3B1cChwb3B1cF81YTQxMWRmMGE4ZDY0ZTg5ODY1ZjdlMWE2N2Y2YzkwYSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl84MGQzYzNmMGEwOGU0NjQ4OTMwNWVjMTEzODBkN2RjYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1NzE2MTgsLTc5LjM3ODkzNzA5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzZkMjU2ZTFiNmJhODQ2Njk5MGQyMmZjZDExNDEwNTNiID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk4NDNiYjk0YjgyYjQ4NjY5YjVkMjVhNTUxOTgyMDY1ID0gJCgnPGRpdiBpZD0iaHRtbF85ODQzYmI5NGI4MmI0ODY2OWI1ZDI1YTU1MTk4MjA2NSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+UnllcnNvbiBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzZkMjU2ZTFiNmJhODQ2Njk5MGQyMmZjZDExNDEwNTNiLnNldENvbnRlbnQoaHRtbF85ODQzYmI5NGI4MmI0ODY2OWI1ZDI1YTU1MTk4MjA2NSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl84MGQzYzNmMGEwOGU0NjQ4OTMwNWVjMTEzODBkN2RjYi5iaW5kUG9wdXAocG9wdXBfNmQyNTZlMWI2YmE4NDY2OTkwZDIyZmNkMTE0MTA1M2IpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMGQyN2JhNjA0NmE1NGJkOGEzN2MyMDEyYTNjYjI2ZmYgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTcxNjE4LC03OS4zNzg5MzcwOTk5OTk5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83YWNmNjBjYzgyZDI0YzNjODhmY2M3NmVmNDkzNjMxNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF84NzA2ZTIzMjVhZDQ0NDNmYWZkMzc1MTdkYzJkYzIwMiA9ICQoJzxkaXYgaWQ9Imh0bWxfODcwNmUyMzI1YWQ0NDQzZmFmZDM3NTE3ZGMyZGMyMDIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkdhcmRlbiBEaXN0cmljdCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzdhY2Y2MGNjODJkMjRjM2M4OGZjYzc2ZWY0OTM2MzE1LnNldENvbnRlbnQoaHRtbF84NzA2ZTIzMjVhZDQ0NDNmYWZkMzc1MTdkYzJkYzIwMik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZDI3YmE2MDQ2YTU0YmQ4YTM3YzIwMTJhM2NiMjZmZi5iaW5kUG9wdXAocG9wdXBfN2FjZjYwY2M4MmQyNGMzYzg4ZmNjNzZlZjQ5MzYzMTUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYmU2N2ExNDQ1MTQ0NDk5NWE4MjIwMTdlZDlmMTE0NGQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTE0OTM5LC03OS4zNzU0MTc5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NkNTY3MTgyZGUzZDQ3Y2RhOTA1ZWUyZjRjMDNhMTY3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzBkNTc2YWFjZTcyMTQ1OTQ5OTEyY2NkYWYyZDQ3YjIxID0gJCgnPGRpdiBpZD0iaHRtbF8wZDU3NmFhY2U3MjE0NTk0OTkxMmNjZGFmMmQ0N2IyMSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+U3QuIEphbWVzIFRvd24gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9jZDU2NzE4MmRlM2Q0N2NkYTkwNWVlMmY0YzAzYTE2Ny5zZXRDb250ZW50KGh0bWxfMGQ1NzZhYWNlNzIxNDU5NDk5MTJjY2RhZjJkNDdiMjEpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfYmU2N2ExNDQ1MTQ0NDk5NWE4MjIwMTdlZDlmMTE0NGQuYmluZFBvcHVwKHBvcHVwX2NkNTY3MTgyZGUzZDQ3Y2RhOTA1ZWUyZjRjMDNhMTY3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzIzOTEyNGNmMmI3MDQwYWVhNjBhODAzNjZkZmNmNzg2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ0NzcwNzk5OTk5OTk2LC03OS4zNzMzMDY0XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzUxNzBjNThmZWZlNjQwMzNhNTA4NjUzZDcyMDNlMzM0ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzI2MzcxOTUxMjg3ZTQ5Y2Q5MjNmZTU1OTQzM2YyNzkzID0gJCgnPGRpdiBpZD0iaHRtbF8yNjM3MTk1MTI4N2U0OWNkOTIzZmU1NTk0MzNmMjc5MyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmVyY3p5IFBhcmsgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF81MTcwYzU4ZmVmZTY0MDMzYTUwODY1M2Q3MjAzZTMzNC5zZXRDb250ZW50KGh0bWxfMjYzNzE5NTEyODdlNDljZDkyM2ZlNTU5NDMzZjI3OTMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjM5MTI0Y2YyYjcwNDBhZWE2MGE4MDM2NmRmY2Y3ODYuYmluZFBvcHVwKHBvcHVwXzUxNzBjNThmZWZlNjQwMzNhNTA4NjUzZDcyMDNlMzM0KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2QxMzNmZWUwZGEwYzQ2MGY4Y2M4YzFjMWEzNmIwNThjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjU3OTUyNCwtNzkuMzg3MzgyNl0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF8yMGM4NzZhMDIyNTI0NDA0YTEzM2JhMGMzZGE1NTI4YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9mMzM3ZWRjNDI1MjY0YzQ1OTE3YzI3YzZlZTJmMDU4YiA9ICQoJzxkaXYgaWQ9Imh0bWxfZjMzN2VkYzQyNTI2NGM0NTkxN2MyN2M2ZWUyZjA1OGIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNlbnRyYWwgQmF5IFN0cmVldCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzIwYzg3NmEwMjI1MjQ0MDRhMTMzYmEwYzNkYTU1MjhjLnNldENvbnRlbnQoaHRtbF9mMzM3ZWRjNDI1MjY0YzQ1OTE3YzI3YzZlZTJmMDU4Yik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9kMTMzZmVlMGRhMGM0NjBmOGNjOGMxYzFhMzZiMDU4Yy5iaW5kUG9wdXAocG9wdXBfMjBjODc2YTAyMjUyNDQwNGExMzNiYTBjM2RhNTUyOGMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZjBjZjdhZDRkZjcxNDg5Y2FkOTdhMmQ3Nzk1YmI4ZDIgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Njk1NDIsLTc5LjQyMjU2MzddLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmYjM2MCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZmIzNjAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNWNjNmFmZTMzYjQ4NDE2MGI5ZmVmYjY1YTc4OGNhNDUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfN2M5NTQ2OTI5Yjc5NDBmOWI1ODkwZWZkZTE5Njc3OTkgPSAkKCc8ZGl2IGlkPSJodG1sXzdjOTU0NjkyOWI3OTQwZjliNTg5MGVmZGUxOTY3Nzk5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DaHJpc3RpZSBDbHVzdGVyIDQ8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzVjYzZhZmUzM2I0ODQxNjBiOWZlZmI2NWE3ODhjYTQ1LnNldENvbnRlbnQoaHRtbF83Yzk1NDY5MjliNzk0MGY5YjU4OTBlZmRlMTk2Nzc5OSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9mMGNmN2FkNGRmNzE0ODljYWQ5N2EyZDc3OTViYjhkMi5iaW5kUG9wdXAocG9wdXBfNWNjNmFmZTMzYjQ4NDE2MGI5ZmVmYjY1YTc4OGNhNDUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNDQyYThmMzBjNTI5NDY2MGI1NmJlNTI1ZWRlMDc0YmMgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9kM2Y2ODRkOGViNWU0MDFjOGM5NTkyZTVjOWMyMWNjNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lYmQ3MDNiMTlmODk0YmI0YTlmN2JhNzUxYmZhZjUxMyA9ICQoJzxkaXYgaWQ9Imh0bWxfZWJkNzAzYjE5Zjg5NGJiNGE5ZjdiYTc1MWJmYWY1MTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkFkZWxhaWRlIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZDNmNjg0ZDhlYjVlNDAxYzhjOTU5MmU1YzljMjFjYzQuc2V0Q29udGVudChodG1sX2ViZDcwM2IxOWY4OTRiYjRhOWY3YmE3NTFiZmFmNTEzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ0MmE4ZjMwYzUyOTQ2NjBiNTZiZTUyNWVkZTA3NGJjLmJpbmRQb3B1cChwb3B1cF9kM2Y2ODRkOGViNWU0MDFjOGM5NTkyZTVjOWMyMWNjNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl85YjRkNzJkYjA4YjA0ZjZiYTY0Mzk4YWM4NDM2ZTcwZiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY1MDU3MTIwMDAwMDAxLC03OS4zODQ1Njc1XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzRjNjBjMzczM2YyNzQ5ZDg5YmI2N2Y5NWUxNjE4NGI1ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzk0MGZkZGRhMzkxODQyY2NhZTFjYjU0ZTkzYjI3ZTlkID0gJCgnPGRpdiBpZD0iaHRtbF85NDBmZGRkYTM5MTg0MmNjYWUxY2I1NGU5M2IyN2U5ZCIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+S2luZyBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzRjNjBjMzczM2YyNzQ5ZDg5YmI2N2Y5NWUxNjE4NGI1LnNldENvbnRlbnQoaHRtbF85NDBmZGRkYTM5MTg0MmNjYWUxY2I1NGU5M2IyN2U5ZCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85YjRkNzJkYjA4YjA0ZjZiYTY0Mzk4YWM4NDM2ZTcwZi5iaW5kUG9wdXAocG9wdXBfNGM2MGMzNzMzZjI3NDlkODliYjY3Zjk1ZTE2MTg0YjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfMjNlOTc5NDI0OTY1NDhkMWEwYTlmYzQ0Nzg4YmVmY2EgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTA1NzEyMDAwMDAwMSwtNzkuMzg0NTY3NV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mNzk1ZTFkNTcyZjI0N2E0ODQxM2FiMzQ0YTJmNzYwZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9lZWI5YmU2OWRlYjU0MzJhODllMWJiMTQ0MzZkNzgxMiA9ICQoJzxkaXYgaWQ9Imh0bWxfZWViOWJlNjlkZWI1NDMyYTg5ZTFiYjE0NDM2ZDc4MTIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlJpY2htb25kIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZjc5NWUxZDU3MmYyNDdhNDg0MTNhYjM0NGEyZjc2MGUuc2V0Q29udGVudChodG1sX2VlYjliZTY5ZGViNTQzMmE4OWUxYmIxNDQzNmQ3ODEyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzIzZTk3OTQyNDk2NTQ4ZDFhMGE5ZmM0NDc4OGJlZmNhLmJpbmRQb3B1cChwb3B1cF9mNzk1ZTFkNTcyZjI0N2E0ODQxM2FiMzQ0YTJmNzYwZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wN2Q3YTc1NDNmMmE0ZmM0ODEyYmNlNjc1MzQzOWYzYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0MDgxNTcsLTc5LjM4MTc1MjI5OTk5OTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzcwYjU4YjcxNThlZjQxOGVhOGVhOGRlYTllZTA1MzEwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzIwNmYwNGMxOWIwMTQ2ZjZhNzdlMGNlODZkNjQ5NzJiID0gJCgnPGRpdiBpZD0iaHRtbF8yMDZmMDRjMTliMDE0NmY2YTc3ZTBjZTg2ZDY0OTcyYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SGFyYm91cmZyb250IEVhc3QgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83MGI1OGI3MTU4ZWY0MThlYThlYThkZWE5ZWUwNTMxMC5zZXRDb250ZW50KGh0bWxfMjA2ZjA0YzE5YjAxNDZmNmE3N2UwY2U4NmQ2NDk3MmIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMDdkN2E3NTQzZjJhNGZjNDgxMmJjZTY3NTM0MzlmM2IuYmluZFBvcHVwKHBvcHVwXzcwYjU4YjcxNThlZjQxOGVhOGVhOGRlYTllZTA1MzEwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2U5NzFiNWFjN2M1NjQyYWE5ZWU1ZTcwYmVmNTEwOTUyID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfOTgxMTU3MjFiNmE1NGUxOWI1MTBkODQwNDJlMGY5NTUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZjAwZGQ3NGE5MjgxNDEzNzhlN2FlOGNhYTVhOGQ4YmMgPSAkKCc8ZGl2IGlkPSJodG1sX2YwMGRkNzRhOTI4MTQxMzc4ZTdhZThjYWE1YThkOGJjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Ub3JvbnRvIElzbGFuZHMgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF85ODExNTcyMWI2YTU0ZTE5YjUxMGQ4NDA0MmUwZjk1NS5zZXRDb250ZW50KGh0bWxfZjAwZGQ3NGE5MjgxNDEzNzhlN2FlOGNhYTVhOGQ4YmMpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZTk3MWI1YWM3YzU2NDJhYTllZTVlNzBiZWY1MTA5NTIuYmluZFBvcHVwKHBvcHVwXzk4MTE1NzIxYjZhNTRlMTliNTEwZDg0MDQyZTBmOTU1KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzZjYThhZTZiNjc5NTQyZjdiYzRmZWE0YThmODE3Nzk4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQwODE1NywtNzkuMzgxNzUyMjk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzVjM2JiMDhmZTlkNDk0N2JlNDVlNDJkZmE2ZGM5MzcgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfOGIxMTMwN2Y0YTMzNDE2OThlNWUzNmFlMjZjMjFkYmUgPSAkKCc8ZGl2IGlkPSJodG1sXzhiMTEzMDdmNGEzMzQxNjk4ZTVlMzZhZTI2YzIxZGJlIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5VbmlvbiBTdGF0aW9uIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzVjM2JiMDhmZTlkNDk0N2JlNDVlNDJkZmE2ZGM5Mzcuc2V0Q29udGVudChodG1sXzhiMTEzMDdmNGEzMzQxNjk4ZTVlMzZhZTI2YzIxZGJlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzZjYThhZTZiNjc5NTQyZjdiYzRmZWE0YThmODE3Nzk4LmJpbmRQb3B1cChwb3B1cF83NWMzYmIwOGZlOWQ0OTQ3YmU0NWU0MmRmYTZkYzkzNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zZTUyYzIyMTVkY2Y0ZjcyYTdkNWFmNTMzYTUyOWY4YSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2NkMjc4Y2VhZGYxNzQ3NGViYzkwMjZiYjIyYTAzOThjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzgxM2M3MGYyMGE5MDRlYTc4YjUwYjhiOThkMjYzODRlID0gJCgnPGRpdiBpZD0iaHRtbF84MTNjNzBmMjBhOTA0ZWE3OGI1MGI4Yjk4ZDI2Mzg0ZSIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+RGVzaWduIEV4Y2hhbmdlIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2QyNzhjZWFkZjE3NDc0ZWJjOTAyNmJiMjJhMDM5OGMuc2V0Q29udGVudChodG1sXzgxM2M3MGYyMGE5MDRlYTc4YjUwYjhiOThkMjYzODRlKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzNlNTJjMjIxNWRjZjRmNzJhN2Q1YWY1MzNhNTI5ZjhhLmJpbmRQb3B1cChwb3B1cF9jZDI3OGNlYWRmMTc0NzRlYmM5MDI2YmIyMmEwMzk4Yyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8wZjkzNTM1ZTI3ODI0MGIzOTYwOGI5MDZjZDZmYWY2NiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0NzE3NjgsLTc5LjM4MTU3NjQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzdlZjAwMWVhODI1ZTRjOTViNTA4ZjVjYzFhZjk2MGM3ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzUxOWIwZWZjOWIyZjQ5YTA4ZTA4YTZhZTkyNTBjYTViID0gJCgnPGRpdiBpZD0iaHRtbF81MTliMGVmYzliMmY0OWEwOGUwOGE2YWU5MjUwY2E1YiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VG9yb250byBEb21pbmlvbiBDZW50cmUgQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF83ZWYwMDFlYTgyNWU0Yzk1YjUwOGY1Y2MxYWY5NjBjNy5zZXRDb250ZW50KGh0bWxfNTE5YjBlZmM5YjJmNDlhMDhlMDhhNmFlOTI1MGNhNWIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMGY5MzUzNWUyNzgyNDBiMzk2MDhiOTA2Y2Q2ZmFmNjYuYmluZFBvcHVwKHBvcHVwXzdlZjAwMWVhODI1ZTRjOTViNTA4ZjVjYzFhZjk2MGM3KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzBkNmIyNGU2Njg2NzQyMWZiMzUyNTE2M2E2NDU1Y2I0ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ4MTk4NSwtNzkuMzc5ODE2OTAwMDAwMDFdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYzA2ZTkyZDc5MDBkNDI5MWIxYmNhN2ZiMjNiNWUxZjUgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYTFlOWJlZmU3ZDBmNDQzYWIzNGY1ZmQ2MDU1ZWMwNmMgPSAkKCc8ZGl2IGlkPSJodG1sX2ExZTliZWZlN2QwZjQ0M2FiMzRmNWZkNjA1NWVjMDZjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Db21tZXJjZSBDb3VydCBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2MwNmU5MmQ3OTAwZDQyOTFiMWJjYTdmYjIzYjVlMWY1LnNldENvbnRlbnQoaHRtbF9hMWU5YmVmZTdkMGY0NDNhYjM0ZjVmZDYwNTVlYzA2Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8wZDZiMjRlNjY4Njc0MjFmYjM1MjUxNjNhNjQ1NWNiNC5iaW5kUG9wdXAocG9wdXBfYzA2ZTkyZDc5MDBkNDI5MWIxYmNhN2ZiMjNiNWUxZjUpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNzI2NmI2ODNjMzkzNDY5NDg0Nzc1YTQ4ZTRkYTg1OTkgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDgxOTg1LC03OS4zNzk4MTY5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF84N2JjYmIzMDM5ZTU0MTcwOTRkZDU4MjRlZDNkNTQzZSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8xOGUxZDI4Y2I1ZmU0NjA1YmQ1OGJlZjQzMjBmODk2MiA9ICQoJzxkaXYgaWQ9Imh0bWxfMThlMWQyOGNiNWZlNDYwNWJkNThiZWY0MzIwZjg5NjIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlZpY3RvcmlhIEhvdGVsIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfODdiY2JiMzAzOWU1NDE3MDk0ZGQ1ODI0ZWQzZDU0M2Uuc2V0Q29udGVudChodG1sXzE4ZTFkMjhjYjVmZTQ2MDViZDU4YmVmNDMyMGY4OTYyKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzcyNjZiNjgzYzM5MzQ2OTQ4NDc3NWE0OGU0ZGE4NTk5LmJpbmRQb3B1cChwb3B1cF84N2JjYmIzMDM5ZTU0MTcwOTRkZDU4MjRlZDNkNTQzZSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yZDliYTRjNGVmMzY0ODJjYWFkYWI1ZWRhY2E2MmRiNCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNzM3Yjk3N2I3OTVlNGJmMjlhODYxYzdmM2U4N2MzYzIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfZDMzODRlY2ZjMmNmNGRkMWJmZWI1YjgyZTQxODIzODggPSAkKCc8ZGl2IGlkPSJodG1sX2QzMzg0ZWNmYzJjZjRkZDFiZmViNWI4MmU0MTgyMzg4IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5IYXJib3JkIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNzM3Yjk3N2I3OTVlNGJmMjlhODYxYzdmM2U4N2MzYzIuc2V0Q29udGVudChodG1sX2QzMzg0ZWNmYzJjZjRkZDFiZmViNWI4MmU0MTgyMzg4KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzJkOWJhNGM0ZWYzNjQ4MmNhYWRhYjVlZGFjYTYyZGI0LmJpbmRQb3B1cChwb3B1cF83MzdiOTc3Yjc5NWU0YmYyOWE4NjFjN2YzZTg3YzNjMik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9kNzFiNmYzMTgzNTA0MGViYTRhMmRjNzZjOGU0MjFjOSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2MjY5NTYsLTc5LjQwMDA0OTNdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzFjN2E4MjNmZWYwNDU4ODkzMWVhNTRjYWU5OGVjNjIgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMjNlYzUxNDFkZmIyNGVhNzg0YTNlMzMyMGZjMWNlN2MgPSAkKCc8ZGl2IGlkPSJodG1sXzIzZWM1MTQxZGZiMjRlYTc4NGEzZTMzMjBmYzFjZTdjIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Vbml2ZXJzaXR5IG9mIFRvcm9udG8gQ2x1c3RlciAwPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8zMWM3YTgyM2ZlZjA0NTg4OTMxZWE1NGNhZTk4ZWM2Mi5zZXRDb250ZW50KGh0bWxfMjNlYzUxNDFkZmIyNGVhNzg0YTNlMzMyMGZjMWNlN2MpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfZDcxYjZmMzE4MzUwNDBlYmE0YTJkYzc2YzhlNDIxYzkuYmluZFBvcHVwKHBvcHVwXzMxYzdhODIzZmVmMDQ1ODg5MzFlYTU0Y2FlOThlYzYyKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzY4MjIzODFlOWNjYjQzNmE4ODYyNTA4ZGY3YjVkNDU4ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9iMTk4NGJhNDVmZjI0OWIzYjcwOWQyYTJlZGU4ZTg3YyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81OTY3MjJiZmZkZTM0MWRjYjRhYWM1NjM2OWQ0MmVkNCA9ICQoJzxkaXYgaWQ9Imh0bWxfNTk2NzIyYmZmZGUzNDFkY2I0YWFjNTYzNjlkNDJlZDQiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNoaW5hdG93biBDbHVzdGVyIDM8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2IxOTg0YmE0NWZmMjQ5YjNiNzA5ZDJhMmVkZThlODdjLnNldENvbnRlbnQoaHRtbF81OTY3MjJiZmZkZTM0MWRjYjRhYWM1NjM2OWQ0MmVkNCk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82ODIyMzgxZTljY2I0MzZhODg2MjUwOGRmN2I1ZDQ1OC5iaW5kUG9wdXAocG9wdXBfYjE5ODRiYTQ1ZmYyNDliM2I3MDlkMmEyZWRlOGU4N2MpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfNjQ1MWMzMTI1NDMwNDgyNWJhZmE3MGMzOTEzY2EyZTAgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NTMyMDU3LC03OS40MDAwNDkzXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MGZmYjQiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODBmZmI0IiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2ZhYzc1ODJiMTYzNzQ4ZGI4NGQ2NWVjNzE2YzViOGIwID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzMyY2Y4MjQ5ZDdiYTQ2MmQ4MGQ5YjM5ODg2ZDY3YzY3ID0gJCgnPGRpdiBpZD0iaHRtbF8zMmNmODI0OWQ3YmE0NjJkODBkOWIzOTg4NmQ2N2M2NyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+R3JhbmdlIFBhcmsgQ2x1c3RlciAzPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF9mYWM3NTgyYjE2Mzc0OGRiODRkNjVlYzcxNmM1YjhiMC5zZXRDb250ZW50KGh0bWxfMzJjZjgyNDlkN2JhNDYyZDgwZDliMzk4ODZkNjdjNjcpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfNjQ1MWMzMTI1NDMwNDgyNWJhZmE3MGMzOTEzY2EyZTAuYmluZFBvcHVwKHBvcHVwX2ZhYzc1ODJiMTYzNzQ4ZGI4NGQ2NWVjNzE2YzViOGIwKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyXzQ3ZmRhNjcxOGRlYjQ2ZWJiNDI0ZjFhMTcxODQ0OWVhID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjUzMjA1NywtNzkuNDAwMDQ5M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjODBmZmI0IiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzgwZmZiNCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF83ZmI5YTEzODM0MmU0ZWI3OGRkZTcxMjdlNDQ5NGI3ZCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF81MTgyMTA3N2ZiZjY0YjQ2ODVhNzc1NDE2NzhjZjY1ZiA9ICQoJzxkaXYgaWQ9Imh0bWxfNTE4MjEwNzdmYmY2NGI0Njg1YTc3NTQxNjc4Y2Y2NWYiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPktlbnNpbmd0b24gTWFya2V0IENsdXN0ZXIgMzwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfN2ZiOWExMzgzNDJlNGViNzhkZGU3MTI3ZTQ0OTRiN2Quc2V0Q29udGVudChodG1sXzUxODIxMDc3ZmJmNjRiNDY4NWE3NzU0MTY3OGNmNjVmKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzQ3ZmRhNjcxOGRlYjQ2ZWJiNDI0ZjFhMTcxODQ0OWVhLmJpbmRQb3B1cChwb3B1cF83ZmI5YTEzODM0MmU0ZWI3OGRkZTcxMjdlNDQ5NGI3ZCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8zNDA2YTlmODU0YTk0ZjJlOTdlNmYzNjIzOWI1ODY2NSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZjEwMTFiMGFmMDU2NDc3YzllMzNiODY4NTJkMGRjZTAgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfNWQyYThlMWM5MjRlNGQwZmJjNGRkNzFjYjFiZTdkZWIgPSAkKCc8ZGl2IGlkPSJodG1sXzVkMmE4ZTFjOTI0ZTRkMGZiYzRkZDcxY2IxYmU3ZGViIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5DTiBUb3dlciBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2YxMDExYjBhZjA1NjQ3N2M5ZTMzYjg2ODUyZDBkY2UwLnNldENvbnRlbnQoaHRtbF81ZDJhOGUxYzkyNGU0ZDBmYmM0ZGQ3MWNiMWJlN2RlYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl8zNDA2YTlmODU0YTk0ZjJlOTdlNmYzNjIzOWI1ODY2NS5iaW5kUG9wdXAocG9wdXBfZjEwMTFiMGFmMDU2NDc3YzllMzNiODY4NTJkMGRjZTApOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfOTc1YzFmOTkzOGMxNGY2ZmI5ZmZkMmQ2Y2ZkODI0YTQgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzM3NzUwM2I3ZWNiZTQyNjM4ZDQ2NzQ4NTFhMzI4MDYxID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzdjOTY1NTc1MDc1OTQ5NzU5YzM5ZGRhMDFlZDQ1MzBiID0gJCgnPGRpdiBpZD0iaHRtbF83Yzk2NTU3NTA3NTk0OTc1OWMzOWRkYTAxZWQ0NTMwYiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+QmF0aHVyc3QgUXVheSBDbHVzdGVyIDI8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzM3NzUwM2I3ZWNiZTQyNjM4ZDQ2NzQ4NTFhMzI4MDYxLnNldENvbnRlbnQoaHRtbF83Yzk2NTU3NTA3NTk0OTc1OWMzOWRkYTAxZWQ0NTMwYik7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl85NzVjMWY5OTM4YzE0ZjZmYjlmZmQyZDZjZmQ4MjRhNC5iaW5kUG9wdXAocG9wdXBfMzc3NTAzYjdlY2JlNDI2MzhkNDY3NDg1MWEzMjgwNjEpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfM2IwYzdkNzZkZmZjNDZiOTg5MjIxY2UwOTNmNjc3Y2MgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42Mjg5NDY3LC03OS4zOTQ0MTk5XSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiMwMGI1ZWIiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjMDBiNWViIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzBhMmE4ZGYwZDQwYTQ5NTliNWRmNjMxMGZkZWYyM2RjID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzc1Y2UyNTFhN2QxOTRmOGFhZGZlMDdiMDQzNzNjMjRmID0gJCgnPGRpdiBpZD0iaHRtbF83NWNlMjUxYTdkMTk0ZjhhYWRmZTA3YjA0MzczYzI0ZiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+SXNsYW5kIGFpcnBvcnQgQ2x1c3RlciAyPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF8wYTJhOGRmMGQ0MGE0OTU5YjVkZjYzMTBmZGVmMjNkYy5zZXRDb250ZW50KGh0bWxfNzVjZTI1MWE3ZDE5NGY4YWFkZmUwN2IwNDM3M2MyNGYpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfM2IwYzdkNzZkZmZjNDZiOTg5MjIxY2UwOTNmNjc3Y2MuYmluZFBvcHVwKHBvcHVwXzBhMmE4ZGYwZDQwYTQ5NTliNWRmNjMxMGZkZWYyM2RjKTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2MxZTI5YjlkZWFlODQ5NWRhMTljZjk4YjA5MDJiZDc2ID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjI4OTQ2NywtNzkuMzk0NDE5OV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjMDBiNWViIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiIzAwYjVlYiIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF80NWZlYWU3OWUzNTk0ODY3YWU3Y2ViNjViOWI1ZTgwNyA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF9iMTk3Y2Y1NzNiZGM0MTM1OGVmODAyMmZmMDM5Zjk5YiA9ICQoJzxkaXYgaWQ9Imh0bWxfYjE5N2NmNTczYmRjNDEzNThlZjgwMjJmZjAzOWY5OWIiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkhhcmJvdXJmcm9udCBXZXN0IENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNDVmZWFlNzllMzU5NDg2N2FlN2NlYjY1YjliNWU4MDcuc2V0Q29udGVudChodG1sX2IxOTdjZjU3M2JkYzQxMzU4ZWY4MDIyZmYwMzlmOTliKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2MxZTI5YjlkZWFlODQ5NWRhMTljZjk4YjA5MDJiZDc2LmJpbmRQb3B1cChwb3B1cF80NWZlYWU3OWUzNTk0ODY3YWU3Y2ViNjViOWI1ZTgwNyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hZDA2OWFiZDk2YzA0OWJlYjkzY2UzZDliOGEzNTZkMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfMzAzODk1ZmZlMmU2NGJkZjgyZDE5MmQ0ZDIwMDYyZjggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfYzdiMzM0YWQxMGQ3NDY5YjhlZGY4MWM3ZGFhYmRiNTkgPSAkKCc8ZGl2IGlkPSJodG1sX2M3YjMzNGFkMTBkNzQ2OWI4ZWRmODFjN2RhYWJkYjU5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5LaW5nIGFuZCBTcGFkaW5hIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfMzAzODk1ZmZlMmU2NGJkZjgyZDE5MmQ0ZDIwMDYyZjguc2V0Q29udGVudChodG1sX2M3YjMzNGFkMTBkNzQ2OWI4ZWRmODFjN2RhYWJkYjU5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2FkMDY5YWJkOTZjMDQ5YmViOTNjZTNkOWI4YTM1NmQzLmJpbmRQb3B1cChwb3B1cF8zMDM4OTVmZmUyZTY0YmRmODJkMTkyZDRkMjAwNjJmOCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9mNDU3NTRmZWRiZjQ0MjY0YWY2NjJlZDU2M2ExNzM3OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfNGNmYzFkNmE1MmIxNDQ5ZmJkNWFjNzEzMWFjZDQzMGMgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWEyNGExYmE0YjdjNDY2NTg1MDI4MjBlNmVlMzBjNzQgPSAkKCc8ZGl2IGlkPSJodG1sXzFhMjRhMWJhNGI3YzQ2NjU4NTAyODIwZTZlZTMwYzc0IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5SYWlsd2F5IExhbmRzIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfNGNmYzFkNmE1MmIxNDQ5ZmJkNWFjNzEzMWFjZDQzMGMuc2V0Q29udGVudChodG1sXzFhMjRhMWJhNGI3YzQ2NjU4NTAyODIwZTZlZTMwYzc0KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2Y0NTc1NGZlZGJmNDQyNjRhZjY2MmVkNTYzYTE3Mzc4LmJpbmRQb3B1cChwb3B1cF80Y2ZjMWQ2YTUyYjE0NDlmYmQ1YWM3MTMxYWNkNDMwYyk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9lZDVkMzQzMjFkYjU0YzBmOTkwZDlmNzNhM2VlYWFjMSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjYyODk0NjcsLTc5LjM5NDQxOTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiIzAwYjVlYiIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiMwMGI1ZWIiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfYjdkZmZiYzQwN2E0NGNhZGIyZGY0NjQwYjc4MWZlMjQgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMzViNzY5YjVhZTE2NDM3NjhmZWRmZjVlOGMzZDkzOWIgPSAkKCc8ZGl2IGlkPSJodG1sXzM1Yjc2OWI1YWUxNjQzNzY4ZmVkZmY1ZThjM2Q5MzliIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5Tb3V0aCBOaWFnYXJhIENsdXN0ZXIgMjwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfYjdkZmZiYzQwN2E0NGNhZGIyZGY0NjQwYjc4MWZlMjQuc2V0Q29udGVudChodG1sXzM1Yjc2OWI1YWUxNjQzNzY4ZmVkZmY1ZThjM2Q5MzliKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2VkNWQzNDMyMWRiNTRjMGY5OTBkOWY3M2EzZWVhYWMxLmJpbmRQb3B1cChwb3B1cF9iN2RmZmJjNDA3YTQ0Y2FkYjJkZjQ2NDBiNzgxZmUyNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl8yNWM4MjljYWFiZGI0ZDZmOTM4ZGQzNDViMTc0OTlmMyA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY3OTU2MjYsLTc5LjM3NzUyOTQwMDAwMDAxXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiM4MDAwZmYiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjODAwMGZmIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwXzY1MjBkYWZjY2JmNjRmYmU5ZWMwZjljMjZiNzI3YWQ5ID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sXzQwOGU5ZDE4Nzk4YzRlMmZhNWVlY2EyMGU5YjlmZTcyID0gJCgnPGRpdiBpZD0iaHRtbF80MDhlOWQxODc5OGM0ZTJmYTVlZWNhMjBlOWI5ZmU3MiIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+Um9zZWRhbGUgQ2x1c3RlciAxPC9kaXY+JylbMF07CiAgICAgICAgICAgICAgICBwb3B1cF82NTIwZGFmY2NiZjY0ZmJlOWVjMGY5YzI2YjcyN2FkOS5zZXRDb250ZW50KGh0bWxfNDA4ZTlkMTg3OThjNGUyZmE1ZWVjYTIwZTliOWZlNzIpOwogICAgICAgICAgICAKCiAgICAgICAgICAgIGNpcmNsZV9tYXJrZXJfMjVjODI5Y2FhYmRiNGQ2ZjkzOGRkMzQ1YjE3NDk5ZjMuYmluZFBvcHVwKHBvcHVwXzY1MjBkYWZjY2JmNjRmYmU5ZWMwZjljMjZiNzI3YWQ5KTsKCiAgICAgICAgICAgIAogICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBjaXJjbGVfbWFya2VyX2ViNDA0NDI5OTNhNzRlNDNiNzE2MmRlMzBkMWY0MjVjID0gTC5jaXJjbGVNYXJrZXIoCiAgICAgICAgICAgICAgICBbNDMuNjQ2NDM1MiwtNzkuMzc0ODQ1OTk5OTk5OTldLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfZmY5NDI3YzUxNGUyNGMzZWI3MTA0OWM3ODc4Y2Q5ZjYgPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMWFlOTAyNGY1ZTBkNGNhNTg4ZDk1ODA0MTM2MTRkOTkgPSAkKCc8ZGl2IGlkPSJodG1sXzFhZTkwMjRmNWUwZDRjYTU4OGQ5NTgwNDEzNjE0ZDk5IiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5TdG4gQSBQTyBCb3hlcyAyNSBUaGUgRXNwbGFuYWRlIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmY5NDI3YzUxNGUyNGMzZWI3MTA0OWM3ODc4Y2Q5ZjYuc2V0Q29udGVudChodG1sXzFhZTkwMjRmNWUwZDRjYTU4OGQ5NTgwNDEzNjE0ZDk5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2ViNDA0NDI5OTNhNzRlNDNiNzE2MmRlMzBkMWY0MjVjLmJpbmRQb3B1cChwb3B1cF9mZjk0MjdjNTE0ZTI0YzNlYjcxMDQ5Yzc4NzhjZDlmNik7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl9hMzZhOGViYzBjMDQ0ZDAwOGZhZDkyODBmYjBjNmYzYiA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF85MDMwZjVlZjFmZWE0ZGQ2YjM4MzZjOGM3NGU2MGUxNCA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF8yZjkwZjM5MGRkZTA0Yjg3ODc4YWFkYWQzNTkxODFkOSA9ICQoJzxkaXYgaWQ9Imh0bWxfMmY5MGYzOTBkZGUwNGI4Nzg3OGFhZGFkMzU5MTgxZDkiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNhYmJhZ2V0b3duIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfOTAzMGY1ZWYxZmVhNGRkNmIzODM2YzhjNzRlNjBlMTQuc2V0Q29udGVudChodG1sXzJmOTBmMzkwZGRlMDRiODc4NzhhYWRhZDM1OTE4MWQ5KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2EzNmE4ZWJjMGMwNDRkMDA4ZmFkOTI4MGZiMGM2ZjNiLmJpbmRQb3B1cChwb3B1cF85MDMwZjVlZjFmZWE0ZGQ2YjM4MzZjOGM3NGU2MGUxNCk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl80ZmQyYWExYjU2MDA0M2ZhYThiZjdjZWRiNDE2Mzk2OCA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY2Nzk2NywtNzkuMzY3Njc1M10sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9jYzk3YTdkODhkZWE0OTM2ODc1ZWY0YjBjYWI5ZTZiNSA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF80NmExMThkYTEzOGU0YzM5YWVmYTVhOGI4ZjdiMTFhMyA9ICQoJzxkaXYgaWQ9Imh0bWxfNDZhMTE4ZGExMzhlNGMzOWFlZmE1YThiOGY3YjExYTMiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPlN0LiBKYW1lcyBUb3duIENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfY2M5N2E3ZDg4ZGVhNDkzNjg3NWVmNGIwY2FiOWU2YjUuc2V0Q29udGVudChodG1sXzQ2YTExOGRhMTM4ZTRjMzlhZWZhNWE4YjhmN2IxMWEzKTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyXzRmZDJhYTFiNTYwMDQzZmFhOGJmN2NlZGI0MTYzOTY4LmJpbmRQb3B1cChwb3B1cF9jYzk3YTdkODhkZWE0OTM2ODc1ZWY0YjBjYWI5ZTZiNSk7CgogICAgICAgICAgICAKICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgY2lyY2xlX21hcmtlcl82NWVlNWM5NzIxY2M0ODUzOTNiNmZlOTk0YjQxYTFmYSA9IEwuY2lyY2xlTWFya2VyKAogICAgICAgICAgICAgICAgWzQzLjY0ODQyOTIsLTc5LjM4MjI4MDJdLAogICAgICAgICAgICAgICAgewogICJidWJibGluZ01vdXNlRXZlbnRzIjogdHJ1ZSwKICAiY29sb3IiOiAiI2ZmMDAwMCIsCiAgImRhc2hBcnJheSI6IG51bGwsCiAgImRhc2hPZmZzZXQiOiBudWxsLAogICJmaWxsIjogdHJ1ZSwKICAiZmlsbENvbG9yIjogIiNmZjAwMDAiLAogICJmaWxsT3BhY2l0eSI6IDAuNywKICAiZmlsbFJ1bGUiOiAiZXZlbm9kZCIsCiAgImxpbmVDYXAiOiAicm91bmQiLAogICJsaW5lSm9pbiI6ICJyb3VuZCIsCiAgIm9wYWNpdHkiOiAxLjAsCiAgInJhZGl1cyI6IDUsCiAgInN0cm9rZSI6IHRydWUsCiAgIndlaWdodCI6IDMKfQogICAgICAgICAgICAgICAgKS5hZGRUbyhtYXBfMzY2MWI2NDc2MzUyNDU2NjljOWMyYjAzODg2MmMwODQpOwogICAgICAgICAgICAKICAgIAogICAgICAgICAgICB2YXIgcG9wdXBfODg0NjU0Y2U2MGVhNGM2YjhiYzM4YmNiN2JhYTYxMzggPSBMLnBvcHVwKHttYXhXaWR0aDogJzMwMCd9KTsKCiAgICAgICAgICAgIAogICAgICAgICAgICAgICAgdmFyIGh0bWxfMGVlMDZhYzc3NzQwNGJjZWJhZWY2YmVmNTlmN2VmN2EgPSAkKCc8ZGl2IGlkPSJodG1sXzBlZTA2YWM3Nzc0MDRiY2ViYWVmNmJlZjU5ZjdlZjdhIiBzdHlsZT0id2lkdGg6IDEwMC4wJTsgaGVpZ2h0OiAxMDAuMCU7Ij5GaXJzdCBDYW5hZGlhbiBQbGFjZSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwXzg4NDY1NGNlNjBlYTRjNmI4YmMzOGJjYjdiYWE2MTM4LnNldENvbnRlbnQoaHRtbF8wZWUwNmFjNzc3NDA0YmNlYmFlZjZiZWY1OWY3ZWY3YSk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl82NWVlNWM5NzIxY2M0ODUzOTNiNmZlOTk0YjQxYTFmYS5iaW5kUG9wdXAocG9wdXBfODg0NjU0Y2U2MGVhNGM2YjhiYzM4YmNiN2JhYTYxMzgpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfYjkwN2MwNTZlZWY1NGYyZmEwYTEwOWU4ZjZjNzAyMWEgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NDg0MjkyLC03OS4zODIyODAyXSwKICAgICAgICAgICAgICAgIHsKICAiYnViYmxpbmdNb3VzZUV2ZW50cyI6IHRydWUsCiAgImNvbG9yIjogIiNmZjAwMDAiLAogICJkYXNoQXJyYXkiOiBudWxsLAogICJkYXNoT2Zmc2V0IjogbnVsbCwKICAiZmlsbCI6IHRydWUsCiAgImZpbGxDb2xvciI6ICIjZmYwMDAwIiwKICAiZmlsbE9wYWNpdHkiOiAwLjcsCiAgImZpbGxSdWxlIjogImV2ZW5vZGQiLAogICJsaW5lQ2FwIjogInJvdW5kIiwKICAibGluZUpvaW4iOiAicm91bmQiLAogICJvcGFjaXR5IjogMS4wLAogICJyYWRpdXMiOiA1LAogICJzdHJva2UiOiB0cnVlLAogICJ3ZWlnaHQiOiAzCn0KICAgICAgICAgICAgICAgICkuYWRkVG8obWFwXzM2NjFiNjQ3NjM1MjQ1NjY5YzljMmIwMzg4NjJjMDg0KTsKICAgICAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIHBvcHVwX2RkODA1Y2ZkMzMzNjQwNzM4M2RiN2Y5MWI5MTc0MjQzID0gTC5wb3B1cCh7bWF4V2lkdGg6ICczMDAnfSk7CgogICAgICAgICAgICAKICAgICAgICAgICAgICAgIHZhciBodG1sX2I2ZmQ2M2U0NGU4MjQyNWJiZDhjYTQ1NTBiNWY1OTVjID0gJCgnPGRpdiBpZD0iaHRtbF9iNmZkNjNlNDRlODI0MjViYmQ4Y2E0NTUwYjVmNTk1YyIgc3R5bGU9IndpZHRoOiAxMDAuMCU7IGhlaWdodDogMTAwLjAlOyI+VW5kZXJncm91bmQgY2l0eSBDbHVzdGVyIDA8L2Rpdj4nKVswXTsKICAgICAgICAgICAgICAgIHBvcHVwX2RkODA1Y2ZkMzMzNjQwNzM4M2RiN2Y5MWI5MTc0MjQzLnNldENvbnRlbnQoaHRtbF9iNmZkNjNlNDRlODI0MjViYmQ4Y2E0NTUwYjVmNTk1Yyk7CiAgICAgICAgICAgIAoKICAgICAgICAgICAgY2lyY2xlX21hcmtlcl9iOTA3YzA1NmVlZjU0ZjJmYTBhMTA5ZThmNmM3MDIxYS5iaW5kUG9wdXAocG9wdXBfZGQ4MDVjZmQzMzM2NDA3MzgzZGI3ZjkxYjkxNzQyNDMpOwoKICAgICAgICAgICAgCiAgICAgICAgCiAgICAKICAgICAgICAgICAgdmFyIGNpcmNsZV9tYXJrZXJfZDM4NGIxNDRiODRiNDg0ZWI5ZDhiN2FhYzFmMmQ0NWUgPSBMLmNpcmNsZU1hcmtlcigKICAgICAgICAgICAgICAgIFs0My42NjU4NTk5LC03OS4zODMxNTk5MDAwMDAwMV0sCiAgICAgICAgICAgICAgICB7CiAgImJ1YmJsaW5nTW91c2VFdmVudHMiOiB0cnVlLAogICJjb2xvciI6ICIjZmYwMDAwIiwKICAiZGFzaEFycmF5IjogbnVsbCwKICAiZGFzaE9mZnNldCI6IG51bGwsCiAgImZpbGwiOiB0cnVlLAogICJmaWxsQ29sb3IiOiAiI2ZmMDAwMCIsCiAgImZpbGxPcGFjaXR5IjogMC43LAogICJmaWxsUnVsZSI6ICJldmVub2RkIiwKICAibGluZUNhcCI6ICJyb3VuZCIsCiAgImxpbmVKb2luIjogInJvdW5kIiwKICAib3BhY2l0eSI6IDEuMCwKICAicmFkaXVzIjogNSwKICAic3Ryb2tlIjogdHJ1ZSwKICAid2VpZ2h0IjogMwp9CiAgICAgICAgICAgICAgICApLmFkZFRvKG1hcF8zNjYxYjY0NzYzNTI0NTY2OWM5YzJiMDM4ODYyYzA4NCk7CiAgICAgICAgICAgIAogICAgCiAgICAgICAgICAgIHZhciBwb3B1cF9mZGJlNWVlOWMwNTc0ZGMxYWY4YmE0YmE4NTVjZjQ4MiA9IEwucG9wdXAoe21heFdpZHRoOiAnMzAwJ30pOwoKICAgICAgICAgICAgCiAgICAgICAgICAgICAgICB2YXIgaHRtbF83YTNjYWViNzFmZjU0MzE4OWMxZGMxODc5MzA1M2VlNSA9ICQoJzxkaXYgaWQ9Imh0bWxfN2EzY2FlYjcxZmY1NDMxODljMWRjMTg3OTMwNTNlZTUiIHN0eWxlPSJ3aWR0aDogMTAwLjAlOyBoZWlnaHQ6IDEwMC4wJTsiPkNodXJjaCBhbmQgV2VsbGVzbGV5IENsdXN0ZXIgMDwvZGl2PicpWzBdOwogICAgICAgICAgICAgICAgcG9wdXBfZmRiZTVlZTljMDU3NGRjMWFmOGJhNGJhODU1Y2Y0ODIuc2V0Q29udGVudChodG1sXzdhM2NhZWI3MWZmNTQzMTg5YzFkYzE4NzkzMDUzZWU1KTsKICAgICAgICAgICAgCgogICAgICAgICAgICBjaXJjbGVfbWFya2VyX2QzODRiMTQ0Yjg0YjQ4NGViOWQ4YjdhYWMxZjJkNDVlLmJpbmRQb3B1cChwb3B1cF9mZGJlNWVlOWMwNTc0ZGMxYWY4YmE0YmE4NTVjZjQ4Mik7CgogICAgICAgICAgICAKICAgICAgICAKPC9zY3JpcHQ+" style="position:absolute;width:100%;height:100%;left:0;top:0;border:none !important;" allowfullscreen webkitallowfullscreen mozallowfullscreen></iframe></div></div>



## Thank You for Reviewing my work!!


```python

```
